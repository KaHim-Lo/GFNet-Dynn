import torch
import torch.nn as nn
from enum import Enum

from tools.conformal_eedn import early_exit_conf_sets


class LossContributionMode(Enum):
    SINGLE = 'single' # a sample contributes to the loss at a single classifier
    WEIGHTED = 'weighted'
    BOOSTED = 'boosted'


class InvalidLossContributionModeException(Exception):
    pass

class ClassifierTrainingHelper:
    def __init__(self, net: nn.Module, loss_contribution_mode: LossContributionMode,G: int, device) -> None:
        self.net = net
        self.device = device
        self.loss_contribution_mode = loss_contribution_mode
        self.set_fractions([1/G for _ in range(G)])
        self.alpha_qhat_dict = None
        # boosted loss behaves as weighted
        if self.loss_contribution_mode == LossContributionMode.BOOSTED: 
            self.loss_contribution_mode = LossContributionMode.WEIGHTED
        
        if self.loss_contribution_mode == LossContributionMode.WEIGHTED:
            self.classifier_criterion = nn.CrossEntropyLoss(reduction='none')
        elif self.loss_contribution_mode == LossContributionMode.SINGLE:
            self.classifier_criterion = nn.CrossEntropyLoss()
         
    def set_conf_thresholds(self, alpha_qhat_dict):
        self.alpha_qhat_dict = alpha_qhat_dict
   
    def set_fractions(self, fractions):
        self.fractions = torch.Tensor(fractions).to(self.device)

    def get_loss(self, inputs: torch.Tensor, targets: torch.tensor, compute_hamming = False):
        intermediate_logits = [] # logits from the intermediate classifiers
        num_exits_per_gate = []
        final_head, intermediate_outs = self.net.module.forward_features(inputs)
        final_logits = self.net.module.head(final_head)
        prob_gates = torch.zeros((inputs.shape[0], 1)).to(inputs.device)
        gated_y_logits = torch.zeros_like(final_logits) # holds the accumulated predictions in a single tensor
        sample_exit_level_map = torch.zeros_like(targets) # holds the exit level of each prediction
        past_exits = torch.zeros((inputs.shape[0], 1)).to(inputs.device)

        # lists for weighted mode
        p_exit_at_gate_list = []
        loss_per_gate_list = []
        G = torch.zeros((targets.shape[0], 1)).to(inputs.device) # holds the g's, the sigmoided gate outputs
        for l, intermediate_head in enumerate(self.net.module.intermediate_heads):
            current_logits = intermediate_head(intermediate_outs[l])
            intermediate_logits.append(current_logits)
            # TODO: Freezing the gate can be done in learning helper when we switch phase.
            with torch.no_grad(): # Prevent backpropagation to gates.
                    exit_gate_logit = self.net.module.get_gate_prediction(l, current_logits)
            g = torch.nn.functional.sigmoid(exit_gate_logit) # g
        
            no_exit_previous_gates_prob = torch.prod(1 - prob_gates, axis=1)[:,None] # prod (1-g)
            
            if self.loss_contribution_mode == LossContributionMode.SINGLE:
                current_gate_activation_prob = torch.clip(g/no_exit_previous_gates_prob, min=0, max=1)
            elif self.loss_contribution_mode == LossContributionMode.WEIGHTED:
                sum_previous_gs = torch.sum(G, dim=1)[:, None]
                p_exit_at_gate = torch.max(torch.zeros((targets.shape[0], 1)).to(inputs.device), torch.min(g, 1 - sum_previous_gs))
                p_exit_at_gate_list.append(p_exit_at_gate)
                current_gate_activation_prob = torch.clip(p_exit_at_gate/no_exit_previous_gates_prob, min=0, max=1)
                G = torch.cat((G, g), dim=1)
                loss_at_gate = self.classifier_criterion(current_logits, targets)
                loss_per_gate_list.append(loss_at_gate[:, None])

            prob_gates = torch.cat((prob_gates, current_gate_activation_prob), dim=1) # gate exits are independent so they won't sum to 1 over all cols
            do_exit = current_gate_activation_prob >= 0.5
            current_exit = torch.logical_and(do_exit, torch.logical_not(past_exits))
            current_exit_index = current_exit.flatten().nonzero()
            sample_exit_level_map[current_exit_index] = l
            num_exits_per_gate.append(torch.sum(current_exit))
            # Update past_exists to include the currently exited ones for next iteration
            past_exits = torch.logical_or(current_exit, past_exits)
            # Update early_exit_logits which include all predictions across all layers
            gated_y_logits = gated_y_logits + torch.mul(current_exit, current_logits)
        final_gate_exit = torch.logical_not(past_exits)
        sample_exit_level_map[final_gate_exit.flatten().nonzero()] = len(self.net.module.intermediate_heads)
        num_exits_per_gate.append(torch.sum(final_gate_exit))
        gated_y_logits = gated_y_logits + torch.mul(final_gate_exit, final_logits) # last gate
        things_of_interest = {
            'intermediate_logits':intermediate_logits,
            'final_logits':final_logits,
            'num_exits_per_gate':num_exits_per_gate,
            'gated_y_logits': gated_y_logits,
            'sample_exit_level_map': sample_exit_level_map,
            'gated_y_logits': gated_y_logits}
        if self.loss_contribution_mode == LossContributionMode.WEIGHTED:
            p_exit_at_gate_T = torch.concatenate(p_exit_at_gate_list, dim=1)
            things_of_interest['p_exit_at_gate']=p_exit_at_gate_T
        loss = 0
        if self.loss_contribution_mode == LossContributionMode.SINGLE:
            loss = self._compute_single_loss(gated_y_logits, targets)
        elif self.loss_contribution_mode ==  LossContributionMode.WEIGHTED:
            loss = self._compute_weighted_loss(p_exit_at_gate_list, loss_per_gate_list)
        else:
            raise InvalidLossContributionModeException('Ca marche pas ton affaire')

        things_of_interest = self.add_conformal_predictions(things_of_interest) 
        
        return loss, things_of_interest
    
    def _compute_weighted_loss(self, p_exit_at_gate_list, loss_per_gate_list):
        P = torch.cat(p_exit_at_gate_list, dim = 1)
        L = torch.cat(loss_per_gate_list, dim = 1)
        loss_per_point = torch.sum(L, dim=1) # we want to maintain this
        weighted_loss = P * L
        ratio = (loss_per_point/torch.sum(weighted_loss, dim=1))[:,None]
        loss = torch.mean(weighted_loss * ratio)
        return loss
    
    def _compute_single_loss(self, gated_y_logits, targets):
        return self.classifier_criterion(gated_y_logits, targets)
    
    def add_conformal_predictions(self, things_of_interest):
        
        if self.alpha_qhat_dict is not None:
            # we compute the prediction set from a single threshold computed on all the outputs, regardless of the gate
            gated_logits = things_of_interest['gated_y_logits']
            sample_exit_level_map = things_of_interest['sample_exit_level_map']
            all_logits = things_of_interest['intermediate_logits'] + [things_of_interest['final_logits']]
            dict_sets = early_exit_conf_sets(self.alpha_qhat_dict, sample_exit_level_map, all_logits, gated_logits)
            things_of_interest = dict_sets | things_of_interest
        return things_of_interest

