import torch
import torch.nn as nn
from queue import Queue
import random 
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from models.custom_modules.gate import GateType
from models.custom_modules.learnable_uncertainty_gate import LearnableUncGate
from models.custom_modules.identity_gate import IdentityGate
from sklearn.metrics import accuracy_score
from enum import Enum
from models.gfnet import GFNet
from models.gfnet import Mlp
from tools.metrics_utils import compute_detached_uncertainty_metrics


class TrainingPhase(Enum):
    CLASSIFIER = 1
    GATE = 2
    WARMUP = 3

class IntermediateOutput:
    def __init__(self, level: int, predictions: torch.Tensor, predictions_idx: torch.Tensor, remaining_idx: torch.Tensor):
        self.level = level
        self.predictions = predictions
        self.predictions_idx = predictions_idx
        self.remaining_idx = remaining_idx

class ClassifierAccuracyTracker:
    def __init__(self, level: int, patience: int = 3):
        self.level = level
        self.patience = patience
        self.test_accs = Queue(maxsize=patience)
        self.frozen = False

    def insert_acc(self, test_acc):
        if self.frozen:
            return
        if self.test_accs.qsize() == self.patience:
            removed_acc = self.test_accs.get()
            print(f"Removing tracked value of {removed_acc} and inserting {test_acc}")
        self.test_accs.put(test_acc)

    def should_freeze(self, most_recent_acc):
        if self.frozen:
            print(f"Classifier {self.level} already frozen")
        if self.test_accs.qsize() < self.patience:
            return False
        should_freeze = True
        while not self.test_accs.empty():
            acc = self.test_accs.get()
            if acc < most_recent_acc:
                should_freeze = False
        self.frozen = should_freeze
        return should_freeze
    

class GFNet_Dynn(GFNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
      
    def set_CE_IC_tradeoff(self, CE_IC_tradeoff):
        self.CE_IC_tradeoff = CE_IC_tradeoff

    def set_intermediate_heads(self, intermediate_head_positions):
        self.intermediate_head_positions = intermediate_head_positions

        self.intermediate_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
            for _ in range(len(self.intermediate_head_positions))])
        
        # Accuracy trackers 新增
        self.accuracy_trackers = []
        for i in range(len(self.intermediate_head_positions)):
            self.accuracy_trackers.append(ClassifierAccuracyTracker(i))
        
    # Predictive exit point  新增
    def get_Predictive_exit_point(self ,current_logits):
        p_maxes, entropies, _, margins, entropy_pows = compute_detached_uncertainty_metrics(current_logits, None)
        p_maxes = torch.tensor(p_maxes)[:, None]
        entropies = torch.tensor(entropies)[:, None]
        margins = torch.tensor(margins)[:, None]
        entropy_pows = torch.tensor(entropy_pows)[:, None]
        uncertainty_metrics = torch.cat((p_maxes, entropies, margins, entropy_pows), dim = 1)
        uncertainty_metrics = uncertainty_metrics.to(current_logits.device)
        uncertainty_metrics = torch.cat((uncertainty_metrics, current_logits), dim=1)
        return self.predictive_engine(uncertainty_metrics)

    def freeze_intermediate_classifier(self, classifier_idx):
        classifier = self.intermediate_heads[classifier_idx]
        for param in classifier.parameters():
            param.requires_grad = False

    def unfreeze_intermediate_classifier(self, classifier_idx):
        classifier = self.intermediate_heads[classifier_idx]
        for param in classifier.parameters():
            param.requires_grad = True

    def unfreeze_all_intermediate_classifiers(self):
        for inter_head in self.intermediate_heads:
            for param in inter_head.parameters():
                param.requires_grad = True

    def set_cost_per_exit(self, mult_add_at_exits: list[float], scale = 1e6):
        normalized_cost = torch.tensor(mult_add_at_exits) / mult_add_at_exits[-1]
        self.mult_add_at_exits = (torch.tensor(mult_add_at_exits) / scale).tolist()
        self.normalized_cost_per_exit = normalized_cost.tolist()

    def are_all_classifiers_frozen(self):
        for inter_head in self.intermediate_heads:
            for param in inter_head.parameters():
                if param.requires_grad:
                    return False
        return True

    def is_classifier_frozen(self, classifier_idx):
        inter_head = self.intermediate_heads[classifier_idx]
        for param in inter_head.parameters():
            if param.requires_grad:
                return False
        return True

    def set_threshold_gates(self, gates):
        assert len(gates) == len(self.intermediate_heads), 'Net should have as many gates as there are intermediate classifiers'
        self.gates = gates

    
    def set_learnable_gates(self, gate_positions, direct_exit_prob_param=False, gate_type=GateType.UNCERTAINTY):
        self.gate_positions = gate_positions
        self.direct_exit_prob_param = direct_exit_prob_param
        self.gate_type = gate_type
        if gate_type == GateType.UNCERTAINTY:
            self.gates = nn.ModuleList([
                LearnableUncGate() for _ in range(len(self.gate_positions))])
        elif gate_type == GateType.IDENTITY:
            self.gates = nn.ModuleList([IdentityGate() for _ in range(len(self.gate_positions))])

    def get_gate_prediction(self, l, current_logits):
        return self.gates[l](current_logits)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        intermediate_z = [] # the embedding fed into the augmenting classifiers
        for blk_idx, blk in enumerate(self.blocks):
            x = blk.forward(x)
            if hasattr(self, 'intermediate_head_positions') and blk_idx in self.intermediate_head_positions:
                intermediate_z.append(self.norm(x).mean(1))
        x = self.norm(x).mean(1)
        return x,  intermediate_z

    def forward(self, x):
        x, intermediate_outs = self.forward_features(x)
        intermediate_logits = []
        if intermediate_outs: 
            for head_idx in range(len(self.intermediate_heads)):
                intermediate_head = self.intermediate_heads[head_idx]
                intermediate_logits.append(intermediate_head(intermediate_outs[head_idx]))
        x = self.head(x)

        return x, intermediate_logits

    def forward_for_inference(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        intermediate_logits = []
        Gates = []
        for blk_idx, blk in enumerate(self.blocks):
            x = blk.forward(x)
            inter_z = self.norm(x)
            inter_logits = None
            if hasattr(self, 'intermediate_heads') and blk_idx in self.intermediate_head_positions:
                index = self.intermediate_head_positions.index(blk_idx)  
                intermediate_head = self.intermediate_heads[index]
                inter_logits = intermediate_head(inter_z.mean(1))
                intermediate_logits.append(inter_logits)

                if hasattr(self, 'gates'):
                    gate = self.gates[index]
                    g = gate(inter_logits)
                else:
                    g = 0

                Gates.append(g)
        x = self.norm(x)
        x = self.head(x.mean(1))
        return x, intermediate_logits, Gates
