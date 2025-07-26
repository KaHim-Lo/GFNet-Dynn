'''Train DYNN from checkpoint of trained backbone'''
import itertools
import os
import torch
import mlflow
from tools.collect_metric_iter import aggregate_metrics, process_things
from tools.conformal_eedn import compute_conf_threshold
from data_loader_helper import get_path_to_project_root, split_dataloader_in_n,  get_abs_path
from learning_helper import LearningHelper
from log_helper import aggregate_metrics_mlflow
from models.classifier_training_helper import LossContributionMode
from utils import aggregate_dicts, progress_bar
from tools.early_exit_utils import switch_training_phase
from models.gfnet_dynn import TrainingPhase
import numpy as np
import pickle as pk
import utils
def display_progress_bar(prefix_logger, training_phase, step, total, log_dict):
    loss = log_dict[prefix_logger+'/loss']
    if training_phase == TrainingPhase.WARMUP:
        progress_bar(step, total,'Loss: %.3f | Warmup  time' % (loss))
    elif training_phase == TrainingPhase.CLASSIFIER:
        gated_acc = log_dict[prefix_logger+'/gated_acc']
        progress_bar(
                step, total,
                'Classifier Loss: %.3f | Classifier Acc: %.3f%%' %
                (loss, gated_acc))
    elif training_phase == TrainingPhase.GATE:
        progress_bar(step, total, 'Gate Loss: %.3f ' % (loss)) 

def train_single_epoch(args, helper: LearningHelper, device, train_loader, epoch, training_phase,
          bilevel_batch_count=20):
    print('\nEpoch: %d' % epoch)
    helper.net.train()
    gate_positions = helper.net.module.intermediate_head_positions
    num_layers = len(helper.net.module.blocks)
    metrics_dict = {}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)
       
        if training_phase == TrainingPhase.WARMUP:
            #  we compute the warmup loss
            loss, things_of_interest = helper.get_warmup_loss(inputs, targets)
        else:
            if batch_idx % bilevel_batch_count == 0:
                if helper.net.module.are_all_classifiers_frozen(): # no need to train classifiers anymore
                    training_phase = TrainingPhase.GATE
                    print("All classifiers are frozen, setting training phase to gate")
                else:
                    metrics_dict = {}
                    training_phase = switch_training_phase(training_phase)
            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets, training_phase)
        loss.backward()
        helper.optimizer.step()
        
        # obtain the metrics associated with the batch
        metrics_of_batch = process_things(things_of_interest, gate_positions=gate_positions,
                                          targets=targets, batch_size=batch_size,
                                          cost_per_exit=helper.net.module.normalized_cost_per_exit, num_layers=num_layers)
        metrics_of_batch['loss'] = (loss.item(), batch_size)
        
        # keep track of the average metrics
        metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gate_positions=gate_positions)

        # format the metric ready to be displayed
        log_dict = aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gate_positions=gate_positions) 

        if args.use_mlflow:
            mlflow.log_metrics(log_dict,
                                step=batch_idx +
                                (epoch * len(train_loader)))
        
        display_progress_bar('train', training_phase, step=batch_idx, total=len(train_loader), log_dict=log_dict)
        
        if args.barely_train:
            if batch_idx > 20:
                print(
                    '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
                )
                return metrics_dict

    return metrics_dict


'''
Warms up the IMs of a model until convergence using the val accuracy as a metric.
'''
def dynamic_warmup(args, helper: LearningHelper, device, train_loader, val_loader, img_size, patience = 2):
    # Prepare folder for serializing IMs
    DYNAMIC_WARMUP_FOLDER = '_warmup'
    subfolder_name = f'{args.data_set}_{img_size}_{args.ce_ic_tradeoff}'

    checkpoint_folder_path = get_abs_path(["checkpoint"])
    dynamic_warmup_folder_path = f'{checkpoint_folder_path}{DYNAMIC_WARMUP_FOLDER}/{subfolder_name}'
    if not os.path.isdir(dynamic_warmup_folder_path):
        os.makedirs(dynamic_warmup_folder_path, exist_ok=True)
    best_val_accs_per_layer = [0 for _ in range(args.G)]
    patience_counter_per_layer = [0 for _ in range(args.G)]
    helper.net.train()
    num_layers = len(helper.net.module.blocks)
    exit_positions = helper.net.module.intermediate_head_positions
    PRINT_FREQ = 50
    best_acc = 0
    for epoch in range(args.max_warmup_epoch):
        print('\nEpoch: %d' % epoch)
        metrics_dict = {}
        # TRAIN
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)
            loss, things_of_interest = helper.get_warmup_loss(inputs, targets)
            loss.backward()
            helper.optimizer.step() # need scheduler step as well.

            # obtain the metrics associated with the batch
            metrics_of_batch = process_things(things_of_interest, gate_positions=exit_positions,
                                              targets=targets, batch_size=batch_size,
                                              cost_per_exit=helper.net.module.normalized_cost_per_exit, num_layers=num_layers)
            metrics_of_batch['loss'] = (loss.item(), batch_size)
            if batch_idx % PRINT_FREQ == 0:
                print(f"Ep: {epoch}, Batch: {batch_idx}/{len(train_loader)}. Loss: {loss}")
            # keep track of the average metrics
            metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gate_positions=exit_positions)
            log_dict = aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gate_positions=exit_positions)

            if args.use_mlflow:
                mlflow.log_metrics(log_dict,
                                   step=batch_idx +
                                        (epoch * len(train_loader)))
        # EVAL
        val_metrics_dict, best_acc, _ = evaluate(best_acc, args, helper, device, val_loader, epoch, 'val', 'dynamic_warmup', store_results=False)

        correct_per_exit = val_metrics_dict['correct_per_gate'][0]
        total_val_samples = val_metrics_dict['correct_per_gate'][1]
        for layer in range(len(correct_per_exit)):
            im = helper.net.module.intermediate_heads[layer]
            if helper.net.module.is_classifier_frozen(layer):
                print(f"Skipper IM {layer}, it was already frozen")
                continue
            layer_acc = correct_per_exit[layer] / total_val_samples
            if layer_acc > best_val_accs_per_layer[layer]:
                best_val_accs_per_layer[layer] = layer_acc
                # reset patience
                patience_counter_per_layer[layer] = 0
                # serialize IM
                im_state_dict = im.state_dict()
                serializable_dict = {
                    'state': im_state_dict,
                    'acc': layer_acc,
                    'epoch': epoch,
                }
                torch.save(serializable_dict,f'{dynamic_warmup_folder_path}/{layer}.pth')
            else:
                patience_counter_per_layer[layer] += 1
                if patience_counter_per_layer[layer] >= patience:
                    # restore optimal weights for IM and freeze IM
                    checkpoint = torch.load(f'{dynamic_warmup_folder_path}/{layer}.pth',
                                            map_location=torch.device(device))
                    im.load_state_dict(checkpoint['state'], strict=False)
                    helper.net.module.freeze_intermediate_classifier(layer)
                    print(f"Freezing IM {layer} after {epoch} epochs with acc {checkpoint['acc']}")
        if helper.net.module.are_all_classifiers_frozen():
            print(f"All classifiers converged. Stopping dynamic warmup after {epoch}/{args.max_warmup_epoch}")
            print(f"Final accs are {best_val_accs_per_layer}")
            return epoch
    # Reached end of max warm up epoch without full convergence, report it and load best IMs so far.
    print("Reached end of dynamic warmup epoch before convergence of all IMs")
    for relative_idx, absolute_idx in enumerate(exit_positions):
        if helper.net.module.is_classifier_frozen(relative_idx):
            print(f"IM {absolute_idx} has reached convergence")
        else:
            im = helper.net.module.intermediate_heads[relative_idx]
            checkpoint = torch.load(f'{dynamic_warmup_folder_path}/{relative_idx}.pth',
                                    map_location=torch.device(device))
            im.load_state_dict(checkpoint['state'], strict=False)
            print(f"IM {absolute_idx} has not converged, loading params with acc {checkpoint['acc']}")
    return epoch

def evaluate(best_acc, args, helper: LearningHelper, device, init_loader, epoch, mode: str, experiment_name: str, store_results=False):
    helper.net.eval()
    num_layers = len(helper.net.module.blocks)
    gate_positions = helper.net.module.intermediate_head_positions
    metrics_dict = {}
    if mode == 'test': # we should split the data and combine at the end
        loaders = split_dataloader_in_n(init_loader, n=2)
    else:
        loaders = [init_loader]
    metrics_dicts = []
    log_dicts_of_trials = {}
    average_trials_log_dict = {}
    for loader in loaders:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)
            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets)
            
            # obtain the metrics associated with the batch
            metrics_of_batch = process_things(things_of_interest, gate_positions=gate_positions,
                                              targets=targets, batch_size=batch_size,
                                              cost_per_exit=helper.net.module.mult_add_at_exits, num_layers=num_layers)
            metrics_of_batch['loss'] = (loss.item(), batch_size)
            

            # keep track of the average metrics
            metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gate_positions=gate_positions)

            # format the metric ready to be displayed
            log_dict = aggregate_metrics_mlflow(
                    prefix_logger=mode,
                    metrics_dict=metrics_dict, gate_positions=gate_positions)
            #display_progress_bar(prefix_logger=mode,training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(loader), log_dict=log_dict)

            if args.barely_train:
                    if batch_idx > 50:
                        print(
                            '++++++++++++++WARNING++++++++++++++ you are barely testing to test some things'
                        )
                        break
        metrics_dicts.append(metrics_dict)
        for k, v in log_dict.items():
            aggregate_dicts(log_dicts_of_trials, k, v)
    for k,v in log_dicts_of_trials.items():
        average_trials_log_dict[k] = np.mean(v)
    
    gated_acc = average_trials_log_dict[mode+'/gated_acc']
    average_trials_log_dict[mode+'/test_acc']= gated_acc
    mlflow.log_metrics(average_trials_log_dict, step=epoch)
    # Save checkpoint.
    gated_acc = round(gated_acc, 2)
    print("gated_acc:",gated_acc)
    print("best_acc:",best_acc)

    if gated_acc > best_acc and mode == 'val':
        print('Saving..  ')
        state = {
            'model': helper.net.module.state_dict(),
            'acc': gated_acc,
            'epoch': epoch,
        }
        checkpoint_path = os.path.join(get_path_to_project_root(), 'checkpoint')
        this_run_checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_{args.data_set}_{args.ce_ic_tradeoff}_confEE')
        if not os.path.isdir(this_run_checkpoint_path):
            os.mkdir(this_run_checkpoint_path)
        torch.save(
            state,
            os.path.join(this_run_checkpoint_path,f'ckpt_{args.data_set}_{epoch}__{args.ce_ic_tradeoff}__{gated_acc}.pth')
        )
        best_acc = gated_acc
    elif gated_acc <= best_acc: 
        print('Not saving..  ')

        
    
    elif mode == 'test' and store_results:
        print('storing results....')
        with open(experiment_name+'_'+args.data_set+"_"+args.arch+"_"+str(args.ce_ic_tradeoff)+'_results.pk', 'wb') as file:
            pk.dump(log_dicts_of_trials, file)
    return metrics_dict, best_acc, log_dicts_of_trials

# Any action based on the validation set
def set_from_validation(learning_helper, val_metrics_dict, freeze_classifier_with_val=False, alpha_conf = 0.04):
   
    # we fix the 1/0 ratios of of gate tasks based on the optimal percent exit in the validation sets
    
    exit_count_optimal_gate = val_metrics_dict['exit_count_optimal_gate'] # ({0: 0, 1: 0, 2: 0, 3: 0, 4: 6, 5: 72}, 128)
    total = exit_count_optimal_gate[1]
    pos_weights = []
    pos_weights_previous = []
    for gate, count in exit_count_optimal_gate[0].items():
        count = max(count, 0.1)
        pos_weight = (total-count) / count # #0/#1
        pos_weight = min(pos_weight, 5) # clip for stability
        pos_weights.append(pos_weight)



    learning_helper.gate_training_helper.set_ratios(pos_weights)
    
    

    ## compute the quantiles for the conformal intervals
    
    mixed_score, n = val_metrics_dict['gated_score']
    scores_per_gate, n = val_metrics_dict['score_per_gate']
    score_per_final_gate, n = val_metrics_dict['score_per_final_gate']

    all_score_per_gates, n = val_metrics_dict['all_score_per_gate']
    all_final_score, n = val_metrics_dict['all_final_score']

    alpha_qhat_dict = compute_conf_threshold(mixed_score, scores_per_gate+[score_per_final_gate], all_score_per_gates+[all_final_score])
    

    learning_helper.classifier_training_helper.set_conf_thresholds(alpha_qhat_dict)
   


def eval_baseline(args, helper: LearningHelper, val_loader, device, epoch, mode: str):
    helper.net.eval()
    metrics_dict = {}
    metrics_dicts = []
    log_dicts_of_trials = {}
    average_trials_log_dict = {}
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)

        loss, things_of_interest, _ = helper.get_warmup_loss(inputs, targets)

        # obtain the metrics associated with the batch
        metrics_of_batch = process_things(things_of_interest, gates_count=args.G,
                                          targets=targets, batch_size=batch_size,
                                          cost_per_exit=helper.net.module.mult_add_at_exits)
        metrics_of_batch['loss'] = (loss.item(), batch_size)


        # keep track of the average metrics
        metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=args.G)

        # format the metric ready to be displayed
        log_dict = aggregate_metrics_mlflow(
            prefix_logger=mode,
            metrics_dict=metrics_dict, gates_count=args.G)
        #display_progress_bar(prefix_logger=prefix_logger,training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(loader), log_dict=log_dict)

    metrics_dicts.append(metrics_dict)
    for k, v in log_dict.items():
        aggregate_dicts(log_dicts_of_trials, k, v)
    for k,v in log_dicts_of_trials.items():
        average_trials_log_dict[k] = np.mean(v)

    # gated_acc = average_trials_log_dict[mode+'/gated_acc']
    # average_trials_log_dict[mode+'/test_acc']= gated_acc
    # mlflow.log_metrics(average_trials_log_dict, step=epoch)
    return metrics_dict, log_dicts_of_trials

@torch.no_grad()
def test_layer(net, test_loader, epoch, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    net.eval()
    n_blocks = len(net.module.intermediate_head_positions) + 1
    corrects = [0] * n_blocks
    totals = [0] * n_blocks
    header = 'Test:'
    num_samlpes = 0
    # ite = 0
    # total = len(test_loader)
    for images, target in metric_logger.log_every(test_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            final,inter_preds = net.module.forward(images)
            inter_preds.append(final)

        batch_size = images.shape[0]
        num_samlpes = num_samlpes + batch_size
        for i in range(len(inter_preds)):
            pred = inter_preds[i]
            max_preds, argmax_preds = pred.max(dim=1, keepdim=False)
            corrects[i] += (argmax_preds == target).sum().item()
            totals[i] += target.shape[0]

    metric_logger.synchronize_between_processes()
    corrects = [c / t * 100 for c, t in zip(corrects, totals)]
    log_dict = {}
    for blk in range(n_blocks):
        log_dict['test' + '/acc' +
                 str(blk)] = corrects[blk]
        print(f"Epoch:{epoch} || acc_layer{blk}_val:{corrects[blk]}")

    return corrects
@torch.no_grad()
def perform_test(model, test_loader, threshold, flops, device):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    n_blocks = len(model.module.intermediate_head_positions) + 1
    n_stage = n_blocks
    exp = torch.zeros(n_stage)
    exp_correct = torch.zeros(n_stage)

    model.eval()
    num = 0
    acc = 0
    header = 'Val:'

    for images, target in metric_logger.log_every(test_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        num = num + len(target)
        
        with torch.cuda.amp.autocast():
            final_logits, intermediate_logits, Gates = model.module.forward_for_inference(images)
            intermediate_logits.append(final_logits)
        
        Gates = torch.cat(Gates, dim=1)
        actual_exits_binary = torch.nn.functional.sigmoid(Gates)
        output = intermediate_logits
        for i in range(final_logits.size(0)):
            for j in range(n_stage):
                    if j < n_stage - 1:
                    # if j in [3,5,7,9]:
                        if actual_exits_binary[i][j] >= threshold: #确定退出
                            pred = output[j]
                            pred = pred[i, :]
                            max_preds, argmax_preds = pred.max(dim=0, keepdim=False)
                            if argmax_preds == target[i]:
                                acc += 1
                                exp_correct[j] += 1
                            exp[j] += 1
                            break
                    else:
                    # elif j == 11:
                        pred = output[j]
                        pred = pred[i, :]
                        # pred = pred[i, :]
                        max_preds, argmax_preds = pred.max(dim=0, keepdim=False)

                        if argmax_preds == target[i]:
                            acc += 1
                            exp_correct[j] += 1
                        exp[j] += 1
        
    exit_rate = [0] * n_stage
    exit_correct_rate = [0] * n_stage
    print(num)
    print(acc)
    print(f"****************{threshold}****************")
    expected_GFLOPs = 0
    for k in range(n_stage):
        index = k
        # index = k*2 + 3
        # index = k*2 +1
        exit_rate[k] = exp[k] * 100.0 / num
        if exp[k] == 0:
            exit_correct_rate[k] = 0
        else:
            exit_correct_rate[k] = exp_correct[k] * 100.0 / exp[k]
        print(f"Exiting Layer{index}:[{exit_rate[k]}/{exit_correct_rate[k]}]")

        expected_GFLOPs += exit_rate[k] * flops[k]
    acc_correct = acc / num * 100
    expected_GFLOPs = expected_GFLOPs / 100
    print(f"acc_val={acc_correct},total_GFLOPs={expected_GFLOPs}G")
    print(f"****************{threshold}****************")
    return acc_correct, expected_GFLOPs

