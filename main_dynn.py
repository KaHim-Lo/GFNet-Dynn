"""
Train DYNN from checkpoint of trained backbone
"""
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from timm.models import *
from functools import partial
import torch.nn as nn
from datetime import datetime
from utils import fix_the_seed
from models.gfnet_dynn import TrainingPhase

# 取消Git提交异常
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# 导入自定义模块
from models.op_counter import measure_model_and_assign_cost_per_exit
from learning_helper import freeze_backbone as freeze_backbone_helper, LearningHelper
from log_helper import setup_mlflow
from models.classifier_training_helper import LossContributionMode
from models.custom_modules.gate import GateType
from models.gate_training_helper import GateObjective
from our_train_helper import (
    set_from_validation, 
    evaluate, 
    train_single_epoch, 
    eval_baseline, 
    dynamic_warmup, 
    test_layer, 
    perform_test
)
from data_loader_helper import build_dataset, build_dataset_new
from models.gfnet_dynn import GFNet_Dynn


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch DYNN Training')
    
    # 训练参数
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--ce_ic_tradeoff', default=0.75, type=float, 
                        help='cost inference and cross entropy loss tradeoff')
    parser.add_argument('--num_epoch', default=200, type=int, help='num of epochs')
    parser.add_argument('--max_warmup_epoch', default=10, type=int, help='max num of warmup epochs')
    parser.add_argument('--bilevel_batch_count', default=60, type=int,
                        help='number of batches before switching training modes')
    parser.add_argument('--transfer-ratio', type=float, default=0.01, 
                        help='lr ratio between classifier and backbone in transfer learning')
    parser.add_argument('--proj_dim', default=32, type=int,
                        help='Target dimension of random projection for ReLU codes')
    parser.add_argument('--num_proj', default=16, type=int,
                        help='Target number of random projection for ReLU codes')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--split_ratio', default=0.8, type=float, help='train/val split ratio')
    
    # 模型架构
    parser.add_argument('--arch', type=str, choices=['GFNet-Dynn'], 
                        default='GFNet-Dynn', help='model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate')
    
    # 数据集参数
    parser.add_argument('--data-path', default='/workspace/data/NaSC', type=str, help='dataset path')
    parser.add_argument('--data-set', default='NaSC', 
                        choices=['UCM', 'NaSC', 'PatternNet', 'AID', 'NWPU'],
                        type=str, help='Dataset name')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 
                                'family', 'genus', 'name'],
                        type=str, help='semantic granularity for iNaturalist')
    
    # 增强参数
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation method')
    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--dist-eval', action='store_true', default=True, 
                        help='Enabling distributed evaluation')
    parser.add_argument('--reprob', type=float, default=0, metavar='PCT',
                        help='Random erase prob')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader')
    
    # 模型特定参数
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--gate', type=GateType, default=GateType.UNCERTAINTY, choices=GateType)
    parser.add_argument('--gate_objective', type=GateObjective, default=GateObjective.CrossEntropy, 
                        choices=GateObjective)
    parser.add_argument('--classifier_loss', type=LossContributionMode, 
                        default=LossContributionMode.BOOSTED, choices=LossContributionMode)
    parser.add_argument('--early_exit_warmup', default=True)
    
    # 运行控制
    parser.add_argument('--barely_train', action='store_true', help='not a real run (for testing)')
    parser.add_argument('--use_mlflow', default=True, help='Store the run with mlflow')
    parser.add_argument('--eval', action='store_true', help='Only run evaluation')
    
    return parser.parse_args()


def setup_experiment(args):
    """设置实验环境"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    
    # 设置MLflow
    if args.use_mlflow:
        now = datetime.now()
        name = "_".join([str(a) for a in [args.ce_ic_tradeoff, args.classifier_loss]])
        cfg = vars(args)
        experiment_name = 'test_run' if args.barely_train else now.strftime("%m-%d-%Y")
        setup_mlflow(name, cfg, experiment_name=experiment_name)
        return experiment_name
    return None


def load_dataset(args):
    """加载数据集"""
    # 特定数据集的路径和输入大小设置
    dataset_paths = {
        'UCM': "/workspace/data/UCM",
        'NWPU': "/workspace/data/RESISC45",
        'NaSC': "/workspace/data-new/NaSC-TG2",
        'PatternNet': "/workspace/data/PatternNet-new",
        'AID': "/workspace/data/AID-new"
    }
    
    input_sizes = {
        'UCM': 256,
        'NWPU': 256,
        'NaSC': 128,
        'PatternNet': 256,
        'AID': 600
    }
    
    # 更新参数
    if args.data_set in dataset_paths:
        args.data_path = dataset_paths[args.data_set]
        args.input_size = input_sizes[args.data_set]
    
    IMG_SIZE = args.input_size
    
    # 构建数据集
    if args.data_set == 'NaSC':
        dataset_train, dataset_val, NUM_CLASSES = build_dataset_new(
            args=args, seed=args.seed, split_ratio=args.split_ratio)
    else:
        dataset_train, NUM_CLASSES = build_dataset(is_train=True, args=args)
        dataset_val, _ = build_dataset(is_train=False, args=args)
    
    # 创建数据加载器
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    return data_loader_train, data_loader_val, IMG_SIZE, NUM_CLASSES


def initialize_model(args, IMG_SIZE, NUM_CLASSES, device):
    """初始化模型"""
    # 创建模型
    model = GFNet_Dynn(
        img_size=IMG_SIZE, 
        num_classes=NUM_CLASSES,
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    
    # 设置门控参数    
    transformer_layer_gating = [3, 5, 7, 9]         # 选择哪些层进行门控
    args.G = len(transformer_layer_gating)
    
    model.set_CE_IC_tradeoff(args.ce_ic_tradeoff)
    model.set_intermediate_heads(transformer_layer_gating)
    model.set_learnable_gates(transformer_layer_gating, direct_exit_prob_param=True)
    
    # 计算FLOPs和参数
    n_flops, n_params, n_flops_at_gates = measure_model_and_assign_cost_per_exit(
        model, IMG_SIZE, IMG_SIZE, num_classes=NUM_CLASSES)
    mult_add_at_exits = (torch.tensor(n_flops_at_gates) / 1e6).tolist()
    print("FLOPs at exits (M):", mult_add_at_exits)
    
    # 移动到设备
    model = model.to(device)
    
    # 多GPU支持
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    return model, model_without_ddp, mult_add_at_exits


def load_checkpoint(model_without_ddp, checkpoint_path, device):
    """加载预训练权重"""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    param_with_issues = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
    
    print("Missing keys:", param_with_issues.missing_keys)
    print("Unexpected keys:", param_with_issues.unexpected_keys)
    
    total_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"Total number of parameters: {total_params}")
    
    return model_without_ddp


def setup_training(args, model, device):
    """设置训练参数"""
    # 冻结主干网络
    unfrozen_modules = ['intermediate_heads', 'gates']
    freeze_backbone_helper(model, unfrozen_modules)
    
    # 设置优化器和学习率调度器
    optimizer = optim.SGD(model.parameters(),
                         lr=args.lr,
                         momentum=0.9,
                         weight_decay=args.wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=args.min_lr, T_max=args.num_epoch)
    
    return optimizer, scheduler


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置实验
    experiment_name = setup_experiment(args)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    # 加载数据集
    data_loader_train, data_loader_val, IMG_SIZE, NUM_CLASSES = load_dataset(args)
    
    # 初始化模型
    model, model_without_ddp, mult_add_at_exits = initialize_model(args, IMG_SIZE, NUM_CLASSES, device)
    
    # 加载检查点
    if args.resume:
        checkpoint_path = "/workspace/GFNet-Dynn/checkpoint/best/ckpt_NaSC_98__0.75__93.02.pth"  
        model_without_ddp = load_checkpoint(model_without_ddp, checkpoint_path, device)
        print("Resuming from checkpoint:", checkpoint_path)
    else:
        print("Training from scratch! No checkpoint resume")
    
    # 设置训练参数
    optimizer, scheduler = setup_training(args, model, device)
    
    # 训练或评估模式
    if args.eval:
        evaluate_model(model, data_loader_val, mult_add_at_exits, device)
    else:
        train_model(args, model, model_without_ddp, optimizer, scheduler, 
                   data_loader_train, data_loader_val, device, experiment_name)


def evaluate_model(model, data_loader_val, mult_add_at_exits, device):
    """评估模型性能"""
    num_tests = 1
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        total_acc = 0
        total_flops = 0
        
        for _ in range(num_tests):
            acc, flops = perform_test(model, data_loader_val, threshold=threshold, 
                                    flops=mult_add_at_exits, device=device)
            total_acc += acc
            total_flops += flops
        
        avg_acc = total_acc / num_tests
        avg_flops = total_flops / num_tests
        print(f"Threshold: {threshold:.1f} -> Avg Acc: {avg_acc:.2f}%, Avg FLOPs: {avg_flops:.2f}G")


def train_model(args, model, model_without_ddp, optimizer, scheduler, 
               data_loader_train, data_loader_val, device, experiment_name):
    """训练模型"""
    best_acc = 0
    learning_helper = LearningHelper(model, optimizer, args, device)
    
    # 动态预热
    warmup_epoch = dynamic_warmup(args, learning_helper, device, 
                                data_loader_train, data_loader_val, args.input_size)
    
    # 解冻所有中间分类器
    print("Unfreezing all intermediate classifiers after warmup")
    model_without_ddp.unfreeze_all_intermediate_classifiers()
    
    # 主训练循环
    for epoch in range(warmup_epoch + 1, args.num_epoch):
        # 训练单个epoch
        train_single_epoch(args, learning_helper, device, data_loader_train, 
                         epoch=epoch, training_phase=TrainingPhase.CLASSIFIER, 
                         bilevel_batch_count=args.bilevel_batch_count)
        
        # 验证
        val_metrics_dict, latest_acc, _ = evaluate(
            best_acc, args, learning_helper, device, data_loader_val, 
            epoch, mode='val', experiment_name=experiment_name)
        
        # 测试各层性能
        test_layer(model, data_loader_val, epoch, device=device)
        
        # 更新最佳准确率
        if latest_acc > best_acc:
            best_acc = latest_acc
        
        # 设置验证指标
        set_from_validation(learning_helper, val_metrics_dict)
        
        # 更新学习率
        scheduler.step()


if __name__ == '__main__':
    main()