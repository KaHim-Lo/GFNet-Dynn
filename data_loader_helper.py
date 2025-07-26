import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from PIL import Image
import pickle
import numpy as np
# from scipy.io import loadmat

def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("GFNet-Dynn")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def split_dataloader_in_n(data_loader, n):
    try:
        indices = data_loader.sampler.indices
    except:
        indices = list(range(len(data_loader.sampler)))
    dataset = data_loader.dataset
    list_indices = np.array_split(np.array(indices),n) 
    batch_size = data_loader.batch_size
    n_loaders = []
    for i in range(n):
        sampler = SubsetRandomSampler(list_indices[i])
        sub_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        n_loaders.append(sub_loader)
    return n_loaders


DATASET_CONFIG = {
    'UCM': {'nb_classes': 21},
    'NWPU': {'nb_classes': 45},
    'NaSC': {'nb_classes': 10},
    'PatternNet': {'nb_classes': 38},
    'AID': {'nb_classes': 30}
}

def build_dataset(is_train, args, infer_no_resize=False):
    """加载单个数据集（训练或测试）"""

    transform = build_transform(is_train, args, infer_no_resize)
    
    # Get dataset configuration
    if args.data_set not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {args.data_set}")
    
    dataset_info = DATASET_CONFIG[args.data_set]
    nb_classes = dataset_info['nb_classes']
    
    # Construct data path
    dataset_dir = 'train' if is_train else 'val'
    root = os.path.join(args.data_path, dataset_dir)
    
    # Create dataset
    dataset = datasets.ImageFolder(root, transform=transform)
    
    return dataset, nb_classes

def build_dataset_new(args, seed=0, split_ratio=0.8, infer_no_resize=False):
    """加载完整数据集并拆分为训练集和验证集"""

    if args.data_set not in DATASET_CONFIG:
        raise ValueError(f"不支持的数据集类型: {args.data_set}。支持的类型: {list(DATASET_CONFIG.keys())}")
    
    # 获取数据集配置
    config = DATASET_CONFIG[args.data_set]
    nb_classes = config['nb_classes']
    
    # 创建变换和初始数据集
    transform_train = build_transform(True, args, infer_no_resize)
    transform_test = build_transform(False, args, infer_no_resize)
    
    # 加载完整数据集（之后会划分）
    full_dataset = datasets.ImageFolder(args.data_path, transform=transform_train)
    
    # 划分数据集
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, test_dataset, nb_classes


def build_transform(is_train, args, infer_no_resize=False):
    if hasattr(args, 'arch'):
        if 'cait' in args.arch and not is_train:
            print('# using cait eval transform')
            transformations = {}
            transformations= transforms.Compose(
                [transforms.Resize(args.input_size, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
            return transformations
    
    if infer_no_resize:
        print('# using cait eval transform')
        transformations = {}
        transformations= transforms.Compose(
            [transforms.Resize(args.input_size, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        return transformations

    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
