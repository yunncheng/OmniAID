# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from copy import deepcopy
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path
import re

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from data.datasets import GenerativeImageDataset
from engine_finetune import train_one_epoch, evaluate

import utils
from utils import NativeScalerWithGradNormCount as NativeScaler, profile_model
from utils import str2bool, remap_checkpoint_keys
import models.OmniAID as OmniAID
import csv
import warnings

warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Resnet fine-tuning', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='AIDE', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--resnet_path', default=None, type=str, metavar='MODEL',
                        help='Path of resnet model')
    parser.add_argument('--convnext_path', default=None, type=str, metavar='MODEL',
                        help='Path of ConvNeXt of model ')
    
    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')    
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                       help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=0.001, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default='path/dataset', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=100, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--use_amp', type=str2bool, default=False, 
                        help="Use apex AMP (Automatic Mixed Precision) or not")
    
    # MoE Training Control
    parser.add_argument('--training_mode', type=str, default='standard',
                        choices=['standard', 'stage1_hard_sampling', 'stage2_router_training'],
                        help="Specifies the training mode. "
                             "'standard': Normal training or evaluation. "
                             "'stage1_hard_sampling': Trains experts on specific datasets. "
                             "'stage2_router_training': Trains the router on a combined dataset.")
    parser.add_argument('--moe_config_path', default='', type=str,
                        help='Path to a JSON file specifying MoE model configuration, including the checkpoint paths for merging each expert and other related hyperparameters.')
    parser.add_argument('--pretrained_checkpoint', default='', type=str,
                        help="Path to a pretrained checkpoint to load model weights from (for hot start/fine-tuning). "
                             "Unlike '--resume', this only loads the model weights and does not restore the optimizer, epoch count, or LR scheduler. "
                             "Use this to train on new data with a pre-trained model.")

    return parser


def setup_for_training(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    return device


def create_model_and_ema(args, device):
    # Create model
    if args.model == "OmniAID":
        print(f"Initializing OmniAID model with {args.num_experts} experts and {args.rank_per_expert} ranks per expert.")
        model = OmniAID.__dict__["OmniAID"](config=args)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    model.to(device)

    # Create EMA
    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    
    model_without_ddp = model

    # Wrap with DDP
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # n_parameters, _ = profile_model(model_without_ddp, 336, device, args)

    total_params = sum(p.numel() for p in model_without_ddp.parameters())
    trainable_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    
    print(f"Total Parameters at model initialization:     {total_params / 1e6:.2f} M")
    print(f"Total number of trainable params at model initialization: {trainable_params / 1e6:.2f} M")

    n_parameters = trainable_params
    return model, model_without_ddp, model_ema, n_parameters


def list_subfolders(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Local data path does not exist: {path}")

    subfolders = []
    print(f"Attempting to list subfolders for path: {path}")
    
    subfolders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(f"Found subdirectories in local path: {sorted(subfolders)}")
    
    return sorted(subfolders)


def create_dataloaders(args, load_train=True, load_val=True):
    """
    Creates dataloaders.
    - If load_train=True, loads the combined training set.
    - If load_val=True, loads the validation set.
    """
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    FOLDER_NAMES = {"GenImage": ("nature", "ai"), "default": ("0_real", "1_fake")}
    
    dataset_train, data_loader_train = None, None
    dataset_val, data_loader_val = None, None

    domain2label = None

    if "Mirage-Train" in args.data_path:
        domain2label = {'Human': 0, 'Animal': 1, 'Object': 2, 'Scene': 3, 'Anime': 4}
    
    elif "GenImage_sd14_classfied" in args.data_path:
        domain2label = {'Human_Animal': 0, 'Object_Scene': 1}
    
    if load_train:
        real_folder_name_train, fake_folder_name_train = FOLDER_NAMES["GenImage" if "GenImage" in args.data_path else "default"]

        data_paths = args.data_path.split(",")
        
        # If multiple comma-separated data paths are provided
        if len(data_paths) > 1:
            list_of_datasets = []
            print("Combining the following training datasets:")
            for data_path in data_paths:
                print(f" - Loading {data_path}")
                subset = GenerativeImageDataset(root=data_path, is_train=True, category2label=domain2label, real_folder_name=real_folder_name_train, fake_folder_name=fake_folder_name_train)
                list_of_datasets.append(subset)
            dataset_train = ConcatDataset(list_of_datasets)
            print(f"\nTotal combined training samples: {len(dataset_train)}")

        # If only one data path is provided
        else:
            trains = list_subfolders(args.data_path)
        
            # Special case for the full GenImage dataset
            if "GenImage" in args.data_path and len(trains) == 8:
                specific_folders = ["Midjourney/imagenet_midjourney/train", "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train", 
                        "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train", "ADM/imagenet_ai_0508_adm/train", "glide/imagenet_glide/train", 
                        "wukong/imagenet_ai_0424_wukong/train", "VQDM/imagenet_ai_0419_vqdm/train", "BigGAN/imagenet_ai_0419_biggan/train"]
                list_of_datasets = []
                print("Combining specific GenImage training datasets:")
                for train_folder in specific_folders:
                    data_path = os.path.join(args.data_path, train_folder)
                    print(f" - Loading {train_folder}")
                    subset = GenerativeImageDataset(root=data_path, is_train=True, category2label=domain2label, real_folder_name=real_folder_name_train, fake_folder_name=fake_folder_name_train)
                    list_of_datasets.append(subset)
                dataset_train = ConcatDataset(list_of_datasets)
                print(f"\nTotal combined training samples: {len(dataset_train)}")
            
            # For any other single folder dataset
            else:
                dataset_train = GenerativeImageDataset(root=args.data_path, is_train=True, category2label=domain2label, real_folder_name=real_folder_name_train, fake_folder_name=fake_folder_name_train)

        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=True,
        )


    if load_val and not args.disable_eval and args.eval_data_path:
        real_folder_name_eval, fake_folder_name_eval = FOLDER_NAMES["GenImage" if "GenImage" in args.eval_data_path else "default"]
        dataset_val = GenerativeImageDataset(root=args.eval_data_path, is_train=False, category2label=domain2label, real_folder_name=real_folder_name_eval, fake_folder_name=fake_folder_name_eval)
        
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
        )

    return dataset_train, data_loader_train, dataset_val, data_loader_val


def create_optimizer_criterion_scaler(args, model_without_ddp):
    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    optimizer = create_optimizer(args, model_without_ddp)
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    print("Criterion: %s" % str(criterion)) 
    return optimizer, criterion, loss_scaler, mixup_fn


def run_training_loop(args, model, model_without_ddp, model_ema, n_parameters, device,
                      criterion, optimizer, loss_scaler, mixup_fn,
                      dataset_train, dataset_val, data_loader_train, data_loader_val):
    """Encapsulates the main training loop, including logging, evaluation, and checkpointing."""
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log_writer = None
    if utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size
    
    max_accuracy = 0.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler, 
            args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, args=args
            )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            if args.dist_eval:
                data_loader_val.sampler.set_epoch(epoch)
            test_stats, acc, real_acc, fake_acc, ap, f1, fnr = evaluate(data_loader_val, model, device)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%, ap: {ap}.")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                for k, v in test_stats.items():
                    if "acc" in k or "loss" in k:
                        log_writer.update(**{f'test_{k}': v}, head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'val_real_acc': real_acc,
                        'val_fake_acc': fake_acc,
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema, acc, real_acc, fake_acc, ap, f1, fnr = evaluate(data_loader_val, model, device)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%, ap: {ap}")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



def load_and_merge_expert_weights(model_without_ddp, config_path):
    """
    Merges expert weights based on a flexible JSON configuration file
    and loads them directly into the provided model instance.
    """
    print(f"\n--- Starting weight merging using config file: {config_path} ---")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_dir = config['stage1_base_dir']
    experts_map = config['experts_map']
    
    # Start with the model's current state dict as the base
    merged_state_dict = model_without_ddp.state_dict()

    # Iterate through the experts defined in the config file
    for expert_idx_str, checkpoint_filename in experts_map.items():
        expert_idx = int(expert_idx_str)
        expert_dir = os.path.join(base_dir, f'expert_{expert_idx}')
        expert_checkpoint_path = os.path.join(expert_dir, checkpoint_filename)
        
        if not os.path.exists(expert_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for expert {expert_idx} not found at {expert_checkpoint_path}")
            
        print(f"Loading weights for expert {expert_idx} from: {expert_checkpoint_path}")
        
        expert_checkpoint = torch.load(expert_checkpoint_path, map_location='cpu')
        
        if 'model' in expert_checkpoint:
            expert_state_dict = expert_checkpoint['model']
        elif 'model_ema' in expert_checkpoint:
            expert_state_dict = expert_checkpoint['model_ema']
        else:
            expert_state_dict = expert_checkpoint
        
        # Find and transplant the weights for the current expert
        expert_pattern = re.compile(f'[USV]_experts\\.{expert_idx}$')
        keys_to_transfer = [key for key in expert_state_dict if expert_pattern.search(key)]

        print(keys_to_transfer)
        if not keys_to_transfer:
            raise ValueError(
                f"Error: No weights found for expert {expert_idx} in checkpoint '{expert_checkpoint_path}'. "
                f"This means the regex pattern could not find any matching weight keys. "
                f"Please check the checkpoint file's contents and the script's logic."
            )
            
        print(f"  - Transferring {len(keys_to_transfer)} weights for expert {expert_idx}.")
        for key in keys_to_transfer:
            merged_state_dict[key] = expert_state_dict[key]

    model_without_ddp.load_state_dict(merged_state_dict)
    print("--- Weight merging complete. Model is ready for Stage 2. ---\n")
    
    return model_without_ddp



def run_stage1_hard_sampling(args):
    device = setup_for_training(args)

    data_paths = args.data_path.split(",")
    
    # If multiple comma-separated data paths are provided
    if len(data_paths) > 1:
        train_domains = []
        for data_path in data_paths:
            sub_domains = list_subfolders(data_path)

            if "GenImage" in data_path and len(sub_domains) == 8:
                specific_folders = ["Midjourney/imagenet_midjourney/train", "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train", 
                        "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/train", "ADM/imagenet_ai_0508_adm/train", "glide/imagenet_glide/train", 
                        "wukong/imagenet_ai_0424_wukong/train", "VQDM/imagenet_ai_0419_vqdm/train", "BigGAN/imagenet_ai_0419_biggan/train"]

                train_domains.extend([os.path.join(data_path, folder) for folder in specific_folders])
            
            elif "Mirage-Train" in data_path and len(sub_domains) == 5:
                specific_folders = ["Human", "Animal", "Object", "Scene", "Anime"]
                train_domains.extend([os.path.join(data_path, folder) for folder in specific_folders])
            
            elif "GenImage_sd14_classfied" in data_path and len(sub_domains) == 2:
                # specific_folders = ["Animal", "Human", "Object", "Scene"]
                specific_folders = ["Human_Animal", "Object_Scene"]
                train_domains.extend([os.path.join(data_path, folder) for folder in specific_folders])
            
            else:
                train_domains.append(data_path)       

    else:
        train_domains = list_subfolders(args.data_path)
        train_domains = [os.path.join(args.data_path, folder) for folder in train_domains]

    assert len(train_domains) == args.num_experts, "Number of experts must match number of data domains."

    _, _, dataset_val, data_loader_val = create_dataloaders(args, load_train=False, load_val=True)
    if data_loader_val is not None:
        print(f"Loaded a global validation set with {len(dataset_val)} images for Stage 1.")


    model, model_without_ddp, model_ema, n_parameters = create_model_and_ema(args, device)
    
    print(f"Starting Stage 1: Training {args.num_experts} experts sequentially...")


    initial_head_state_dict = deepcopy(model_without_ddp.head.state_dict())
    
    data_path_copy = args.data_path  # Save the original data path
    output_dir_copy = args.output_dir # Save the original output directory
    log_dir_copy = args.log_dir # Save the original log directory

    for expert_idx, domain_name in enumerate(train_domains):
        domain_name = train_domains[expert_idx]

        print(f"\n{'='*25} PREPARING EXPERT {expert_idx} | DOMAIN: {domain_name} {'='*25}")

        model_without_ddp.head.load_state_dict(initial_head_state_dict)
        print(f"Classification head has been reset to its initial state for expert {expert_idx}.")

        if hasattr(model_without_ddp, 'set_training_mode'):
            model_without_ddp.set_training_mode('hard_sampling', expert_idx)
        
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print(f"\nNumber of trainable parameters for Expert {expert_idx}: {n_parameters}")

        args.data_path = domain_name
        args.output_dir = os.path.join(output_dir_copy, f"expert_{expert_idx}")
        args.log_dir = os.path.join(log_dir_copy, f"expert_{expert_idx}")

        dataset_train, data_loader_train, _, _ = create_dataloaders(args, load_train=True, load_val=False)
        optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)
        
        # Run training loop for each expert independently
        run_training_loop(
            args, model, model_without_ddp, model_ema, n_parameters, device,
            criterion, optimizer, loss_scaler, mixup_fn,
            dataset_train, dataset_val, data_loader_train, data_loader_val
        )
        print(f"\n===== Finished training Expert {expert_idx} for Domain: {domain_name} =====")

    # Reset
    args.data_path = data_path_copy
    args.output_dir = output_dir_copy
    args.log_dir = log_dir_copy

    print("\n--- Stage 1 finished. ---")


def run_stage2_router_training(args):
    assert args.model == 'OmniAID', "Stage 2 is only for the 'OmniAID' model."
    device = setup_for_training(args)
    
    args.output_dir = os.path.join(args.output_dir , "router")
    args.log_dir = os.path.join(args.log_dir , "router")   
    
    model, model_without_ddp, model_ema, n_parameters = create_model_and_ema(args, device)
    # Set the model to router training mode
    if hasattr(model_without_ddp, 'set_training_mode'):
        model_without_ddp.set_training_mode('router_training')

    # Continue training
    if args.pretrained_checkpoint:
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
    
    # Starting a new Stage 2 run from merged Stage 1 experts
    else:
        assert args.moe_config_path, "For Stage 2, a merge config JSON file must be provided via --moe_config_path."
        model_without_ddp = load_and_merge_expert_weights(model_without_ddp, args.moe_config_path)

    optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)

    print(f"Starting Stage 2: Training Router with {args.num_experts} experts...")
    
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print(f"Number of trainable parameters for Router: {n_parameters}")

    dataset_train, data_loader_train, dataset_val, data_loader_val = create_dataloaders(args, load_train=True, load_val=True)
    
    run_training_loop(
        args, model, model_without_ddp, model_ema, n_parameters, device,
        criterion, optimizer, loss_scaler, mixup_fn,
        dataset_train, dataset_val, data_loader_train, data_loader_val,
        )

    print("\n--- Stage 2 finished. ---")


def run_eval(args):
    device = setup_for_training(args)

    print("Running in Evaluation-Only Mode...")
    model, model_without_ddp, model_ema, _ = create_model_and_ema(args, device)
    utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None, model_ema=model_ema)
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    FOLDER_NAMES = {"GenImage": ("nature", "ai"), "default": ("0_real", "1_fake")}
    real_folder_name_eval, fake_folder_name_eval = FOLDER_NAMES["GenImage" if "GenImage" in args.eval_data_path else "default"]

    domain2label = None
    if "Mirage-Test" in args.eval_data_path:
        domain2label = {'Human': 0, 'Animal': 1, 'Object': 2, 'Scene': 3, 'Anime': 4}
    
    elif "GenImage_sd14_classfied" in args.eval_data_path:
        domain2label = {'Human_Animal': 0, 'Object_Scene': 1}

    vals = list_subfolders(args.eval_data_path)

    if "DRCT-2M" in args.eval_data_path and len(vals) == 16:
        vals = ['ldm-text2im-large-256/val2017', 'stable-diffusion-v1-4/val2017', 'stable-diffusion-v1-5/val2017', 'stable-diffusion-2-1/val2017',
        'stable-diffusion-xl-base-1.0/val2017', 'stable-diffusion-xl-refiner-1.0/val2017', 
        'sd-turbo/val2017', 'sdxl-turbo/val2017',
        'lcm-lora-sdv1-5/val2017', 'lcm-lora-sdxl/val2017', 
        'sd-controlnet-canny/val2017', 'sd21-controlnet-canny/val2017', 'controlnet-canny-sdxl-1.0/val2017',
        'stable-diffusion-inpainting/val2017', 'stable-diffusion-2-inpainting/val2017', 'stable-diffusion-xl-1.0-inpainting-0.1/val2017']
    elif "AIGCDetectionBenchMark" in args.eval_data_path and len(vals) == 17:
        vals = ["progan", "stylegan", "biggan", "cyclegan", "stargan", "gaugan", "stylegan2", "whichfaceisreal", "ADM", "Glide", 
        "Midjourney", "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong", "DALLE2", "sd_xl"]
    elif "GenImage" in args.eval_data_path and len(vals) == 8:
        vals = ["Midjourney/imagenet_midjourney/val", "stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/val", 
                "stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/val", "ADM/imagenet_ai_0508_adm/val", "glide/imagenet_glide/val", 
                "wukong/imagenet_ai_0424_wukong/val", "VQDM/imagenet_ai_0419_vqdm/val", "BigGAN/imagenet_ai_0419_biggan/val"]
    elif "Mirage-test" in args.eval_data_path and len(vals) == 5:
        vals = ['Human', 'Animal', 'Object', 'Scene','Anime']
    else:
        vals = [args.eval_data_path]
    
    rows = [["{} model testing on...".format(args.resume)],
        ['testset', 'accuracy', "real_accuracy", "fake_accuracy", 'avg precision', 'f1_score', 'fnr']]

    total_acc = 0.0
    total_real_acc = 0.0
    total_fake_acc = 0.0
    total_ap = 0.0
    total_f1 = 0.0
    total_fnr = 0.0

    count_acc = 0
    count_real_acc = 0
    count_fake_acc = 0
    count_ap = 0
    count_f1 = 0
    count_fnr = 0
    num_testsets = len(vals)

    for v_id, val in enumerate(vals):
        
        eval_data_path = os.path.join(args.eval_data_path, val)
        dataset_val = GenerativeImageDataset(root=eval_data_path, is_train=False, category2label=domain2label, real_folder_name=real_folder_name_eval, fake_folder_name=fake_folder_name_eval)

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        test_stats, acc, real_acc, fake_acc, ap, f1, fnr = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
    
        print(f"test dataset is {val} acc: {acc}, real acc: {real_acc}, fake acc: {fake_acc}, ap: {ap}, f1: {f1}, fnr: {fnr}")
        print("***********************************")
 
        if acc is None:
            acc = 'N/A'
        else:
            total_acc += acc
            count_acc += 1 
        if real_acc is None:
            real_acc = 'N/A'
        else:
            total_real_acc += real_acc
            count_real_acc += 1
        if fake_acc is None:
            fake_acc = 'N/A'
        else:
            total_fake_acc += fake_acc
            count_fake_acc += 1
        if ap is None:
            ap = 'N/A'
        else:
            total_ap += ap
            count_ap += 1
        if f1 is None:
            f1 = 'N/A'
        else:
            total_f1 += f1
            count_f1 += 1
        if fnr is None:
            fnr = 'N/A'
        else:
            total_fnr += fnr
            count_fnr += 1

        rows.append([val, acc, real_acc, fake_acc, ap, f1, fnr])

    # Calculate the average if there are any testsets
    if num_testsets > 0:
        avg_acc = total_acc / count_acc if count_acc > 0 else None
        avg_real_acc = total_real_acc / count_real_acc if count_real_acc > 0 else None
        avg_fake_acc = total_fake_acc / count_fake_acc if count_fake_acc > 0 else None
        avg_ap = total_ap / count_ap if count_ap > 0 else None
        avg_f1 = total_f1 / count_f1 if count_f1 > 0 else None
        avg_fnr = total_fnr / count_fnr if count_fnr > 0 else None


        avg_acc_str = f'{avg_acc:.4f}' if avg_acc is not None else 'N/A'
        avg_real_str = f'{avg_real_acc:.4f}' if avg_real_acc is not None else 'N/A'
        avg_fake_str = f'{avg_fake_acc:.4f}' if avg_fake_acc is not None else 'N/A'
        avg_ap_str = f'{avg_ap:.4f}' if avg_ap is not None else 'N/A'
        avg_f1_str = f'{avg_f1:.4f}' if avg_f1 is not None else 'N/A'
        avg_fnr_str = f'{avg_fnr:.4f}' if avg_fnr is not None else 'N/A'

        rows.append(['Average', avg_acc_str, avg_real_str, avg_fake_str, avg_ap_str, avg_f1_str, avg_fnr_str])
        print(f"Average over {num_testsets} testsets -> Acc: {avg_acc_str}, Real Acc: {avg_real_str}, Fake Acc: {avg_fake_str}, AP: {avg_ap_str}, F1: {avg_f1_str}, FNR: {avg_fnr_str}")
        print("***********************************")

    test_dataset_name  = args.eval_data_path.split('/')[-2] + '_' + args.eval_data_path.split('/')[-1]

    csv_name = os.path.join(args.output_dir, f'{os.path.basename(args.resume)}_{test_dataset_name}.csv')
    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(rows)
    return


def run_standard_training(args):
    """Controller for the standard training flow."""
    device = setup_for_training(args)

    # Standard training flow
    print("\n--- Running in Standard Training Mode. ---")
    model, model_without_ddp, model_ema, n_parameters = create_model_and_ema(args, device)
    optimizer, criterion, loss_scaler, mixup_fn = create_optimizer_criterion_scaler(args, model_without_ddp)

    # Load checkpoint if resuming standard training
    if args.resume:
        utils.auto_load_model(args=args, model=model, model_without_ddp=model_without_ddp,
                              optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    
    dataset_train, data_loader_train, dataset_val, data_loader_val = create_dataloaders(args, load_train=True, load_val=True)
    run_training_loop(
        args, model, model_without_ddp, model_ema, n_parameters, device,
        criterion, optimizer, loss_scaler, mixup_fn,
        dataset_train, dataset_val, data_loader_train, data_loader_val,
        )
    print("\n--- Standard training finished. ---")


def main(args):

    if args.moe_config_path:
        if not os.path.exists(args.moe_config_path):
            raise FileNotFoundError(f"MoE config file not found at: {args.moe_config_path}")

        with open(args.moe_config_path, 'r') as f:
            moe_config = json.load(f)

        for key, value in moe_config.items():
            if getattr(args, key, None) is None: # Only set if not provided via command line
                setattr(args, key, value)

    if args.eval:
        run_eval(args)
    else:
        if args.training_mode == 'stage1_hard_sampling':
            run_stage1_hard_sampling(args)
        elif args.training_mode == 'stage2_router_training':
            run_stage2_router_training(args)
        else:
            run_standard_training(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('OmniAID traning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
