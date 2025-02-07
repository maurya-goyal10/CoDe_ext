# ----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 28-Sept-2024
#
# ---------------------------------------------------------------------------
# Contains the code to compute style scores
# ---------------------------------------------------------------------------

import argparse
import builtins
import os
import pathlib
import random
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
import torchvision.models as torchvision_models
from torchvision.models import VGG16_Weights

import utils
from utils import extract_features_pca

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

parser = argparse.ArgumentParser('dynamicDistances-Embedding Generation Module')

parser.add_argument('--qsplit', default='query', choices=['query', 'database'], type=str, help="The inferences")
parser.add_argument('--data-dir', type=str, default=None,
                        help='The directory of concerned dataset')
parser.add_argument('--pt_style', default='csd', type=str)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--multiscale', default=False, type=utils.bool_flag)

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--num_loss_chunks', default=1, type=int)
parser.add_argument('--isvit', action='store_true')
parser.add_argument('--layer', default=1, type=int, help="layer from end to create descriptors from.")
parser.add_argument('--feattype', default='normal', type=str, choices=['otprojected', 'weighted', 'concated', 'gram', 'normal'])
parser.add_argument('--projdim', default=256, type=int)

parser.add_argument('-mp', '--model_path', type=str, default=None)
parser.add_argument('--gram_dims', default=1024, type=int)
parser.add_argument('--query_count', default=-1, type=int, help='Number of queries to consider for final evaluation. Works only for domainnet')

parser.add_argument('--embed_dir', default='./embeddings', type=str, help='Directory to save embeddings')

## Additional config for CSD
parser.add_argument('--eval_embed', default='head', choices=['head', 'backbone'], help="Which embed to use for eval")
parser.add_argument('--skip_val', action='store_true')

def main(args):

    assert args.model_path is not None, "Model path missing for CSD model"
    from CSD.model import CSD_CLIP
    from CSD.utils import has_batchnorms, convert_state_dict
    from CSD.loss_utils import transforms_branch0

    args.content_proj_head = "default"
    model = CSD_CLIP(args.arch, args.content_proj_head)
    if has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    checkpoint = torch.load(args.model_path, map_location="cpu")
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint with msg {msg}")
    preprocess = transforms_branch0

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

if __name__ == '__main__':
    main()