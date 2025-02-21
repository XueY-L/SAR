"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent and EATA code.

CUDA_VISIBLE_DEVICES=1 python3 main_cifar100c.py --exp_type normal --model resnext  --test_batch_size 200
"""
from logging import debug
import os
import time
import argparse
import json
import random
import numpy as np
from pycm import *

import math
from typing import ValuesView

from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.cli_utils import *

import torch
import torch.nn.functional as F

import tent
import eata
import sar
from sam import SAM
import timm

import models.Res as Resnet

from robustbench.data import load_imagenetc, load_cifar100c
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel


def get_args():

    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--data', default='/home/yuanxue/data/ILSVRC', help='path to dataset')
    parser.add_argument('--data_corruption', default='/home/yuanxue/data/ImageNet-C', help='path to corruption dataset')
    parser.add_argument('--output', default='./exps', help='the output directory of this experiment')

    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')

    # dataloader
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--test_batch_size', default=50, type=int, help='mini-batch size for testing, before default value is 4')
    parser.add_argument('--if_shuffle', default=False, type=bool, help='if shuffle the test set.')

    # corruption settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='sar', type=str, help='no_adapt, tent, eata, sar')
    parser.add_argument('--model', default='vitbase_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='label_shifts', type=str, help='normal, mix_shifts, bs1, label_shifts')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(100)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
    parser.add_argument('--imbalance_ratio', default=500000, type=float, help='imbalance ratio for label shift exps, selected from [1, 1000, 2000, 3000, 4000, 5000, 500000], 1  denotes totally uniform and 500000 denotes (almost the same to Pure Class Order). See Section 4.3 for details;')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if not os.path.exists(args.output): # and args.local_rank == 0
        os.makedirs(args.output, exist_ok=True)


    args.logger_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-{}-{}-level{}-seed{}.txt".format(args.method, args.model, args.level, args.seed)
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False) 
        
    
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

    if args.exp_type == 'mix_shifts':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)

            val_dataset, _ = prepare_test_data(args)
            if args.method in ['tent', 'no_adapt', 'eata', 'sar']:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            datasets.append(val_dataset)

        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.test_batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)
        common_corruptions = ['mix_shifts']
    
    if args.exp_type == 'bs1':
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")

    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")


    acc1s, acc5s = [], []
    ir = args.imbalance_ratio
    t1 = time.time()
    for corrupt in common_corruptions:
        args.corruption = corrupt
        bs = args.test_batch_size
        # args.print_freq = 50000 // 20 // bs
        args.print_freq = 10

        # 我们没用原来的代码搞数据集
        # if args.method in ['tent', 'eata', 'sar', 'no_adapt']:
        #     if args.corruption != 'mix_shifts':
        #         val_dataset, val_loader = prepare_test_data(args)
        #         val_dataset.switch_mode(True, False)
        # else:
        #     assert False, NotImplementedError
        # # construt new dataset with online imbalanced label distribution shifts, see Section 4.3 for details
        # # note that this operation does not support mix-domain-shifts exps
        # if args.exp_type == 'label_shifts':
        #     logger.info(f"imbalance ratio is {ir}")
        #     if args.seed == 2021:
        #         indices_path = './dataset/total_{}_ir_{}_class_order_shuffle_yes.npy'.format(100000, ir)
        #     else:
        #         indices_path = './dataset/seed{}_total_{}_ir_{}_class_order_shuffle_yes.npy'.format(args.seed, 100000, ir)
        #     logger.info(f"label_shifts_indices_path is {indices_path}")
        #     indices = np.load(indices_path)
        #     val_dataset.set_specific_subset(indices.astype(int).tolist())
        
        # build model for adaptation
        if args.method in ['tent', 'eata', 'sar', 'no_adapt']:
            if args.model == "resnet50_gn_timm":
                net = timm.create_model('resnet50_gn', pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == "vitbase_timm":
                net = timm.create_model('vit_base_patch16_224', pretrained=True)
                args.lr = (0.001 / 64) * bs
            elif args.model == "resnet50_bn_torch":
                net = Resnet.__dicst__['resnet50'](pretrained=True)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == 'resnext':
                net = load_model('Hendrycks2020AugMix_ResNeXt', './ckpt', 'cifar100', ThreatModel.corruptions).cuda()
                net.load_state_dict(torch.load('./ckpt/cifar100/corruptions/ckpt_cifar100_[\'gaussian_noise\']_[1].pt')['model'])
            else:
                assert False, NotImplementedError
            net = net.cuda()
        else:
            assert False, NotImplementedError

        if args.exp_type == 'bs1' and args.method == 'sar':
            args.lr = 2 * args.lr
            logger.info("double lr for sar under bs=1")

        logger.info(args)

        if args.method in ['sar']:
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            logger.info(param_names)

            args.lr = 0.00025
            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)

            x_test, y_test = load_cifar100c(10000, 5, '/home/yuanxue/data', False, [corrupt])
            x_test, y_test = x_test.cuda(), y_test.cuda()

            acc = 0.
            n_batches = math.ceil(x_test.shape[0] / args.test_batch_size)
            with torch.no_grad():
                for counter in range(n_batches):
                    x_curr = x_test[counter * args.test_batch_size:(counter + 1) *
                            args.test_batch_size].cuda()
                    y_curr = y_test[counter * args.test_batch_size:(counter + 1) *
                            args.test_batch_size].cuda()

                    output = adapt_model(x_curr)
                    acc += (output.max(1)[1] == y_curr).float().sum()
            logger.info(f"{corrupt} accuracy: {acc.item() / x_test.shape[0]:.4}")

        else:
            assert False, NotImplementedError
    t2 = time.time()
    logger.info(f"Time: {t2-t1}s")