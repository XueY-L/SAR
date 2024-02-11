"""
Copyright to SAR Authors, ICLR 2023 Oral (notable-top-5%)
built upon on Tent and EATA code.

CUDA_VISIBLE_DEVICES=0 python3 main_domainnet126.py --exp_type normal --model resnet50_bn_torch
"""
from collections import OrderedDict
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
import torchvision.transforms as transforms
import torchvision.models as tmodels

import tent
import eata
import sar
from sam import SAM
import timm

import models.Res as Resnet

from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

from image_list import ImageList


def validate(val_loader, model, criterion, args, mode='eval'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0], dl[1]
            if args.gpu is not None:
                images = images.cuda()
            if torch.cuda.is_available():
                target = target.cuda()
            # compute output
            output = model(images)
            # _, targets = output.max(1)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            if i > 10 and args.debug:
                break
    return top1.avg, top5.avg



def get_args():

    parser = argparse.ArgumentParser(description='SAR exps')

    # path
    parser.add_argument('--data', default='/home/yxue/datasets/ILSVRC', help='path to dataset')
    parser.add_argument('--data_corruption', default='/home/yxue/datasets/DomainNet-126', help='path to corruption dataset')
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
    parser.add_argument('--e_margin', type=float, default=math.log(126)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')

    # Exp Settings
    parser.add_argument('--method', default='sar', type=str, help='no_adapt, tent, eata, sar')
    parser.add_argument('--model', default='vitbase_timm', type=str, help='resnet50_gn_timm or resnet50_bn_torch or vitbase_timm')
    parser.add_argument('--exp_type', default='label_shifts', type=str, help='normal, mix_shifts, bs1, label_shifts')

    # SAR parameters
    parser.add_argument('--sar_margin_e0', default=math.log(126)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')
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
        
    
    source_domain = 'painting'
    common_corruptions = ['clipart', 'real']

    if args.exp_type == 'bs1':
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")

    acc1s, acc5s = [], []
    ir = args.imbalance_ratio
    for corrupt in common_corruptions:
        args.print_freq = 50

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        label_file = os.path.join(args.data_corruption, f"{corrupt}_list.txt")
        test_dataset = ImageList(args.data_corruption, label_file, transform=test_transform)
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)

        # build model for adaptation
        if args.method in ['tent', 'eata', 'sar', 'no_adapt']:
            class ImageNormalizer(nn.Module):
                def __init__(self, mean, std):
                    super(ImageNormalizer, self).__init__()

                    self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
                    self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

                def forward(self, inp):
                    if isinstance(inp, tuple):
                        return ((inp[0] - self.mean) / self.std, inp[1])
                    else:
                        return (inp - self.mean) / self.std
            
            net = nn.Sequential(
                OrderedDict([
                    ('normalize', ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
                    ('model', tmodels.resnet50(num_classes=126)),
                ])
            ).cuda()
            net.model.load_state_dict(torch.load(f'/home/yxue/model_fusion_dnn/ckpt_res50_domainnet126/checkpoint/ckpt_{source_domain}__sgd_lr-s0.001_lr-w-1.0_bs32_seed42_source-[]_DomainNet126_resnet50-1.0x_SingleTraining-DomainNet126_lrd-[-2, -1]_wd-0.0005.pth')['net'])
            
            args.lr = (0.00025 / 64) * args.test_batch_size * 2 if args.test_batch_size < 32 else 0.00025

        if args.exp_type == 'bs1' and args.method == 'sar':
            args.lr = 2 * args.lr
            logger.info("double lr for sar under bs=1")

        logger.info(args)

        if args.method in ['sar']:
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            logger.info(param_names)

            base_optimizer = torch.optim.SGD
            optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)

            batch_time = AverageMeter('Time', ':6.3f')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(val_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
            end = time.time()
            for i, dl in enumerate(val_loader):
                images, target = dl[0], dl[1]
                if args.gpu is not None:
                    images = images.cuda()
                if torch.cuda.is_available():
                    target = target.cuda()
                output = adapt_model(images)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i)

            acc1 = top1.avg
            acc5 = top5.avg

            logger.info(f"Result under {corrupt}. The adaptation accuracy of SAR is top1: {acc1:.5f} and top5: {acc5:.5f}")

            acc1s.append(top1.avg.item())
            acc5s.append(top5.avg.item())

            logger.info(f"acc1s are {acc1s}, acc1-mean is {np.mean(acc1s)}")
            logger.info(f"acc5s are {acc5s}, acc5-mean is {np.mean(acc5s)}")

        else:
            assert False, NotImplementedError