#
# Copyright (c) Microsoft Corporation.
#

#
# Methods for compressing a network using an ensemble of interpolants
#

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import argparse
import random
from wrapper import Wrapper
import utils
from eval_metric import eval_single_perturbation
from datasets import *
from explain import explain

def eval_clipping(args):
    device ="cuda" if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model)

    for param in model.parameters():
        param.requires_grad = False

    save_dir = os.path.join(args.save_dir, "{}_{}_{}".format(args.model, args.dataset,args.defense_mode))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = utils.numpy_to_tensor(np.zeros((1, 224, 224, 3))).to(device)
    with torch.no_grad():
        tar = model(img).max(1)[1].item()
    result = dict()
    correct = 0
    count = 0
    for i in range(1000):
        if i == tar:
            continue
        attr = explain(model, img, i, defense_mode = args.defense_mode, reference_func = torch.zeros_like)
        with torch.no_grad():
            cur_tar = model(img*attr).max(1)[1].item()
            if cur_tar == i:
                correct +=1
            count +=1
        result["acc"] = correct/count
        print(result["acc"])
        torch.save(result,"{}/defense_acc.pth".format(save_dir))




def main():
    parser = argparse.ArgumentParser("method")
    parser.add_argument('--defense_mode', type=str, default="IBM", choices = ["IBM", "VM", "IVM", "AVM", "NONE"])
    parser.add_argument('--dataset', type=str, default='imagenet', choices = ["imagenet","cifar10"], help='dataset to work on')
    parser.add_argument('--model', type=str, default="vgg16", choices = ["vgg19","resnet18","resnet34","resnet50", "resnet101", "vgg16","googlenet","alexnet"], help='model name')
    parser.add_argument('--save_dir', type=str, default="./result_defense", help='directory to store result')


    args = parser.parse_args()

    
    s = time.time()
    eval_clipping(args)
    e = time.time()
    print("total time: ", e-s)
    #except:
    #    print("error")
    #    exit(1)


if __name__ == '__main__':
    main()

    
