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

    cur_set = get_dataset(args.dataset)

    metric_model = nn.Sequential(model, nn.Softmax(dim = 1))

    #all_idxs = torch.load("result_eval/vgg16/result_inact_val_fix.pth")["idxs"]
    #all_idxs = torch.load("./idxs.txt"
    #np.savetxt("data_index.txt",np.array(all_idxs).astype(int), fmt='%i', delimiter=",")
    all_idxs = np.loadtxt('data_index.txt', int, delimiter=",")
    
    save_dir = os.path.join(args.save_dir, "{}_{}_{}_{}".format(args.model, args.dataset,args.defense_mode, args.ablation))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result = dict()
    result["idxs"]=all_idxs
    
    for i, idx in enumerate(all_idxs):
        cur_dir = os.path.join(save_dir, str(idx))

        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        img, data_target, model_target = select_data(cur_set, idx, model)
        if args.ablation == "binary":
            attr = explain(model, img, model_target, defense_mode = args.defense_mode, save_dir = cur_dir, binary = True, threshold = 300, epochs = 500, visualize =args.visualize)
        elif args.ablation == "fully":
            attr = explain(model, img, model_target, defense_mode = args.defense_mode, save_dir = cur_dir, epochs = 500, visualize =args.visualize)
        elif args.ablation == "restart":
            attr = explain(model, img, model_target, defense_mode = args.defense_mode, save_dir = cur_dir, restart = True, visualize =args.visualize)
        elif args.ablation == "no_defense":
            attr = explain(model, img, model_target, defense_mode = "none", save_dir = cur_dir, visualize =args.visualize)
        elif args.ablation == "no_ensemble":
            attr = explain(model, img, model_target, defense_mode = args.defense_mode, save_dir = cur_dir, area_mode = "none", epochs = 500, visualize =args.visualize)
        elif args.ablation == "l1":
            attr = explain(model, img, model_target, defense_mode = args.defense_mode, save_dir = cur_dir, area_mode = "l1", epochs = 500, visualize =args.visualize)


        
        eval_single_perturbation(metric_model, attr, img, result)


        torch.save(result,"{}/result.pth".format(save_dir))

    #torch.save(selected,"selected.pth")






def main():
    parser = argparse.ArgumentParser("method")
    parser.add_argument('--defense_mode', type=str, default="IBM", choices = ["IBM", "VM", "IVM", "AVM", "NONE"])
    parser.add_argument('--dataset', type=str, default='imagenet', choices = ["imagenet","cifar10"], help='dataset to work on')
    parser.add_argument('--model', type=str, default="vgg16", choices = ["vgg19","resnet18","resnet34","resnet50", "resnet101", "vgg16","googlenet","alexnet"], help='model name')
    parser.add_argument('--ablation', type=str, default="binary", choices = ["binary","fully","restart","no_defense","no_ensemble","l1"], help='model name')
    parser.add_argument('--visualize', action = "store_true", help='visualize the result')
    parser.add_argument('--save_dir', type=str, default="./result_ablation", help='directory to store result')


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

    
