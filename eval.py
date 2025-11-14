import numpy as np
import torch
import torch.nn as nn
import os
import time
import argparse
import utils
from eval_metric import eval_single_perturbation
from datasets import get_dataset, select_data
from explain import explain

def eval_clipping(args):
    model = utils.load_model(args.model, args.dataset)

    for param in model.parameters():
        param.requires_grad = False

    cur_set = get_dataset(args.dataset)

    metric_model = nn.Sequential(model, nn.Softmax(dim=1))

    if args.dataset == "cub":
        all_idxs = np.loadtxt('data_index_cub.txt', int, delimiter=",")
    elif args.dataset == "imagenet":
        all_idxs = np.loadtxt('data_index_imagenet.txt', int, delimiter=",")

    save_dir = os.path.join(args.save_dir, f"{args.dataset}/{args.dataset}/{args.model}/{args.defense_mode}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result = dict()
    result["idxs"] = all_idxs

    for idx in all_idxs:
        cur_dir = os.path.join(save_dir, str(idx))

        img, _, model_target = select_data(cur_set, idx, model)

        attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=cur_dir, visualize=args.visualize)
        eval_single_perturbation(metric_model, attr, img, result)


        torch.save(result, "{}/result.pth".format(save_dir))






def main():
    parser = argparse.ArgumentParser("method")
    parser.add_argument('--defense_mode', type=str, default="IBM", choices=["IBM", "VM", "IVM", "AVM", "NONE"])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=["imagenet", "cifar10", "cub"], help='dataset to work on')
    parser.add_argument('--model', type=str, default="vgg16", choices=["vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "googlenet", "alexnet"], help='model name')
    parser.add_argument('--visualize', action="store_true", help='visualize the result')
    parser.add_argument('--save_dir', type=str, default="./result", help='directory to store result')


    args = parser.parse_args()

    s = time.time()
    eval_clipping(args)
    e = time.time()
    print("total time: ", e - s)

if __name__ == '__main__':
    main()
