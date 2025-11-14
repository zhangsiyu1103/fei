import numpy as np
import os
import time
import argparse
import utils
from datasets import get_dataset, select_data
from explain import image_recover

def image_reconstruct(args):
    model = utils.load_model(args.model)

    for param in model.parameters():
        param.requires_grad = False

    cur_set = get_dataset(args.dataset)

    if args.dataset == "cub":
        all_idxs = np.loadtxt('data_index_cub.txt', int, delimiter=",")
    elif args.dataset == "imagenet":
        all_idxs = np.loadtxt('data_index_imagenet.txt', int, delimiter=",")
    
    save_dir = os.path.join(args.save_dir, "{}_{}_{}".format(args.model, args.dataset,args.defense_mode))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result = dict()
    result["idxs"]=all_idxs
    
    for idx in all_idxs:
        cur_dir = os.path.join(save_dir, str(idx))

        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        img, _, model_target = select_data(cur_set, idx, model)
        image_recover(model, img, model_target, defense_mode = args.defense_mode, save_dir = cur_dir)








def main():
    parser = argparse.ArgumentParser("method")
    parser.add_argument('--defense_mode', type=str, default="IBM", choices = ["IBM", "VM", "IVM", "AVM", "NONE"])
    parser.add_argument('--dataset', type=str, default='imagenet', choices = ["imagenet","cifar10"], help='dataset to work on')
    parser.add_argument('--model', type=str, default="vgg16", choices = ["vgg19","resnet18","resnet34","resnet50", "resnet101", "vgg16","googlenet","alexnet"], help='model name')
    parser.add_argument('--save_dir', type=str, default="./result_rec", help='directory to store result')


    args = parser.parse_args()

    
    s = time.time()
    image_reconstruct(args)
    e = time.time()
    print("total time: ", e-s)



if __name__ == '__main__':
    main()

    