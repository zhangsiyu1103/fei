import numpy as np
import os
import time
import argparse
import utils
from datasets import get_dataset, select_data
from explain import explain 

def save_attribution(args, existing_attrs):
    """Save attribution maps to files"""
    all_attrs = np.array(existing_attrs)
    filepath = os.path.join(args.save_dir, f"{args.ablation}_attributions.npy")
    np.save(filepath, all_attrs)

def eval_ablation(args):
    model = utils.load_model(args.model, args.dataset)
    print(model)
    for param in model.parameters():
        param.requires_grad = False

    cur_set = get_dataset(args.dataset)

    if args.dataset == "cub":
        all_idxs = np.loadtxt('data_index_cub.txt', int, delimiter=",")
    elif args.dataset == "imagenet":
        all_idxs = np.loadtxt('data_index_imagenet.txt', int, delimiter=",")

    args.save_dir = os.path.join(args.save_dir, "{}".format(args.model))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    result = dict()
    result["idxs"] = all_idxs

    current_attrs = []

    for idx in all_idxs:
        img, _, model_target = select_data(cur_set, idx, model)
        if args.ablation == "no_ensemble":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, area_mode="none", epochs=500, visualize=args.visualize)
        elif args.ablation == "beta_cons":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, beta_const=True)
        elif args.ablation == "fractile3":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, areas=[0.1, 0.5, 0.9])
        elif args.ablation == "fractile9":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, areas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        elif args.ablation == "early_clipping":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, clipping_=(0, 16))
        elif args.ablation == "post_clipping":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, clipping_=(16, 30))
        elif args.ablation == "partial":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, epochs=50, visualize=args.visualize, areas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        elif args.ablation == "early_clipping8":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, clipping_=(0, 8))
        elif args.ablation == "early_clipping4":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, clipping_=(0, 4))
        elif args.ablation == "beta_cons3":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, beta_const=True, beta=1e-3)
        elif args.ablation == "beta_cons0":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, beta_const=True, beta=1)
        elif args.ablation == "beta_cons4":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, beta_const=True, beta=1e-4)
        elif args.ablation == "L1_3":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, area_mode="l1", beta=1e-3, epochs=500)
        elif args.ablation == "L1_4":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, area_mode="l1", beta=1e-4, epochs=500)
        elif args.ablation == "L1_2":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, area_mode="l1", beta=1e-2, epochs=500)
        elif args.ablation == "fully_200":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, epochs=200)
        elif args.ablation == "fully_300":
            attr = explain(model, img, model_target, defense_mode=args.defense_mode, save_dir=args.save_dir, visualize=args.visualize, epochs=300)


        current_attrs.append(attr.detach().cpu().numpy())
        save_attribution(args, current_attrs)






def main():
    parser = argparse.ArgumentParser("method")
    parser.add_argument('--defense_mode', type=str, default="IBM", choices=["IBM", "VM", "IVM", "AVM", "NONE"])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=["imagenet", "cifar10", "cub"], help='dataset to work on')
    parser.add_argument('--model', type=str, default="vgg16", choices=["vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "googlenet", "alexnet"], help='model name')
    parser.add_argument('--ablation', type=str, default="early_clipping8")
    parser.add_argument('--visualize', action="store_true", help='visualize the result')
    parser.add_argument('--save_dir', type=str, default="./result_ablation_save", help='directory to store result')


    args = parser.parse_args()


    s = time.time()
    eval_ablation(args)
    e = time.time()
    print("total time: ", e - s)


if __name__ == '__main__':
    main()

    
