import numpy as np
import torch
import torch.optim as optim
import os
import random
from wrapper import Wrapper
import utils


def get_random_reference(img):
    shape = img.shape
    color = range(256)
    ref = np.array([random.choice(color), random.choice(color), random.choice(color)])
    reference = np.repeat(ref, shape[2] * shape[3], axis=0).reshape(1, 3, shape[2], shape[3]).transpose(0, 2, 3, 1) / 255.0
    reference = utils.numpy_to_tensor(reference)
    return reference


def generate_attr(wrapped_model, img, target, area, start, delta_attr=None, mode="preservation", shape=None, lr=0.01, epochs=100, beta=1e-2, reference_func=get_random_reference, binary=False, threshold=None, area_regulation=True, area_as_ratio=True, metric_func=None, beta_const=False):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = wrapped_model.model

    img = img.to(device)
    if delta_attr is None:
        if shape is None:
            delta_attr = torch.zeros((img.shape[0], 1, *img.shape[2:])).to(device)
        else:
            delta_attr = torch.zeros(shape).to(device)

    delta_attr.requires_grad = True

    n_ele = torch.numel(delta_attr[0])

    optimizer = optim.Adam([delta_attr], lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    if area_as_ratio:
        n = int(n_ele * area)
    else:
        n = area

    wrapped_model.defense()

    max_attr = torch.ones_like(start) - start

    for i in range(epochs):
        attr = delta_attr + start
        reference = reference_func(img).to(device)
        if area_regulation:
            if beta_const:
                loss_l1 = beta * torch.abs(torch.sum(attr, [1, 2, 3]) - n).mean()
            else:
                loss_l1 = i * beta * torch.abs(torch.sum(attr, [1, 2, 3]) - n).mean()
        else:
            loss_l1 = torch.tensor(0).to(device)

        if binary and i >= threshold:
            loss_force = i * beta * torch.mul(attr, 1 - attr).mean(0).sum()
        else:
            loss_force = torch.tensor(0).to(device)
        loss = loss_l1 + loss_force

        new_attr = attr.expand(*img.shape)

        loss_pre = torch.tensor(0).to(device)
        loss_del = torch.tensor(0).to(device)

        if mode == "preservation" or mode == "hybrid":
            input_pre = torch.mul(img, new_attr) + torch.mul(reference, 1 - new_attr)
            out_pre = model(input_pre)
            if metric_func is not None:
                loss_pre = metric_func(out_pre).mean()
            else:
                loss_pre = -out_pre[:, target].mean()
        if mode == "deletion" or mode == "hybrid":
            input_del = torch.mul(img, 1 - new_attr) + torch.mul(reference, new_attr)
            out_del = model(input_del)
            if metric_func is not None:
                loss_del = -metric_func(out_del).mean()
            else:
                loss_del = out_del[:, target].mean()

        loss = loss + loss_pre + loss_del

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        delta_attr.data.clamp_(0)
        delta_attr.data.copy_(torch.min(delta_attr.data, max_attr))
        print("epoch {}, loss: {:.4f}, loss_l1: {:.4f}, loss_force: {:.4f}, loss_pre: {:.4f}, loss_del: {:.4f}".format(str(i), loss.item(), loss_l1.item(), loss_force.item(), loss_pre.item(), loss_del.item()))
    wrapped_model.remove_bhooks()
    attr = delta_attr + start

    return attr.detach()


def image_recover(model, img, target, idx, defense_mode="IBM", lr=0.1, epochs=100, save_dir=None):

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(img)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = wrapped_model.model

    img = img.to(device)
    black = np.zeros((1, 224, 224, 3))
    black = utils.numpy_to_tensor(black).to(device)
    white = np.ones((1, 224, 224, 3))
    white = utils.numpy_to_tensor(white).to(device)
    rec_img = (white - black) * torch.rand_like(img) + black
    rec_img.requires_grad = True

    optimizer = optim.Adam([rec_img], lr=lr)

    wrapped_model.defense()

    for i in range(epochs):
        out_pre = model(rec_img)

        loss = -out_pre[:, target].mean()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        rec_img.data.clamp_(black, white)
        print("epoch {}, loss: {:.7f}".format(str(i), loss.item()))
    utils.visualize_imgs(img, idx, save_dir)
    wrapped_model.remove_bhooks()

    return rec_img.detach()


def explain(model, img, target, mode="preservation", area_mode="ensemble", defense_mode="IBM", shape=None, lr=0.1, epochs=100, beta=1e-1, reference_func=get_random_reference, binary=False, threshold=None, visualize=False, save_dir="./result", areas=None, suffix=None, test_mode=False, metric_func=None, clipping_=None, beta_const=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(model, Wrapper):
        wrapped_model = model
    else:
        wrapped_model = Wrapper(model, defense_mode)
    for param in wrapped_model.model.parameters():
        param.requires_grad = False

    if clipping_ is not None:
        wrapped_model.pre_defense(img, start=clipping_[0], end=clipping_[1])
    else:
        if not test_mode:
            wrapped_model.pre_defense(img)
        else:
            wrapped_model.pre_defense(img, start=121)
            print(wrapped_model._layers_names[121:])

    if area_mode == "ensemble":
        if areas is None:
            areas = [0.1, 0.3, 0.5, 0.7, 0.9]
        if defense_mode == "FGVIS":
            attr = torch.zeros(1, 3, 224, 224).to(device)
            cur_attr = torch.zeros(1, 3, 224, 224).to(device)
        else:
            attr = torch.zeros(1, 1, 224, 224).to(device)
            cur_attr = torch.zeros(1, 1, 224, 224).to(device)
        for area in areas:
            cur_attr = generate_attr(wrapped_model, img, target, area, mode=mode, start=cur_attr, shape=attr.shape, lr=lr, epochs=epochs, beta=beta, reference_func=reference_func, binary=binary, threshold=threshold, metric_func=metric_func, beta_const=beta_const)
            attr += cur_attr
        attr = attr / 5
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_overlay(img, attr, None, save_dir, suffix)
    elif area_mode == "l1":
        attr = torch.zeros(1, 1, 224, 224).to(device)
        delta_attr = torch.ones(1, 1, 224, 224).to(device)
        attr = generate_attr(wrapped_model, img, target, 0, mode=mode, start=attr, delta_attr=delta_attr, shape=shape, lr=lr, epochs=epochs, beta=beta, reference_func=reference_func, binary=binary, threshold=threshold, beta_const=beta_const)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(img, attr, "l1", save_dir, suffix)
    elif area_mode == "none":
        attr = torch.zeros(1, 1, 224, 224).to(device)
        attr = generate_attr(wrapped_model, img, target, 0, mode=mode, start=attr, shape=shape, lr=lr, epochs=epochs, beta=beta, reference_func=reference_func, binary=binary, threshold=threshold, area_regulation=False)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(img, attr, "ensemble_none", save_dir, suffix)

    wrapped_model.remove_hook()
    return attr