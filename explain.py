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
from torchvision import datasets, transforms,models
from matplotlib import pyplot as plt
import json


def get_random_reference(inp):
    shape = inp.shape
    color = range(256)
    ref = np.array([random.choice(color), random.choice(color), random.choice(color)])
    reference = np.repeat(ref, shape[2]*shape[3], axis =0).reshape(1,3,shape[2],shape[3]).transpose(0,2,3,1)/255.0
    reference = utils.numpy_to_tensor(reference)
    return reference

def generate_attr(wrapped_model, inp, target, area, attr = None, defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference,  binary = False, threshold = None, area_regulation = False):

    device  ="cuda" if torch.cuda.is_available() else "cpu"
    model = wrapped_model.model

    inp = inp.to(device)
    if attr is None:
        if shape is None:
            attr = torch.zeros((inp.shape[0],1,*inp.shape[2:])).to(device)
        else:
            attr = torch.zeros(shape).to(device)

    attr.requires_grad = True

    n_ele = torch.numel(attr[0])

    optimizer = optim.Adam([attr], lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    n = int(n_ele*area)

    wrapped_model.defense()

    #inp_save = inp.clone()
    for i in range(epochs):
        reference = reference_func(inp).to(device)

        if area_regulation:
            loss_l1 = i*beta*torch.abs(torch.sum(attr, [1,2,3]) - n).mean()
        else:
            loss_l1 = torch.tensor(0).to(device)

        if binary and i >= threshold:
            loss_force = i*beta*torch.mul(attr, 1 - attr).mean(0).sum()
        else:
            loss_force = torch.tensor(0).to(device)
        loss = loss_l1 + loss_force

        new_attr = attr.expand(*inp.shape)

        loss_pre = torch.tensor(0).to(device)
        loss_del = torch.tensor(0).to(device)
        
        input_pre = torch.mul(inp, new_attr) + torch.mul(reference, 1 - new_attr)
        out_pre = model(input_pre)

        loss_pre = -out_pre[:,target].mean()

        loss = loss + loss_pre


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        attr.data.clamp_(0,1)
        print("epoch {}, loss: {:.4f}, loss l1:{:.4f}, loss force:{:.4f}, loss pre:{:.4f}".format(str(i), loss.item(), loss_l1.item(), loss_force.item(),  loss_pre.item()))
    wrapped_model.remove_bhooks()
    
    return attr.detach()


def image_recover(model, inp, target, defense_mode = "IBM",  lr = 0.1, epochs = 100, save_dir = None):

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp)

    device  ="cuda" if torch.cuda.is_available() else "cpu"
    model = wrapped_model.model

    inp = inp.to(device)
    black = np.zeros((1,224,224,3))
    black = utils.numpy_to_tensor(black).to(device)
    white = np.ones((1,224,224,3))
    white = utils.numpy_to_tensor(white).to(device)
    rec_inp = (white-black)*torch.rand_like(inp)+black
    rec_inp.requires_grad = True


    optimizer = optim.Adam([rec_inp], lr = lr)

    wrapped_model.defense()

    for i in range(epochs):
        out_pre = model(rec_inp)

        loss = -out_pre[:,target].mean()


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        rec_inp.data.clamp_(black, white)
        print("epoch {}, loss: {:.7f}".format(str(i), loss.item()))
    utils.visualize_imgs(inp, rec_inp, save_dir)
    wrapped_model.remove_bhooks()

    return rec_inp.detach()



def explain(model, inp, target, area_mode = "ensemble", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False):
    device  ="cuda" if torch.cuda.is_available() else "cpu"

    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp)

    if area_mode == "ensemble":
        attr = torch.zeros(1,1,224,224).to(device)
        cur_attr = torch.zeros(1,1,224,224).to(device)
        for area in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if restart:
                cur_attr = generate_attr(wrapped_model, inp, target, area, attr = torch.zeros_like(cur_attr), defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func, binary = binary, threshold = threshold)
            else:
                cur_attr = generate_attr(wrapped_model, inp, target, area, attr = cur_attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func, binary = binary, threshold = threshold)
            if visualize:
                utils.visualize_attr(inp, cur_attr, "ensemble_{}".format(str(area)), save_dir)
            attr += cur_attr
        attr = attr/5
        #attr = generate_attr(img, wrapped_model, 0, config, cur_attr)
        if visualize:
            utils.visualize_attr(inp, cur_attr, "ensemble_sum", save_dir)
    elif area_mode == "l1":
        attr = torch.zeros(1,1,224,224).to(device)
        attr = generate_attr(wrapped_model, inp, target, 0, attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold)
        if visualize:
            utils.visualize_attr(inp, attr, "l1", save_dir)
    elif area_mode == "none":
        attr = torch.zeros(1,1,224,224).to(device)
        attr = generate_attr(wrapped_model, inp, target, 0, attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold, area_regulation = False)
        if visualize:
            utils.visualize_attr(inp, attr, "ensemble_none", save_dir)

    

    wrapped_model.remove_hook()
    return attr






    
