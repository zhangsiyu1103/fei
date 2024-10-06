import sys
import numpy as np
import torch
from evaluate import *
from utils import *


def eval_single_perturbation(model, attrs, inp, result):
    ins_score = eval_attr_single(model, inp, attrs, "ins_rand")
    del_score = eval_attr_single(model, inp, attrs, "del_rand")
    if "ins_rand" in result:
        result["ins_rand"].append(ins_score)
    else:
        result["ins_rand"] = [ins_score]
    if "del_rand" in result:
        result["del_rand"].append(del_score)
    else:
        result["del_rand"] = [del_score]

    ins_score = eval_attr_single(model, inp, attrs, "ins_blur")
    del_score = eval_attr_single(model, inp, attrs, "del_blur")
    if "ins_blur" in result:
        result["ins_blur"].append(ins_score)
    else:
        result["ins_blur"] = [ins_score]
    if "del_blur" in result:
        result["del_blur"].append(del_score)
    else:
        result["del_blur"] = [del_score]
    ins_score = eval_attr_single(model, inp, attrs, "ins_zero")
    del_score = eval_attr_single(model, inp, attrs, "del_zero")
    if "ins_zero" in result:
        result["ins_zero"].append(ins_score)
    else:
        result["ins_zero"] = [ins_score]
    if "del_zero" in result:
        result["del_zero"].append(del_score)
    else:
        result["del_zero"] = [del_score]



def eval_attr_single(model, inp, attr, method):
    klen = 11
    ksig = 5
    kern = gkern(klen,ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
    if method == "ins_zero":
        step = inp.shape[-1]
        attr_np = attr.detach().cpu().numpy()
        ins_metric = CausalMetric(model, "ins", 224*8, torch.zeros_like)
        score = ins_metric.evaluate(inp.detach().cpu(), attr_np,inp.shape[0])
        score = auc(score.mean(1))
    elif method == "del_zero":
        step = inp.shape[-1]
        attr_np = attr.detach().cpu().numpy()
        del_metric = CausalMetric(model, "del", 224*8, torch.zeros_like)
        score = del_metric.evaluate(inp.detach().cpu(), attr_np,inp.shape[0])
        score = auc(score.mean(1))
    elif method == "ins_rand":
        step = inp.shape[-1]
        attr_np = attr.detach().cpu().numpy()
        ins_metric = CausalMetric(model, "ins", step,torch.randn_like)
        score = ins_metric.evaluate(inp.detach().cpu(), attr_np,inp.shape[0])
        score = auc(score.mean(1))
    elif method == "del_rand":
        step = inp.shape[-1]
        attr_np = attr.detach().cpu().numpy()
        del_metric = CausalMetric(model, "del", 224*8, torch.randn_like)
        score = del_metric.evaluate(inp.detach().cpu(), attr_np,inp.shape[0])
        score = auc(score.mean(1))
    elif method == "ins_blur":
        step = inp.shape[-1]
        attr_np = attr.detach().cpu().numpy()
        ins_metric = CausalMetric(model, "ins", 224*8, blur)
        score = ins_metric.evaluate(inp.detach().cpu(), attr_np,inp.shape[0])
        score = auc(score.mean(1))
    elif method == "del_blur":
        step = inp.shape[-1]
        attr_np = attr.detach().cpu().numpy()
        del_metric = CausalMetric(model, "del", 224*8 ,blur)
        score = del_metric.evaluate(inp.detach().cpu(), attr_np,inp.shape[0])
        score = auc(score.mean(1))
    return score

