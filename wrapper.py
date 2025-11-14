import torch
import torch.nn as nn

MID_LAYER = {
    'vgg19': 36,
    'vgg16': 30,
    'alexnet': 12,
    'googlenet': 183,
    'resnet18': 49,
    'resnet34': 89,
    'resnet50': 123,
    'resnet101': 242
}



def get_seq_model(layers):
    cut_point = None
    for i in range(len(layers)):
        if isinstance(layers[i], nn.Linear):
            cut_point = i
            break


    class CriticalModel(nn.Module):
        def __init__(self):
            super(CriticalModel, self).__init__()
            self.features = nn.Sequential(*layers[:cut_point])
            if cut_point is not None:
                self.classifiers = nn.Sequential(*layers[cut_point:])

        def forward(self, x):
            x = self.features(x)
            if hasattr(self, "classifiers"):
                x = x.reshape(x.size(0), -1)
                x = self.classifiers(x)
            return x
    return CriticalModel()


class Wrapper(object):

    # Constructor from a torch model.

    def __init__(self, model, defense_mode):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.defense_mode = defense_mode.upper()
        self._layers = []
        self._modules = []
        self._layers_names = []
        self._modules_names = []
        self.weight_save = dict()
        self.bias_save = dict()
        for name, layer in model.named_modules():
            if len(list(layer.children())) == 0:
                self._layers.append(layer)
                layer_str = str(layer)
                end_idx = layer_str.find("(")
                layer_str = layer_str[:end_idx]
                new_name = "{}: {}.{}".format(len(self._layers_names), name, layer_str)
                self._layers_names.append(new_name)
            else:
                self._modules.append(layer)
                layer_str = str(layer)
                end_idx = layer_str.find("(")
                layer_str = layer_str[:end_idx]
                new_name = "{}: {}.{}".format(len(self._modules_names), name, layer_str)
                self._modules_names.append(new_name)




    def layer_shape_hook(self, layer_idx):
        def hook(module, inp, out):
            self.layer_shapes[layer_idx] = out.shape
        return hook

    def module_shape_hook(self, layer_idx):
        def hook(module, inp, out):
            self.module_shapes[layer_idx] = out.shape
        return hook

    def init_layer_shape(self, inp):
        self.layer_shapes = [None for idx in range(len(self._layers))]
        self.model.eval()
        shape_hooks = []
        for i, layer in enumerate(self._layers):
            shape_hooks.append(layer.register_forward_hook(self.layer_shape_hook(i)))
        with torch.no_grad():
            self.model(inp)
        for hook in shape_hooks:
            hook.remove()

    def init_module_shape(self, inp):
        self.module_shapes = [None for idx in range(len(self._modules))]
        self.model.eval()
        shape_hooks = []
        for i, module in enumerate(self._modules):
            shape_hooks.append(module.register_forward_hook(self.module_shape_hook(i)))
        with torch.no_grad():
            self.model(inp)
        for hook in shape_hooks:
            hook.remove()




    def reorder_hook(self, idx):
        def hook(module, inp, out):
            self.order_indices.append(idx)
        return hook

    def reorder_layers(self, inp):
        self.order_layers = []
        self.order_layers_names = []
        self.order_indices = []
        hooks = []
        for i, layer in enumerate(self._layers):
            hooks.append(layer.register_forward_hook(self.reorder_hook(i)))

        with torch.no_grad():
            self.model(inp)
        for hook in hooks:
            hook.remove()

        for idx in self.order_indices:
            self.order_layers.append(self._layers[idx])
            self.order_layers_names.append(self._layers_names[idx])

    def act_hook(self):
        def hook(module, inp, out):
            self.act_out.append(out.detach().clone())
        return hook



    def defense_forward_hook(self):
        def hook(module, inp, out):
            if self.act_idx == self.max_idx:
                self.act_idx = 0
            act = self.act_out[self.act_idx]
            if self.defense_mode == "IVM":
                clip = out <= act
            elif self.defense_mode == "AVM" or self.defense_mode == "VM":
                clip = out >= act
            elif self.defense_mode == "IBM":
                clip = act > 0
            elif self.defense_mode == "FGVIS":
                bu = torch.clamp(act, min=0)
                bl = torch.clamp(act, max=0)
                clip = (out >= bl) * (out <= bu)
            else:
                raise RuntimeError("defense mode not supported")
            if len(self.clips) == self.max_idx:
                self.clips[self.act_idx] = clip.detach()
            else:
                self.clips.append(clip)
            self.act_idx += 1

        return hook

    def defense_backward_hook(self):
        def hook(module, grad_inp, grad_out):
            if self.act_idx == 0:
                self.act_idx = self.max_idx
            self.act_idx -= 1
            clip = self.clips[self.act_idx]
            clip = clip.to(self.device)
            if self.defense_mode == "IBM" or self.defense_mode == "IVM":
                cur_clip = grad_inp[0] > 0
                new_clip = torch.logical_or(cur_clip, clip)
                ret = (torch.mul(grad_inp[0], new_clip),)
            elif self.defense_mode == "AVM":
                cur_clip = grad_out[0] < 0
                new_clip = torch.logical_or(cur_clip, clip)
                ret = (torch.mul(grad_inp[0], new_clip),)
            elif self.defense_mode == "VM":
                cur_clip = grad_inp[0] > 0
                clip1 = torch.logical_and(cur_clip, clip)
                clip2 = torch.logical_not(torch.logical_or(cur_clip, clip))
                new_clip = torch.logical_or(clip1, clip2)
                ret = (torch.mul(grad_inp[0], new_clip),)
            elif self.defense_mode == "FGVIS":
                ret = (torch.mul(grad_inp[0], clip),)
            return ret
        return hook

    def remove_bhooks(self):
        if hasattr(self, "bhooks"):
            for hook in self.bhooks:
                hook.remove()
        self.bhooks = []

    def defense(self):
        if self.defense_mode == "NONE":
            return
        if hasattr(self, "bhooks"):
            for hook in self.bhooks:
                hook.remove()

        self.bhooks = []
        for i in range(self.start, self.end):
            layer = self._layers[i]
            if isinstance(layer, nn.Linear):
                break
            if isinstance(layer, nn.ReLU):
                self.bhooks.append(layer.register_full_backward_hook(self.defense_backward_hook()))





    def pre_defense(self, inp, start=None, end=None):
        if self.defense_mode == "NONE":
            return

        if start is None:
            start = 0
        if end is None:
            end = len(self._layers)

        self.start = start
        self.end = end

        fhook = []
        self.act_out = []
        self.clips = []
        self.act_idx = 0
        inp = inp.to(self.device)
        for i in range(start, end):
            layer = self._layers[i]
            if isinstance(layer, nn.Linear):
                break
            if isinstance(layer, nn.ReLU):
                layer.inplace = False
                fhook.append(layer.register_forward_hook(self.act_hook()))

        with torch.no_grad():
            _ = self.model(inp)
        for hook in fhook:
            hook.remove()
        self.max_idx = len(self.act_out)

        self.fhooks = []
        for i in range(start, end):
            layer = self._layers[i]
            if isinstance(layer, nn.Linear):
                break
            if isinstance(layer, nn.ReLU):
                self.fhooks.append(layer.register_forward_hook(self.defense_forward_hook()))


    def remove_hook(self):
        self.mid_out = []
        self.clips = []
        self.act_idx = 0
        self.max_idx = None
        if hasattr(self, "fhooks"):
            for hook in self.fhooks:
                hook.remove()
        if hasattr(self, "bhooks"):
            for hook in self.bhooks:
                hook.remove()
        if hasattr(self, "loss_hooks"):
            for hook in self.loss_hooks:
                hook.remove()
        self.fhooks = []
        self.bhooks = []
        self.loss_hooks = []


    def sanity_random(self, idx):
        self.weight_save = dict()
        self.bias_save = dict()
        m = self._layers[idx]
        self.weight_save[idx] = m.weight.clone()
        if m.bias is not None:
            self.bias_save[idx] = m.bias.clone()
        m.reset_parameters()

    def model_recover(self):
        with torch.no_grad():
            for k,v in self.weight_save.items():
                cur_layer = self._layers[k]
                cur_layer.weight.copy_(v)
                if k in self.bias_save.keys():
                    cur_layer.bias.copy_(self.bias_save[k])

        self.weight_save = dict()
        self.bias_save = dict()


    def divide(self, model_name):
        self.model_name = model_name
        self.mid = MID_LAYER[self.model_name]

        if self.model_name == "resnet18":
            layers = self._layers[:4] + [self._modules[1], self._modules[4], self._modules[8], self._modules[12]]
        elif self.model_name == "resnet34":
            layers = self._layers[:4] + [self._modules[1], self._modules[5], self._modules[11], self._modules[19]]
        elif self.model_name == "resnet50":
            layers = self._layers[:4] + [self._modules[1], self._modules[6], self._modules[12], self._modules[20]]
        elif self.model_name == "resnet101":
            layers = self._layers[:4] + [self._modules[1], self._modules[6], self._modules[12], self._modules[37]]
        elif self.model_name == "googlenet":
            layers = [self._modules[1], self._layers[3], self._modules[2], self._modules[3], self._layers[10], self._modules[4], self._modules[14], self._layers[49], self._modules[24], self._modules[34], self._modules[44], self._modules[54], self._modules[64], self._layers[145], self._modules[74], self._modules[84]]
        else:
            layers = self._layers[:self.mid+1]

        sub_model1 = get_seq_model(layers)

        layers = self._layers[self.mid+1:]
        sub_model2 = get_seq_model(layers)

        return sub_model1, sub_model2



