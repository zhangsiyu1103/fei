
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

        def forward(self,x):
            x= self.features(x)
            if hasattr(self, "classifiers"):
                x = x.reshape(x.size(0), -1)
                x = self.classifiers(x)
            return x
    return CriticalModel()

def get_reg_layer(act):
    class RegLayer(nn.Module):
        def __init__(self):
            super(RegLayer, self).__init__()
            #self.act = act > 0
            self.act = act

        def forward(self,x):
            bu = torch.clamp(self.act, min = 0)
            bl = torch.clamp(self.act, max = 0)
            ret = torch.clamp(x, min=bl, max=bu)
            #ret = torch.mul(x, self.act)
            return ret
    return RegLayer()

class Wrapper(object):

    # Constructor from a torch model.

    def __init__(self,model, defense_mode):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.defense_mode = defense_mode.upper()
        self._layers=[]
        self._modules=[]
        self._layers_names=[]
        self._modules_names=[]
        self.weight_save = dict()
        self.bias_save = dict()
        for  name,layer in model.named_modules():
            if len(list(layer.children()))==0:
                self._layers.append(layer)
                layer_str = str(layer)
                end_idx= layer_str.find("(")
                layer_str = layer_str[:end_idx]
                new_name = "{}: {}.{}".format(len(self._layers_names), name, layer_str)
                #self._layers_names.append(name)
                self._layers_names.append(new_name)
            else:
                self._modules.append(layer)
                layer_str = str(layer)
                end_idx= layer_str.find("(")
                layer_str = layer_str[:end_idx]
                new_name = "{}: {}.{}".format(len(self._modules_names), name, layer_str)
                self._modules_names.append(new_name)




    def layer_shape_hook(self,layer_idx):
        def hook(module,inp,out):
            self.layer_shapes[layer_idx] = out.shape
        return hook

    def module_shape_hook(self,layer_idx):
        def hook(module,inp,out):
            self.module_shapes[layer_idx] = out.shape
        return hook

    def init_layer_shape(self,inp):
        self.layer_shapes = [None for idx in range(len(self._layers))]
        self.model.eval()
        shape_hooks = []
        for i,l in enumerate(self._layers):
            shape_hooks.append(l.register_forward_hook(self.layer_shape_hook(i)))
        with torch.no_grad():
            self.model(inp)
        for hook in shape_hooks:
            hook.remove()

    def init_module_shape(self,inp):
        self.module_shapes = [None for idx in range(len(self._modules))]
        self.model.eval()
        shape_hooks = []
        for i,m in enumerate(self._modules):
            shape_hooks.append(m.register_forward_hook(self.module_shape_hook(i)))
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
        for i, l in enumerate(self._layers):
            hooks.append(l.register_forward_hook(self.reorder_hook(i)))

        with torch.no_grad():
            self.model(inp)
        for hook in hooks:
            hook.remove()

        for idx in self.order_indices:
            self.order_layers.append(self._layers[idx])
            self.order_layers_names.append(self._layers_names[idx])

    def act_hook(self):
        def hook(module,inp,out):
            #self.act_out.append(out.detach().cpu().clone())
            self.act_out.append(out.detach().clone())

        return hook


    def defense_forward_hook(self):
        def hook(module,inp,out):
            if self.act_idx == self.max_idx:
                self.act_idx = 0
            act = self.act_out[self.act_idx]
            if self.defense_mode == "IVM":
                clip = out <= act
            elif self.defense_mode == "AVM" or self.defense_mode == "VM":
                clip = out >= act
            elif self.defense_mode == "IBM":
                clip = act > 0
                clip = clip.detach()
            else:
                raise RuntimeError("defense mode not supported")
            if len(self.clips) == self.max_idx:
                self.clips[self.act_idx] = clip
            else:
                self.clips.append(clip)
            self.act_idx += 1

        return hook

    def defense_backward_hook(self):
        def hook(module,grad_inp,grad_out):
            if self.act_idx == 0:
                self.act_idx = self.max_idx
            self.act_idx -= 1
            #clip = self.clips[self.act_idx].to(self.device)
            clip = self.clips[self.act_idx]
            clip = clip.to(self.device)
            if self.defense_mode == "IBM" or self.defense_mode == "IAM":
                #print("relaxed")
                #clip = clip.to(self.device)
                cur_clip = grad_inp[0] > 0
                new_clip = torch.logical_or(cur_clip, clip)
                ret = (torch.mul(grad_inp[0], new_clip),)
                #print("relaxed")
            elif self.defense_mode == "AVM":
                #print("reverse")
                #clip = clip.to(self.device)
                cur_clip = grad_out[0] < 0
                new_clip = torch.logical_or(cur_clip, clip)
                ret = (torch.mul(grad_inp[0], new_clip),)
            elif self.defense_mode == "VM":
                cur_clip = grad_inp[0] > 0
                clip1 = torch.logical_and(cur_clip, clip)
                clip2 = torch.logical_not(torch.logical_or(cur_clip, clip))
                new_clip = torch.logical_or(clip1, clip2)
                ret = (torch.mul(grad_inp[0], new_clip),)
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
        #for i, l in enumerate(self._layers):
        for i in range(self.start, self.end):
            l = self._layers[i]
            if isinstance(l, nn.Linear):
                break
            if isinstance(l, nn.ReLU):
                self.bhooks.append(l.register_full_backward_hook(self.defense_backward_hook()))


        #return self.model


    def pre_defense(self, inp, start = None, end =None):
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
            l = self._layers[i]
            if isinstance(l, nn.Linear):
                break
            if isinstance(l, nn.ReLU):
                l.inplace = False
                fhook.append(l.register_forward_hook(self.act_hook()))

        with torch.no_grad():
            out = self.model(inp)
        for hook in fhook:
            hook.remove()
        self.max_idx = len(self.act_out)

        self.fhooks = []
        for i in range(start, end):
            l = self._layers[i]
            if isinstance(l, nn.Linear):
                break
            if isinstance(l, nn.ReLU):
                self.fhooks.append(l.register_forward_hook(self.defense_forward_hook()))


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
        self.fhooks = []
        self.bhooks = []


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







