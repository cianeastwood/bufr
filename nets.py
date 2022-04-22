"""
Network architectures and functionality for adding stats layers to networks
"""

from __future__ import division, print_function, absolute_import
import torch
import torch.nn as nn
import numpy as np
import torch.nn.utils.weight_norm as weightNorm
from lib.stats_layers import *
from lib.utils import weights_init
import torch.nn.functional as F


def get_learner_param_layers(model):
    layer_names = []
    param_layers = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)):
            layer_names.append(n)
            param_layers.append(m)
    return layer_names, param_layers


def learner_distances(init_learner, final_learner, distance_type="mean", zero_atol=1e-4, ret_layer_names=False,
                      is_tracked_net=True):
    """
    :param init_learner: learner before updates.
    :param final_learner: learner after updates.
    :param distance_type: type of distance (summary) to return. Choice: per-unit, mean or fract_moved.
    :param zero_atol: effective zero, i.e. absolute tolerance when checking values have not moved (distance=0).
    :param ret_layer_names: return a list of layer/module names.
    :param is_tracked_net: init_learner and final_learner are instances of TrackedNet.
    :return: list of layer distances.
    """

    all_ds = []
    m_names = []
    if is_tracked_net:
        init_learner_param_layers = init_learner.get_param_layers()
        final_learner_param_layers = final_learner.get_param_layers()
    else:
        names, init_learner_param_layers = get_learner_param_layers(init_learner)
        names, final_learner_param_layers = get_learner_param_layers(final_learner)

    for m_i, m_f in zip(list(init_learner_param_layers), list(final_learner_param_layers)):
        # Calculate distance
        if 'weight' in m_i._parameters.keys():
            w_i, w_f = m_i._parameters['weight'].detach(), m_f._parameters['weight'].detach()
            sum_axes = list(range(len(w_i.shape)))[1:]  # [1,2,3] for conv, [1] for linear, [] for batch norm
            if sum_axes != []:
                d_2s = torch.square(w_i - w_f).sum(sum_axes)
            else:
                d_2s = torch.square(w_i - w_f)
        else:  # for weight norm
            w_i_g, w_f_g = m_i._parameters['weight_g'].detach(), m_f._parameters['weight_g'].detach()
            w_i_v, w_f_v = m_i._parameters['weight_v'].detach(), m_f._parameters['weight_v'].detach()
            sum_axes = list(range(len(w_i_g.shape)))[1:]  # [1,2,3] for conv, [1] for linear
            d_2s = torch.square(w_i_g - w_f_g).sum(sum_axes) + torch.square(w_i_v - w_f_v).sum(sum_axes)
        if m_i._parameters['bias'] is not None:
            b_i, b_f = m_i._parameters['bias'].detach(), m_f._parameters['bias'].detach()
            d_2s += torch.square(b_i - b_f)  # square dist moved, weights + bias
        else:
            pass

        ds = torch.sqrt(d_2s)
        m_names.append(str(m_i).split(",")[0] + ")")

        # Append desired value (value "type")
        if distance_type == "per_unit":
            all_ds.append(ds.item())
        elif distance_type == "mean":
            all_ds.append(torch.mean(ds).item())
        elif distance_type == "max":
            all_ds.append(torch.max(ds).item())
        elif distance_type == "median":
            all_ds.append(torch.median(ds).item())
        elif distance_type == "fract_moved":
            fract_of_units_moved = torch.sum(torch.where(ds >= zero_atol, 1., 0.)) / ds.size()[0]
            all_ds.append(fract_of_units_moved.item())
        elif distance_type == "all":
            fract_of_units_moved = torch.sum(torch.where(ds >= zero_atol, 1., 0.)) / ds.size()[0]
            all_ds.append([torch.mean(ds).item(), torch.max(ds).item(), fract_of_units_moved.item()])
        else:
            raise ValueError("Invalid distance type {0}".format(type))

    if ret_layer_names:
        return all_ds, m_names

    return all_ds

## --------- CNNs --------- ##

def get_num_features(m, out=True):
    if isinstance(m, nn.Conv2d):
        if out:
            return m.out_channels
        else:
            return m.in_channels
    elif isinstance(m, nn.Linear):
        if out:
            return m.out_features
        else:
            return m.in_features
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        if m.affine:    # affine --> has learnable parameters
            return m.num_features
        else:
            return None
    else:
        return None


def add_stats_layers_to_module_list(module_list, stats_layer_type="bins", surprise_score="PSI", tau=0.01):
    new_module_list = []
    added_stats_layers = []
    stats_2d, stats_1d = get_stats_layer_by_name(stats_layer_type)
    for i in range(len(module_list)):
        #  Add module to new list
        m = module_list[i]
        n_features = get_num_features(m)
        new_module_list.append(m)
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):  # don't track BN modules
            continue
        if n_features is None:      # don't track modules with no parameters
            continue

        # Select stats layer
        stats_layer = stats_2d if "2d" in str(m).lower() else stats_1d
        if stats_layer_type == "soft_bins":
            stats_layer = stats_layer(n_features, surprise_score=surprise_score, tau=tau)
        else:
            stats_layer = stats_layer(n_features, surprise_score=surprise_score)

        # Add stats layer in desired position
        new_module_list.append(stats_layer)
        added_stats_layers.append(stats_layer)

    return nn.Sequential(*new_module_list), added_stats_layers


def add_stats_layers_to_cnn_classifier(net, stats_layer="bins", surprise_score="PSI", tau=0.01):
    net.stats_layers = []
    net.stats_layer = stats_layer
    net.classifier, added_stats_layers = add_stats_layers_to_module_list(net.classifier, stats_layer, surprise_score,
                                                                         tau)
    net.stats_layers.extend(added_stats_layers)


def add_stats_layers_to_cnn_everywhere(net, stats_layer="bins", surprise_score="PSI", tau=0.01):
    net.stats_layers = []
    net.stats_layer = stats_layer
    net.features, added_stats_layers = add_stats_layers_to_module_list(net.features, stats_layer, surprise_score,
                                                                       tau)
    net.stats_layers.extend(added_stats_layers)
    net.classifier, added_stats_layers = add_stats_layers_to_module_list(net.classifier, stats_layer, surprise_score,
                                                                         tau)
    net.stats_layers.extend(added_stats_layers)


class MNISTCNNBase(nn.Module):
    # Based on DTN from shot but not exactly the same
    def __init__(self, input_shape, n_classes, act_fn=nn.ReLU(), weight_norm=False):
        super(MNISTCNNBase, self).__init__()
        self.act_fn = act_fn
        self.features = nn.Sequential(
            # conv 1: (28x28 --> 14x14)
            nn.Conv2d(3, 64, 5, padding=2, stride=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            self.act_fn,

            # conv 2: (14x14 --> 8x8)
            nn.Conv2d(64, 128, 3, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            self.act_fn,

            # conv 3: (8x8 --> 5x5)
            nn.Conv2d(128, 256, 3, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            self.act_fn
        )

        self.clr_in = 5
        if weight_norm:  # for SHOT baseline
            self.classifier = nn.Sequential(
                nn.Linear(256 * self.clr_in * self.clr_in, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.5),
                self.act_fn,

                weightNorm(nn.Linear(128, n_classes)),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(256 * self.clr_in * self.clr_in, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(p=0.5),
                self.act_fn,

                nn.Linear(128, n_classes),
            )

        self.apply(weights_init)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


## --------- Resnets --------- ##

def append_stats_layer_to_module(m, n_out_features, n_in_features=None, stats_layer_type="bins", surprise_score="PSI",
                                 tau=0.01):
    # Setup
    stats_2d, stats_1d = get_stats_layer_by_name(stats_layer_type)
    if hasattr(m, "layer"):
        m_list = m.layer
        stats_layer = stats_2d

    elif isinstance(m, nn.Sequential):
        m_list = list(m)
        stats_layer = stats_2d

    elif isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.BatchNorm1d)):
        m_list = [m]
        stats_layer = stats_2d if "2d" in str(m).lower() else stats_1d

    else:
        raise NotImplementedError("Cannot add stats layer to track module {0}".format(m))

    # Add stats layer/module after m
    if stats_layer_type == "soft_bins":
        stats_m = stats_layer(n_out_features, surprise_score=surprise_score, tau=tau)
    else:
        stats_m = stats_layer(n_out_features, surprise_score=surprise_score)
    stats_m_list = [stats_m]
    m_list.append(stats_m)

    # Add stats layer/module before m
    if n_in_features is not None:  # add stats layer before
        if stats_layer_type == "soft_bins":
            stats_m_input = stats_layer(n_in_features, surprise_score=surprise_score, tau=tau)
        else:
            stats_m_input = stats_layer(n_in_features, surprise_score=surprise_score)
        m_list = [stats_m_input] + m_list
        stats_m_list = [stats_m_input] + stats_m_list

    # if isinstance(m, nn.Linear):
    #     stats_m_softmax = SoftBinStatsSoftmax(n_out_features, surprise_score=surprise_score)
    #     stats_m_list.append(stats_m_softmax)
    #     m_list.append(stats_m_softmax)

    # Create new list
    m_list = nn.Sequential(*m_list)

    if hasattr(m, "layer"):
        m.layer = m_list
        return m

    return m_list, stats_m_list


def add_stats_layer_to_resnet_named_modules(model, module_names, module_features_out, module_features_in=None,
                                            stats_layer_type="bins", surprise_score="PSI", tau=0.01):
    module_features_in = [None] * len(module_names) if module_features_in is None else module_features_in
    model.stats_layers = []

    for module_name, n_out_features, n_in_features in zip(module_names, module_features_out, module_features_in):
        named_module = getattr(model, module_name)
        tracked_named_module, stats_layers = append_stats_layer_to_module(named_module, n_out_features, n_in_features,
                                                                          stats_layer_type, surprise_score, tau)
        setattr(model, module_name, tracked_named_module)
        model.stats_layers.extend(stats_layers)

    model.stats_layer_type = stats_layer_type


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    """
    Cifar sizes, printing x.shape and every out.shape in the forward pass
    torch.Size([128, 3, 32, 32])    - Input                                                                                                                                                                           
    torch.Size([128, 64, 32, 32])   - First layers (conv+bn) output                                                                                                                                                                             
    torch.Size([128, 64, 32, 32])   - self.layer1 output                                                                                                                                                                            
    torch.Size([128, 128, 16, 16])  - self.layer2 output                                                                                                                                                                                
    torch.Size([128, 256, 8, 8])    - self.layer3 output                                                                                                                                                                                
    torch.Size([128, 512, 4, 4])    - self.layer4 output                                                                                                                                                                                
    torch.Size([128, 512, 1, 1])    - avg_pool output                                                                                                                                                                            
    torch.Size([128, 512])          - reshape output                                                                                                                                                                            
    torch.Size([128, 10])           - linear output
    
    Camelyon using this resnet:
    torch.Size([128, 3, 96, 96])
    torch.Size([128, 64, 96, 96])
    torch.Size([128, 64, 96, 96])
    torch.Size([128, 128, 48, 48])
    torch.Size([128, 256, 24, 24])
    torch.Size([128, 512, 12, 12])
    torch.Size([128, 512, 3, 3])
    torch.Size([128, 4608])   <- breaks here as this is should be 128, 512 after reshape then go to 
    
    If we use the resnet in nets_wilds:
    torch.Size([128, 3, 96, 96])   - Input                                                                                                                                                                             
    torch.Size([128, 64, 48, 48])  - First layers (conv+bn) output                                                                                                                                                                             
    torch.Size([128, 64, 24, 24])  - Max pool output                                                                                                                                                                             
    torch.Size([128, 64, 24, 24])  - self.layer1 output                                                                                                                                                                              
    torch.Size([128, 128, 12, 12]) - self.layer2 output                                                                                                                                                                              
    torch.Size([128, 256, 6, 6])   - self.layer3 output                                                                                                                                                                              
    torch.Size([128, 512, 3, 3])   - self.layer4 output                                                                                                                                                                               
    torch.Size([128, 512, 1, 1])   - AdaptiveAvgPool2d output                                                                                                                                                                           
    torch.Size([128, 512])         - reshape output                                                                                                                                                                             
    torch.Size([128, 1])           - linear output

    """


def ResNet18(n_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=n_classes)


def ResNet34(n_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=n_classes)


def ResNet50(n_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=n_classes)


def ResNet101(n_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=n_classes)
