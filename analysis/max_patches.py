"""
Creates topk maximum activating patches (or images) for a specific unit in a neural network
"""

import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from nets import MNISTCNNBase, add_stats_layers_to_cnn_classifier
from lib.data_utils import get_static_emnist_dataloaders
from lib.utils import *
from analysis.torch_receptive_field.receptive_field import receptive_field, receptive_field_for_unit


def unravel_index(index, shape):
    """
    https://discuss.pytorch.org/t/how-to-do-a-unravel-index-in-pytorch-just-like-in-numpy/12987/2
    """
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


class MaxConvHook:
    def __init__(self, module, unit, topk):
        self.unit = unit
        self.topk = topk
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        feature_map = output[:, unit, :, :]
        self.max_values, self.ravelled_max_locations = torch.topk(torch.flatten(feature_map), k=self.topk)
        self.feature_map_shape = feature_map.shape
        assert feature_map[unravel_index(self.ravelled_max_locations[0], feature_map.shape)] == self.max_values[0]

    def close(self):
        self.hook.remove()


class MaxLinHook:
    def __init__(self, module, unit, topk):
        self.unit = unit
        self.topk = topk
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        feature = output[:, unit]
        self.max_values, self.max_locations = torch.topk(feature, k=self.topk)
        assert feature[self.max_locations[0]] == self.max_values[0]

    def close(self):
        self.hook.remove()


# Hardcoded flags
data_root = "./datasets/"
ckpt_dir = "./ckpts/emnist/"
output_dir = "/home/ian/meta-lstm/figs/max_patches/"
seed = 123
shift_name = "grass"  # "identity"
alg_name = "bufr"  # source_only

total_n_classes = 47
input_shape = [3, 28, 28]
adapt_classes = list(range(47))
tau = 0.01
batch_size = 256
learner = MNISTCNNBase(input_shape, total_n_classes)
n_workers = 4
pin_mem = True
dev = torch.device('cuda')

unit = 15  # The unit for which we get the max activating patch
topk = 4  # 16
assert topk in [1, 4, 9, 16]  # square number

if alg_name == "source_only":
    shift_ckpt_name = "pretrain-learner-150_DigitCNN_{}.pth.tar".format(seed)
elif alg_name == "bufr":
    shift_ckpt_name = "adapted-learner-30_{}_sgd_1.0_BUFR_fc_0.01_{}.pth.tar".format(shift_name, seed)
else:
    raise ValueError("Invalid algorithm name for calibration diagrams")

if alg_name == "bufr":
    add_stats_layers_to_cnn_classifier(learner, "soft_bins", "PSI", tau)
    for learner_stats_layer in learner.stats_layers:
        learner_stats_layer.calc_surprise = True
learner = learner.to(dev)
receptive_field_dict = receptive_field(learner.features, input_shape)
# # The receptive field does take into account padding but returns only the receptive field
# # that is within an image - so not always a square image will be returned - but this is okay


if shift_ckpt_name is not None:
    model_ckpt_path = os.path.join(ckpt_dir, shift_ckpt_name)
    if "pretrain" in shift_ckpt_name:
        _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)
    else:
        _, learner = load_ckpt('adapted-learner', learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)

# Hook first conv and first linear
for module in learner.modules():
    if isinstance(module, nn.Conv2d):
        print("Hooked module: {}".format(module))
        hooked_conv = MaxConvHook(module, unit, topk)
        break
for module in learner.modules():
    if isinstance(module, nn.Linear):
        print("Hooked module: {}".format(module))
        hooked_lin = MaxLinHook(module, unit, topk)
        break

ds_path = os.path.join(data_root, "EMNIST", shift_name)
tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, batch_size, False, n_workers, pin_mem)

# Source only trains on tr_dl and bufr adapts on tst_dl
if "pretrain" in shift_ckpt_name:
    training_dataloader = tr_dl
else:
    training_dataloader = tst_dl

learner.eval()
with torch.no_grad():
    conv_maxs_so_far = None
    lin_maxs_so_far = None
    for x_src, y_src in training_dataloader:
        x_src, y_src = x_src.to(dev), y_src.to(dev)

        learner(x_src)

        conv_maxs = hooked_conv.max_values.detach().cpu().numpy()
        lin_maxs = hooked_lin.max_values.detach().cpu().numpy()

        if conv_maxs_so_far is None:
            conv_maxs_so_far = conv_maxs
            patches = []
            for i in range(topk):
                unravelled = unravel_index(hooked_conv.ravelled_max_locations[i], hooked_conv.feature_map_shape)
                patch_location = (int(unravelled[0]), int(unravelled[1]), int(unravelled[2]))
                rf = receptive_field_for_unit(receptive_field_dict, layer="1", unit_position=patch_location[1:])
                patches.append(x_src[patch_location[0], :, int(rf[0][0]):int(rf[0][1]),
                               int(rf[1][0]):int(rf[1][1])].detach().cpu().numpy())

        if lin_maxs_so_far is None:
            lin_maxs_so_far = lin_maxs
            images = []
            for i in range(topk):
                images.append(x_src[hooked_lin.max_locations[i]].detach().cpu().numpy())
            print("Initialised. Continuing for loop...")
            continue

        for i, cm in enumerate(conv_maxs):
            if cm > np.min(conv_maxs_so_far):
                print("Old conv maxs {}".format(conv_maxs_so_far))
                min_idx = np.argmin(conv_maxs_so_far)
                conv_maxs_so_far[min_idx] = cm
                unravelled = unravel_index(hooked_conv.ravelled_max_locations[i], hooked_conv.feature_map_shape)
                patch_location = (int(unravelled[0]), int(unravelled[1]), int(unravelled[2]))
                rf = receptive_field_for_unit(receptive_field_dict, layer="1", unit_position=patch_location[1:])
                patches[min_idx] = x_src[patch_location[0], :, int(rf[0][0]):int(rf[0][1]),
                                                             int(rf[1][0]):int(rf[1][1])].detach().cpu().numpy()
                print("New conv max {}".format(conv_maxs_so_far))

        for i, lm in enumerate(lin_maxs):
            if lm > np.min(lin_maxs_so_far):
                print("Old lin maxs {}".format(lin_maxs_so_far))
                min_idx = np.argmin(lin_maxs_so_far)
                lin_maxs_so_far[min_idx] = lm
                images[min_idx] = x_src[hooked_lin.max_locations[i]].detach().cpu().numpy()
                print("New lin maxs {}".format(lin_maxs_so_far))


fpath = output_dir + "max_patch_{}_model_{}_unit_{}_top_{}.png".format(alg_name, shift_name, unit, topk)
fig, axs = plt.subplots(int(np.sqrt(topk)), int(np.sqrt(topk)))
for i in range(int(np.sqrt(topk))):
    for j in range(int(np.sqrt(topk))):
        patch = np.transpose(patches[i * int(np.sqrt(topk)) + j], (1, 2, 0))
        patch = (patch + 1.) / 2.  # denormalize
        axs[i, j].imshow(patch)
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticklabels([])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
# fig.subplots_adjust(hspace=0.05, wspace=0.05)
fig.savefig(fpath, bbox_inches='tight')
print("Saved fig to: {}".format(fpath))

fpath = output_dir + "max_img_{}_model_{}_unit_{}_top_{}.png".format(alg_name, shift_name, unit, topk)
fig, axs = plt.subplots(int(np.sqrt(topk)), int(np.sqrt(topk)))
for i in range(int(np.sqrt(topk))):
    for j in range(int(np.sqrt(topk))):
        image = np.transpose(images[i * int(np.sqrt(topk)) + j], (1, 2, 0))
        image = (image + 1.) / 2.  # denormalize
        axs[i, j].imshow(image)
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticklabels([])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
# fig.subplots_adjust(hspace=0.05, wspace=0.05)
fig.savefig(fpath, bbox_inches='tight')
print("Saved fig to: {}".format(fpath))
