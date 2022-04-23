import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from nets import MNISTCNNBase, add_stats_layers_to_cnn_classifier
from lib.data_utils import get_static_emnist_dataloaders
from lib.utils import *
from data.digits import get_mnistm_dataloaders

# Hardcoded flags
data_root = "./datasets/"
ckpt_dir = "./ckpts/emnist/"
output_dir = "/home/ian/meta-lstm/figs/tsne/"
seed = 123
shift_name = "crystals"
alg_name = "bufr"  # source_only
include_source_data = True
num_classes_to_plot = 5

total_n_classes = 47
input_shape = [3, 28, 28]
tau = 0.01
n_workers = 4
pin_mem = True
dev = torch.device('cuda')

adapt_classes = list(range(num_classes_to_plot))
batch_size = 100 * num_classes_to_plot  # use one batch of this size
cmap = matplotlib.cm.get_cmap('tab20')
if num_classes_to_plot > 10:
    raise ValueError("Colormap cannot handle more than 10 classes (20 colors)")

pretr_ckpt_name = "pretrain-learner-150_DigitCNN_{}.pth.tar".format(seed)
if alg_name == "source_only":
    shift_ckpt_name = "pretrain-learner-150_DigitCNN_{}.pth.tar".format(seed)
elif alg_name == "bufr":
    shift_ckpt_name = "adapted-learner-30_{}_sgd_1.0_BUFR_fc_0.01_{}.pth.tar".format(shift_name, seed)
else:
    raise ValueError("Invalid algorithm name for tsne")


# Source data on source model
if include_source_data:
    learner = MNISTCNNBase(input_shape, total_n_classes)
    learner = learner.to(dev)

    _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)

    # Gets first linear output - for plotting features
    for module in learner.modules():
        if isinstance(module, nn.Linear):
            print("Hooked module: {}".format(module))
            hooked_bn = FwdHook(module)
            break

    print(learner)

    ds_path = os.path.join(data_root, "EMNIST", "identity")
    tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, batch_size, True, n_workers, pin_mem)

    # Source only trains on tr_dl
    tr_batch = next(iter(tr_dl))
    x_src, y_src = tr_batch
    x_src, y_src = x_src.to(dev), y_src.to(dev)

    learner.eval()

    learner(x_src)
    src_features = hooked_bn.output.detach().cpu().numpy()
    src_labels = y_src.detach().cpu().numpy()


# Shift/Target data on adapted model (or source model if want source only)
learner = MNISTCNNBase(input_shape, total_n_classes)
if alg_name == "fr" or alg_name == "bufr":
    add_stats_layers_to_cnn_classifier(learner, "soft_bins", "PSI", tau)
    for learner_stats_layer in learner.stats_layers:
        learner_stats_layer.calc_surprise = True
learner = learner.to(dev)


if shift_ckpt_name is not None:
    model_ckpt_path = os.path.join(ckpt_dir, shift_ckpt_name)
    if "pretrain" in shift_ckpt_name:
        _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)
    else:
        _, learner = load_ckpt('adapted-learner', learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)


# Gets first linear output - for plotting features
for module in learner.modules():
    if isinstance(module, nn.Linear):
        print("Hooked module: {}".format(module))
        hooked_bn = FwdHook(module)
        break

print(learner)

# Create iterator dataloaders --------------
ds_path = os.path.join(data_root, "EMNIST", shift_name)
tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, batch_size, True, n_workers, pin_mem)

# We adapt on tst_dl
tst_batch = next(iter(tst_dl))
x_shift, y_shift = tst_batch
x_shift, y_shift = x_shift.to(dev), y_shift.to(dev)

learner.eval()
learner(x_shift)
shift_features = hooked_bn.output.detach().cpu().numpy()
shift_labels = y_shift.detach().cpu().numpy()


# Plotting
shift_colors = shift_labels / 10. + 1. / 20.
if include_source_data:
    features = np.concatenate([src_features, shift_features])
    labels = np.concatenate([src_labels, shift_labels])
    src_colors = src_labels / 10.
    colors = np.concatenate([src_colors, shift_colors])
else:
    features = shift_features
    labels = shift_labels
    colors = shift_colors

print("Features shape before t-sne {}. {} classes, {} total target samples, "
      "{} feature dimensionality.".format(features.shape, num_classes_to_plot, batch_size, features.shape[1]))
print("Labels shape {}.".format(labels.shape))

# Can play with TSNE perplexity to change the figures somewhat https://distill.pub/2016/misread-tsne/
# From experimentation 30 seems to be the best
embedded = TSNE(n_components=2, perplexity=30).fit_transform(features)
print("Features shape after t-sne {}.".format(embedded.shape))

fpath = output_dir + "tsne_{}_src_data_{}_{}_model.png".format(shift_name, include_source_data, alg_name)

fig = plt.figure()
ax = fig.gca()
ax.scatter(embedded[:, 0], embedded[:, 1], s=40.0, c=cmap(colors))
# Turn off tick labels
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(fpath, bbox_inches='tight')
print("Saved fig to: {}".format(fpath))
