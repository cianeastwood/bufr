import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from nets import MNISTCNNBase, add_stats_layers_to_cnn_classifier
from lib.data_utils import get_static_emnist_dataloaders
from lib.utils import *

# Hardcoded flags
data_root = "./datasets/"
ckpt_dir = "./ckpts/emnist/"
output_dir = "/home/ian/meta-lstm/figs/calibration/"
seed = 123
shift_name = "sky"  # "shot_noise"
alg_name = "im"  #  "fr", "bufr", "source_only"

total_n_classes = 47
input_shape = [3, 28, 28]
adapt_classes = list(range(47))
tau = 0.01
batch_size = 256
learner = MNISTCNNBase(input_shape, total_n_classes)
n_workers = 4
pin_mem = True
dev = torch.device('cuda')

if alg_name == "source_only":
    shift_ckpt_name = "pretrain-learner-150_DigitCNN_{}.pth.tar".format(seed)
elif alg_name == "im":
    shift_ckpt_name = "adapted-learner-150_{}_sgd_0.1_IM_{}.pth.tar".format(shift_name, seed)
elif alg_name == "fr":
    shift_ckpt_name = "adapted-learner-150_{}_sgd_1.0_FR_fc_0.01_{}.pth.tar".format(shift_name, seed)
elif alg_name == "bufr":
    shift_ckpt_name = "adapted-learner-30_{}_sgd_1.0_BUFR_fc_0.01_{}.pth.tar".format(shift_name, seed)
else:
    raise ValueError("Invalid algorithm name for calibration diagrams")


if alg_name == "fr" or alg_name == "bufr":
    add_stats_layers_to_cnn_classifier(learner, "soft_bins", "PSI", tau)
    for learner_stats_layer in learner.stats_layers:
        learner_stats_layer.calc_surprise = True
learner = learner.to(dev)
if shift_ckpt_name is not None:
    model_ckpt_path = os.path.join(ckpt_dir, shift_ckpt_name)
    ckpt = torch.load(model_ckpt_path, map_location=dev)
    if "pretrain" in shift_ckpt_name:
        _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)
    else:
        _, learner = load_ckpt('adapted-learner', learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)

ds_path = os.path.join(data_root, "EMNIST", shift_name)
tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, batch_size, False, n_workers, pin_mem)

# This code is as in expected_calibration_error in utils.py but we need access to some variables inside the function
num_bins = 10
bin_size = 1. / num_bins
bin_indices = np.linspace(0, 1, num_bins + 1)
bin_confidences_sum = np.zeros(num_bins)
bin_accuracies_sum = np.zeros(num_bins)
bin_counts = np.zeros(num_bins)
learner.eval()
with torch.no_grad():
    # Calibration error always calculated using tst_dl
    for x_src, y_src in tst_dl:
        x_src, y_src = x_src.to(dev), y_src.to(dev)

        outputs = learner(x_src)
        outputs = nn.Softmax(dim=1)(outputs)
        confidences, predictions = torch.max(outputs, 1)
        confidences = confidences.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
        targets = y_src.detach().cpu().numpy()

        confidence_bins = np.digitize(confidences, bin_indices, right=True)
        if np.any(confidence_bins == 0) or np.any(confidence_bins > num_bins):
            raise RuntimeError("It should be impossible for the softmax outputs to be <=0 or >1")
        # Every sample now has a bin index between 1 and num_bins (inclusive)

        for i in range(len(predictions)): # Probably can be vectorised but this is clearer for now
            bin_confidences_sum[confidence_bins[i] - 1] += confidences[i]
            bin_counts[confidence_bins[i] - 1] += 1
            bin_accuracies_sum[confidence_bins[i] - 1] += predictions[i] == targets[i]

    assert np.sum(bin_counts) == len(tst_dl) * batch_size

    print("Average test accuracy = {:6.3f}%".format(100 * np.sum(bin_accuracies_sum) / np.sum(bin_counts)))

    total_count = np.sum(bin_counts)
    bin_counts[np.where(bin_counts == 0)] += 1  # as accuracy or conf sum is zero anyway this just stops a nan error
    per_bin_accuracies = bin_accuracies_sum / bin_counts
    per_bin_confidences = bin_confidences_sum / bin_counts

    # Eqn 3 in https://arxiv.org/pdf/1706.04599.pdf
    ece = np.sum((bin_counts / total_count) * np.abs(per_bin_accuracies - per_bin_confidences))
    print("Expected Calibration Error = {:6.3f}%".format(100 * ece))

    # Eqn 5 in https://arxiv.org/pdf/1706.04599.pdf
    mce = np.max(np.abs(per_bin_accuracies - per_bin_confidences))
    print("Maximum Calibration Error = {:6.3f}%".format(100 * mce))

    print("Per bin accuracies:")
    print(per_bin_accuracies)
    print("Per bin confidences:")
    print(per_bin_confidences)

    # Reliability Diagram
    # Some of this code based on https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py
    # Figure set up
    SMALL_SIZE = 10
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 30
    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fpath = output_dir + "reliability_diagram_{}_model_{}.png".format(alg_name, shift_name)
    fig = plt.figure()
    ax = fig.gca()
    acc_plt = ax.bar(bin_indices[:-1], per_bin_accuracies, width=bin_size, align="edge",
                     edgecolor="black", color="b", alpha=1.0, linewidth=3,
                     label="Accuracy")

    gap_plt = ax.bar(bin_indices[:-1], np.abs(per_bin_accuracies - per_bin_confidences),
                     bottom=np.minimum(per_bin_accuracies, per_bin_confidences), width=bin_size, align="edge",
                     edgecolor=[1,0,0,0.8], color=[1,0,0,0.3], linewidth=3, hatch="/", label="Gap")

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.68, " ECE:{:4.1f}%".format(100 * ece), color="black",
            ha="left", va="center", transform=ax.transAxes)
    ax.text(0.02, 0.58, " MCE:{:4.1f}%".format(100 * mce), color="black",
            ha="left", va="center", transform=ax.transAxes)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(handles=[acc_plt, gap_plt], loc='upper left')
    fig.savefig(fpath, bbox_inches='tight')
    print("Saved fig to: {}".format(fpath))

    # Confidence Histogram
    fpath = output_dir + "confidence_histogram_{}_model_{}.png".format(alg_name, shift_name)
    fig = plt.figure()
    ax = fig.gca()
    ax.bar(bin_indices[:-1], bin_counts / total_count, width=bin_size, align="edge",
           edgecolor="black", color="b", alpha=1.0, linewidth=3)

    acc_plt = ax.axvline(x=np.sum(bin_accuracies_sum) / total_count, ls="dotted", lw=3,
                         c="black", label="Avg. Accuracy")
    conf_plt = ax.axvline(x=np.sum(bin_confidences_sum) / total_count, ls="dashed", lw=3,
                          c="red", label="Avg. confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Fraction of Samples")
    ax.legend(handles=[acc_plt, conf_plt], loc='upper left')

    ax.set_aspect("equal")
    ax.set_xlim(0, 1.0015)  # slightly bigger xlim to allow for confidence line to show more clearly when close to 1
    ax.set_ylim(0, 1)
    fig.savefig(fpath, bbox_inches='tight')
    print("Saved fig to: {}".format(fpath))

