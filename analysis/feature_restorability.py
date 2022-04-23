import os
import argparse
import yaml
import torch
import torch.nn as nn
from lib.data_utils import get_static_emnist_dataloaders
from nets import MNISTCNNBase, add_stats_layers_to_cnn_classifier
from lib.utils import *

"""
This is designed to measure how much the features are restored

Takes a sample from emnist source (identity) and calculate the activations of the features on pretrained net
Take the same sample from emnist target and calculate the activations of the features on pretrained net
The difference is "how far apart" the features are before adaptation

Take the same sample from emnist target and calculate the activations of the features on adapted net
The difference is how much this method restores the features

Technical note: we actually adapt on unseen samples (e.g. we adapt on EMNIST shifted test data) so a choice has to be
 made about whether the samples we use to evaluate this come from the EMNIST train or test set (we use test set)
"""

FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--data-root', type=str, default='datasets/',
                   help="path to data")
FLAGS.add_argument('--output-dir', type=str, default='./',
                   help="path to logs and ckpts")
FLAGS.add_argument('--alg-config',
                   help="path to yaml config file for algorithm settings")
FLAGS.add_argument('--data-config',
                   help="path to yaml config file for dataset settings")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', action='store_true',
                   help="DataLoader pin_memory")
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")


def restorability(shift_name, data_config, alg_config, alg_name, data_root="datasets/", ckpt_dir="ckpts/",
                  n_workers=0, pin_mem=False, dev=torch.device('cpu'), seed=123):

    adapt_classes = list(range(data_config["total_n_classes"]))
    # Standard
    # For EMNIST-DA we follow existing UDA works adapting on corrupted samples from the separate test set
    # (without labels) and then report accuracy on this same set (with labels).
    ds_path = os.path.join(data_root, "EMNIST", "identity")
    tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                          False, n_workers, pin_mem)
    id_dataloader = tst_dl

    ds_path = os.path.join(data_root, "EMNIST", shift_name)
    tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                          False, n_workers, pin_mem)
    shift_dataloader = tst_dl

    if data_config["network"] == "DigitCNN":
        id_learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
        shift_learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
    else:
        raise ValueError("Invalid network name {}".format(data_config["network"]))

    # Add stats layers to model (*before* loading weights and units stats)---
    if alg_name == "fr" or alg_name == "bufr":
        if alg_config["stats_layer"] == "all":
            stats_layers = ["gaussian", "bins", "mogs", "soft_bins"]
        elif alg_config["stats_layer"] is None:
            stats_layers = []
        else:
            stats_layers = [alg_config["stats_layer"]]

        if len(stats_layers) > 0:
            for stats_layer in stats_layers:
                add_stats_layers_to_cnn_classifier(id_learner, stats_layer, alg_config["surprise_score"],
                                                   alg_config["tau"])
                add_stats_layers_to_cnn_classifier(shift_learner, stats_layer, alg_config["surprise_score"],
                                                   alg_config["tau"])

                for learner_stats_layer in id_learner.stats_layers:
                    learner_stats_layer.calc_surprise = True
                id_learner_stats_layers = id_learner.stats_layers

                for learner_stats_layer in shift_learner.stats_layers:
                    learner_stats_layer.calc_surprise = True
                shift_learner_stats_layers = id_learner.stats_layers

        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed,
                                        alg_config["stats_layer"], alg_config["tau"])
    else:
        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed)

    # Load base learner parameters (pre-trained)--------------
    id_learner = id_learner.to(dev)
    _, id_learner = load_ckpt('pretrain-learner', id_learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)
    print("==========================")

    shift_learner = shift_learner.to(dev)
    # Hardcoded paths
    if alg_name == "adabn":
        shift_ckpt_name = "adapted-learner-0_{}_AdaBN_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "bnm":
        shift_ckpt_name = "adapted-learner-150_{}_sgd_0.01_BNM_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "bnm_im":
        shift_ckpt_name = "adapted-learner-150_{}_sgd_0.01_BNM_IM_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "jg":
        shift_ckpt_name = "adapted-learner-150_{}_sgd_0.001_JOINT_GAUSSIAN_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "im":
        shift_ckpt_name = "adapted-learner-150_{}_sgd_0.1_IM_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "pl":
        shift_ckpt_name = "adapted-learner-150_{}_sgd_0.01_PL_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "fr":
        shift_ckpt_name = "adapted-learner-150_{}_sgd_1.0_FR_fc_0.01_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "bufr":
        shift_ckpt_name = "adapted-learner-30_{}_sgd_1.0_BUFR_fc_0.01_{}.pth.tar".format(shift_name, seed)
    elif alg_name == "source":
        pass
    else:
        raise ValueError("Invalid algorithm name for restorability scores")

    if alg_name == "source":
        _, shift_learner = load_ckpt('pretrain-learner', shift_learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)
    else:
        _, shift_learner = load_ckpt('adapted-learner', shift_learner, os.path.join(ckpt_dir, shift_ckpt_name), dev)
    print("==========================")

    id_hooked_modules = []
    for module in id_learner.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            print("Identity network. Hooked module: {}.".format(module))
            id_hooked_modules.append(FwdHook(module))
    print("==========================")

    shift_hooked_modules = []
    for module in shift_learner.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            print("Adapted network. Hooked module: {}.".format(module))
            shift_hooked_modules.append(FwdHook(module))
    print("==========================")

    id_learner.eval()
    shift_learner.eval()
    with torch.no_grad():
        id_acc = 0
        shift_acc = 0
        id_acc_wrong_net = 0
        shift_acc_wrong_net = 0
        total_samples = 0
        restoration_scores = [0] * len(id_hooked_modules)
        restoration_scores_id_to_id = [0] * len(id_hooked_modules)
        restoration_scores_shift_to_shift = [0] * len(id_hooked_modules)
        for id_batch, shift_batch in zip(id_dataloader, shift_dataloader):
            id_x, id_y = id_batch
            shift_x, shift_y = shift_batch
            id_x, id_y = id_x.to(dev), id_y.to(dev)
            shift_x, shift_y = shift_x.to(dev), shift_y.to(dev)
            assert (len(id_y) == len(shift_y))

            n_samples = len(id_y)
            total_samples += n_samples

            id_predictions = id_learner(id_x)
            shift_predictions = shift_learner(shift_x)
            id_acc += accuracy(id_predictions, id_y) * n_samples
            shift_acc += accuracy(shift_predictions, shift_y) * n_samples

            # If we want values for a single unit for linear layers we just take one of the entries of
            # id_hooked_modules - shift_hooked_modules. For convolutions we would need to average over a channel.
            # As it is, we want a general picture so in both cases we can just take the mean over all samples in layer.
            for i in range(len(id_hooked_modules)):
                restoration_scores[i] += np.mean(np.abs(id_hooked_modules[i].output.detach().cpu().numpy() -
                                         shift_hooked_modules[i].output.detach().cpu().numpy())) * n_samples

            # # Extra functionality - see how features change with same data on different networks
            # id_predictions = id_learner(id_x)  # set hook correctly
            # id_predictions_wrong_net = shift_learner(id_x)
            # id_acc_wrong_net += accuracy(id_predictions_wrong_net, id_y) * n_samples
            # for i in range(len(id_hooked_modules)):
            #     restoration_scores_id_to_id[i] += np.mean(np.abs(id_hooked_modules[i].output.detach().cpu().numpy() -
            #                                       shift_hooked_modules[i].output.detach().cpu().numpy())) * n_samples
            #
            #
            # shift_predictions = shift_learner(shift_x)  # set hook correctly
            # shift_predictions_wrong_net = id_learner(shift_x)
            # shift_acc_wrong_net += accuracy(shift_predictions_wrong_net, shift_y) * n_samples
            # for i in range(len(id_hooked_modules)):
            #     restoration_scores_shift_to_shift[i] += np.mean(np.abs(id_hooked_modules[i].output.detach().cpu().numpy() -
            #                                             shift_hooked_modules[i].output.detach().cpu().numpy())) * n_samples

    print("=======================================")
    mean_restoration_scores = [score / total_samples for score in restoration_scores]
    # print("Identity data, identity network, accuracy: {:6.4f}".format(id_acc / total_samples))
    # print("Shifted data, adapted network, accuracy: {:6.4f}".format(shift_acc / total_samples))
    print("Restoration scores per layer: {}".format(mean_restoration_scores))
    # print("Mean over layers (unweighted by layer size): {}".format(np.mean(mean_restoration_scores)))
    # print("Mean over linear layers (unweighted by layer size): {}".format(np.mean(mean_restoration_scores[-2:])))
    print("=======================================")
    # mean_restoration_scores_id_to_id = [score / total_samples for score in restoration_scores_id_to_id]
    # print("Identity data, adapted network, accuracy: {:6.4f}".format(id_acc_wrong_net / total_samples))
    # print("Restoration scores per layer: {}".format(mean_restoration_scores_id_to_id))
    # print("Mean over layers (unweighted by layer size): {}".format(np.mean(mean_restoration_scores_id_to_id)))
    # print("Mean over linear layers (unweighted by layer size): {}".format(np.mean(mean_restoration_scores_id_to_id[-2:])))
    # print("=======================================")
    # mean_restoration_scores_shift_to_shift = [score / total_samples for score in restoration_scores_shift_to_shift]
    # print("Shifted data, identity network, accuracy: {:6.4f}".format(shift_acc_wrong_net / total_samples))
    # print("Restoration scores per layer: {}".format(mean_restoration_scores_shift_to_shift))
    # print("Mean over layers (unweighted by layer size): {}".format(np.mean(mean_restoration_scores_shift_to_shift)))
    # print("Mean over linear layers (unweighted by layer size): {}".format(np.mean(
    #                                                                mean_restoration_scores_shift_to_shift[-2:])))
    # print("=======================================")

    mean_restoration_scores = [score / total_samples for score in restoration_scores]
    return shift_acc / total_samples, np.mean(mean_restoration_scores), mean_restoration_scores[-2]


if __name__ == "__main__":
    #  Setup args, seed and logger -----------------------------------------------------
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {0} not recognized".format(unparsed))
    with open(args.data_config) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.alg_config) as f:
        alg_config = yaml.load(f, Loader=yaml.FullLoader)
    if args.cpu:
        dev = torch.device('cpu')
        print("USING CPU")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')
        print("USING GPU")

    # Create folders -----------------------------------------------------------------
    ckpt_dir = os.path.join(args.output_dir, "ckpts", data_config["dataset_name"])
    outputs_dir = os.path.join(args.output_dir, "outputs", data_config["dataset_name"])
    mkdir_p(ckpt_dir)
    mkdir_p(outputs_dir)

    # Quick run, one seed and only feature restoration
    # seeds = [123]
    # alg_names = ["fr"]

    seeds = [123, 234, 345, 456, 567]
    alg_names = ["adabn", "bnm", "bnm_im", "im", "jg", "pl", "fr", "bufr"]

    shift_names = data_config["shifts"]
    shift_names.sort()

    seed_accs, seed_all_restorations, seed_lin_restorations = [], [], []
    for seed in seeds:
        alg_accs, alg_all_restorations, alg_lin_restorations = [], [], []
        for alg_name in alg_names:
            shift_accs, shift_all_restorations, shift_lin_restorations = [], [], []
            for shift_name in shift_names:
                reset_rngs(seed=seed, deterministic=alg_config["deterministic"])
                shift_acc, shift_all_restoration, shift_lin_restoration = restorability(shift_name, data_config,
                                                                                        alg_config, alg_name,
                                                                                        args.data_root, ckpt_dir,
                                                                                        n_workers=args.n_workers,
                                                                                        pin_mem=args.pin_mem, dev=dev,
                                                                                        seed=seed)
                shift_accs.append(shift_acc)
                shift_all_restorations.append(shift_all_restoration)
                shift_lin_restorations.append(shift_lin_restoration)
            alg_accs.append(shift_accs)
            alg_all_restorations.append(shift_all_restorations)
            alg_lin_restorations.append(shift_lin_restorations)
        seed_accs.append(alg_accs)
        seed_all_restorations.append(alg_all_restorations)
        seed_lin_restorations.append(alg_lin_restorations)

    titles = ["Accuracies", "All layers restoration score (unweighted)", "Features only restoration score"]
    for i, results in enumerate([seed_accs, seed_all_restorations, seed_lin_restorations]):
        experiment_names, experiment_results = [], []
        for j, seed_results in enumerate(results):
            for k, alg_results in enumerate(seed_results):
                experiment_names.append("{}. Seed {}.".format(alg_names[k], seeds[j]))
                experiment_results.append(alg_results)
        print_table(experiment_results, shift_names, experiment_names, titles[i])
