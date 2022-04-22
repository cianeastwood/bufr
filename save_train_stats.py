"""
Save source statistics (e.g. soft bin parameters) after pre-training.
"""

import os
import argparse
import yaml
from torch.utils.data import DataLoader
from nets import add_stats_layers_to_cnn_classifier, add_stats_layers_to_cnn_everywhere, MNISTCNNBase, \
    add_stats_layer_to_resnet_named_modules, ResNet18
import nets_wilds
from lib.utils import *
from lib.stats_layers import *
from lib.data_utils import get_cifar10_dataloaders, get_cifar100_dataloaders, per_hospital_wilds_dataloader
from data.digits import *
from data.emnist import DynamicEMNIST

FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--data-root', type=str, default='datasets/',
                   help="path to data")
FLAGS.add_argument('--output-dir', type=str, default='./',
                   help="path to logs and ckpts")
FLAGS.add_argument('--alg-config',
                   help="path to yaml config file for algorithm settings")
FLAGS.add_argument('--data-config',
                   help="path to yaml config file for dataset settings")
FLAGS.add_argument('--seed', type=int,
                   help="random seed, should match a pretraining seed")
FLAGS.add_argument('--deterministic', action='store_true',
                   help="Set this to make everything deterministic")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', action='store_true',
                   help="DataLoader pin_memory")
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")


def main():
    #  Setup args, seed and logger ------------------------------------------
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {0} not recognized".format(unparsed))
    with open(args.alg_config) as f:
        alg_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.data_config) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    reset_rngs(seed=args.seed, deterministic=args.deterministic)

    if args.cpu:
        args.dev = torch.device('cpu')
        print("USING CPU")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        args.dev = torch.device('cuda')
        print("USING GPU")

    if alg_config["stats_layer"] == "all":
        stats_layers = ["gaussian", "bins", "mogs", "soft_bins"]
    elif alg_config["stats_layer"] is None:
        stats_layers = []
    else:
        stats_layers = [alg_config["stats_layer"]]

    print(alg_config["stats_layer"])

    # Create folders --------------------------------------------------------
    ckpt_dir = os.path.join(args.output_dir, "ckpts", data_config["dataset_name"])
    mkdir_p(ckpt_dir)

    # Experiment flags for name (affixes) -----------------------------------
    # exp_affixes = [alg_config["stats_layer"]]

    # Get data --------------------------------------------------------------
    if data_config["dataset_name"] == 'emnist':
        keep_classes = list(range(data_config["total_n_classes"]))
        ds = DynamicEMNIST(root=args.data_root, norm=True, color=True, which_set="train",
                           keep_classes=keep_classes)
        dataloader = DataLoader(ds, batch_size=alg_config["batch_size"], shuffle=False,
                                num_workers=args.n_workers, pin_memory=args.pin_mem, drop_last=True)

    elif data_config["dataset_name"] == 'cifar10':
        dataloader, _, _ = get_cifar10_dataloaders(args.data_root, alg_config["batch_size"], True,
                                                   args.n_workers, args.pin_mem, train_split=1.,
                                                   normalize=True)
    elif data_config["dataset_name"] == 'cifar100':
        dataloader, _, _ = get_cifar100_dataloaders(args.data_root, alg_config["batch_size"], True,
                                                    args.n_workers, args.pin_mem, train_split=1.,
                                                    normalize=True)
    elif data_config["dataset_name"] == 'camelyon17':
        dataloader = per_hospital_wilds_dataloader(args.data_root, alg_config["hospital_idx"], alg_config["batch_size"],
                                                   args.n_workers, args.pin_mem)
    elif data_config["dataset_name"] == "mnist":
        dataloader, _, _ = get_mnist_dataloaders(args.data_root, alg_config["batch_size"], shuffle=False, augment=False,
                                                 n_workers=args.n_workers, pin_mem=args.pin_mem, split_seed=12345)
    else:
        raise NotImplementedError("Dataset {} not implemented".format(data_config["dataset_name"]))

    if data_config["network"] == "DigitCNN":
        learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], args.seed)
        pretrain_epoch, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name),
                                            args.dev)
    elif data_config["network"] == "resnet18":
        learner = ResNet18(n_classes=data_config["total_n_classes"])
        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], args.seed)
        pretrain_epoch, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name),
                                            args.dev)
        modules_to_track = ['linear']
        module_features_out = [data_config["total_n_classes"]]
        module_features_in = [512]
    elif data_config["network"] == "resnet18wilds":
        learner = nets_wilds.ResNet18(num_classes=data_config["total_n_classes"])
        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], args.seed)
        pretrain_epoch, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name),
                                            args.dev)
        modules_to_track = ['fc']
        module_features_out = [data_config["total_n_classes"]]
        module_features_in = [512]
    else:
        raise ValueError("Invalid network name {}".format(data_config["network"]))

    print(learner)
    # Calculate num batches to use ------------
    possible_n_batches = len(dataloader)
    if alg_config["n_batches"] == -1:
        n_batches = possible_n_batches
    else:
        n_batches = min(alg_config["n_batches"], possible_n_batches)

    # Add stats layers (*after* loading base model weights) -----------------
    for stats_layer in stats_layers:
        if ("resnet" in data_config["network"]) or ("dense" in data_config["network"]):
            add_stats_layer_to_resnet_named_modules(learner, modules_to_track, module_features_out, module_features_in,
                                                    stats_layer_type=stats_layer, tau=alg_config["tau"])
        else:
            add_stats_layers_to_cnn_classifier(learner, stats_layer, tau=alg_config["tau"])
            # For adding stats layers everywhere, e.g. to make the max patches figure, use the below
            # add_stats_layers_to_cnn_everywhere(learner, stats_layer, tau=alg_config["tau"])

    learner_stats_layers = learner.stats_layers
    learner = learner.to(args.dev)
    print(learner)
    print(learner.stats_layers)

    # Calibrate bin range (iff using a BinStats layer) ----------------------
    if any([isinstance(stats_layer, BinStats) or isinstance(stats_layer, MomentStats)
            for stats_layer in learner_stats_layers]):
        print("Calibrating bin ranges...")
        for stats_layer in learner_stats_layers:
            stats_layer.track_range = True
        learner.eval()
        with torch.no_grad():
            for i, data_tuple in enumerate(dataloader):
                if i >= n_batches:
                    break
                x_tr = data_tuple[0].to(args.dev)
                _ = learner(x_tr)
        print("Bin ranges calibrated.")

    # Gather train (p) stats ------------------------------------------------
    print("Gathering unit statistics on the training data (i.e. forming \'p\')...")
    for stats_layer in learner_stats_layers:
        stats_layer.track_range = False
        stats_layer.track_stats = True

    learner.eval()
    with torch.no_grad():
        for i, data_tuple in enumerate(dataloader):
            if i >= n_batches:
                break
            x_tr = data_tuple[0].to(args.dev)
            _ = learner(x_tr)
    print("Unit statistics gathered (p formed).")

    for stats_layer in learner_stats_layers:
        # print(stats_layer.maxs - stats_layer.mins)
        # print(stats_layer.m2)
        print(stats_layer.feature_ranges[:, 0])
        print(stats_layer.bin_edges[:, -2] - stats_layer.bin_edges[:, 1])

    # Save model ------------------------------------------------------------
    save_stats_affixes = [pretr_ckpt_name, stats_layers[0], alg_config["tau"]]
    # For adding stats layers everywhere, e.g. to make the max patches figure, use the below
    # save_stats_affixes = [pretr_ckpt_name, stats_layers[0], alg_config["tau"], "all"]
    ckpt_path = save_ckpt(ckpt_dir, "pretrain-learner", learner, None, pretrain_epoch, *save_stats_affixes)
    if "camelyon" in data_config["dataset_name"]:
        print("Used hospital {}".format(alg_config["hospital_idx"]))
    print("Saved model ckpt to {0}".format(ckpt_path))

    # Joint Gaussian mean and covariance ------------------------------------------------------------
    if data_config["network"] == "DigitCNN":
        print("Calculating Full Gauss. stats")
        hooked_modules = hook_linears(learner)
        learner.eval()
        with torch.no_grad():
            first_batch = True
            for i, data_tuple in enumerate(dataloader):
                if i >= n_batches:
                    break
                x_tr = data_tuple[0].to(args.dev)
                _ = learner(x_tr)

                first_linear_output_feats = hooked_modules[0].output
                if first_batch:
                    n_samples = len(first_linear_output_feats)
                    cum_sum = torch.sum(first_linear_output_feats, dim=0)
                    first_batch = False
                else:
                    n_samples += len(first_linear_output_feats)
                    cum_sum += torch.sum(first_linear_output_feats, dim=0)

            mean_feats = cum_sum / n_samples
            print("Mean calculated.")
            first_batch = True

            for i, data_tuple in enumerate(dataloader):
                if i >= n_batches:
                    break
                x_tr = data_tuple[0].to(args.dev)
                _ = learner(x_tr)

                first_linear_output_feats = hooked_modules[0].output
                this_bs = len(first_linear_output_feats)

                if first_batch:
                    cov_sum = torch.mean(torch.einsum('bi,bj->bij', first_linear_output_feats - mean_feats,
                                                      first_linear_output_feats - mean_feats), dim=0) * this_bs
                    first_batch = False
                else:
                    cov_sum += torch.mean(torch.einsum('bi,bj->bij', first_linear_output_feats - mean_feats,
                                                       first_linear_output_feats - mean_feats), dim=0) * this_bs

            cov = cov_sum / (n_samples - 1)
            print("Covariance matrix calculated.")
            print(cov)  # symmetric?

        # Save mean and covariance ------------------------------------------------------------
        tensor_path = save_ckpt_tensor(ckpt_dir, "pretrain-learner", mean_feats, pretrain_epoch,
                                       *[pretr_ckpt_name, "joint_gaussian_mean"])
        print("Saved mean to {0}".format(tensor_path))
        tensor_path = save_ckpt_tensor(ckpt_dir, "pretrain-learner", cov, pretrain_epoch,
                                       *[pretr_ckpt_name, "joint_gaussian_cov"])
        print("Saved covariance to {0}".format(tensor_path))


if __name__ == '__main__':
    main()
