import os
import torchvision
from torch.utils.data import DataLoader
import argparse
import yaml
from nets import MNISTCNNBase, add_stats_layers_to_cnn_classifier, ResNet18, add_stats_layer_to_resnet_named_modules
from lib.utils import *
from lib.stats_layers import *
from lib.surprise import surprise_bins, surprise_gaussian, surprise_mogs
from lib.data_utils import get_static_emnist_dataloaders
from lib.data_utils import get_cifar10c_dataloaders, get_cifar100c_dataloaders

FLAGS = argparse.ArgumentParser()

# Data parameters
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


class BasePlotter(object):
    def __init__(self, run_name, output_dir, data, test_dist_names, plotter_type, surprise_type=None, model_name=None):
        super(BasePlotter, self).__init__()
        self.run_name = run_name
        self.output_dir = os.path.join(output_dir, plotter_type)
        self.test_dist_names = test_dist_names
        self.plotter_type = plotter_type
        self.surprise_type = surprise_type
        self.labels = ["source", "target"]
        self.n_test_dists = len(test_dist_names)
        self.n_layers = [len(v) for k, v in data.items() if 'tr_' in k][0]  # len of each tr_item = n_layers
        # self.layer_is_conv_layer = [input_shape[-1] > 0 for input_shape in data["tr_input_shape"]]
        # self.layer_types = ["Conv" if self.layer_is_conv_layer[l] else "FC" for l in range(self.n_layers_)]
        self.layer_types = ["Layer" for _ in range(self.n_layers)]
        self.titles = [["D:{0} | L:{1}{2}".format(dist_name, self.layer_types[l], l + 1)
                        for l in range(self.n_layers)]
                       for dist_name in self.test_dist_names]
        self.fpaths = [[os.path.join(self.output_dir, "d{0}_l{1}_{2}.jpg".format(dist_name, l + 1, self.run_name))
                        for l in range(self.n_layers)]
                       for dist_name in self.test_dist_names]
        if model_name is not None:
            self.fpaths = [[l_fpath.strip(".jpg") + "_{0}.jpg".format(model_name)
                            for l_fpath in d_fpaths]
                           for d_fpaths in self.fpaths]
        mkdir_p(self.output_dir)

    def calc_surprise(self):
        if self.surprise_type is None:
            return None
        print("Calculating unit-level surprise using {0}...".format(self.plotter_type))
        surprises = [[[]] * self.n_layers] * self.n_test_dists  # empty lists to hold surprises for var-sized layers
        for d, dist_name in enumerate(self.test_dist_names):
            print("\t-{0}".format(dist_name))
            for l in range(self.n_layers):
                surprises[d][l] = self.calc_layer_surprises(d, l)
        print("Unit-level surprise calculated.")
        return surprises

    def plot(self):
        print("Plotting {0}...".format(self.plotter_type))
        if len(self.test_dist_names) == 0:
            for l in range(self.n_layers):
                self.plot_layer_dists(0, l)
        else:
            for d, dist_name in enumerate(self.test_dist_names):
                print("\t-{0}".format(dist_name))
                for l in range(self.n_layers):
                    self.plot_layer_dists(d, l)
        print("Finished plotting.")

    def plot_layer_dists(self, d, l):
        raise NotImplementedError

    def calc_layer_surprises(self, d, l):
        raise NotImplementedError


class BinsPlotter(BasePlotter):
    def __init__(self, run_name, output_dir, data, test_dist_names, surprise_type=None):
        super(BinsPlotter, self).__init__(run_name, output_dir, data, test_dist_names, "bins", surprise_type)
        self.edges = data["tr_edges"]
        self.train_counts = data["tr_counts"]
        if "tst_counts" in data:
            self.test_counts = data["tst_counts"]
        else:
            self.test_counts = None
        self.surprises = self.calc_surprise()

    def calc_layer_surprises(self, d, l):
        if self.test_counts is None:
            return None
        return surprise_bins(self.train_counts[l], self.test_counts[d][l], self.surprise_type, fast=False)

    def plot_layer_dists(self, d, l):
        layer_surprises = None
        all_dist_counts = [self.train_counts[l]]

        if self.test_counts is not None:
            all_dist_counts.append(self.test_counts[d][l])
            if self.surprises is not None:
                layer_surprises = self.surprises[d][l]

        if l == 4:  # 2 for cifar, 4 for emnist archs
            ax_title = "a"
        else:
            ax_title = "z"
        plot_bins(self.edges[l][:6], all_dist_counts[:6], self.titles[d][l], self.labels, self.fpaths[d][l],
                  layer_surprises[:6], self.surprise_type, ax_title=ax_title)


class SoftBinsPlotter(BinsPlotter):
    def __init__(self, run_name, output_dir, data, test_dist_names, surprise_type=None, model_name=None):
        super(BinsPlotter, self).__init__(run_name, output_dir, data, test_dist_names, "soft_bins", surprise_type,
                                          model_name)
        self.edges = data["tr_soft_edges"]
        self.train_counts = data["tr_soft_counts"]
        if "tst_soft_counts" in data:
            self.test_counts = data["tst_soft_counts"]
        else:
            self.test_counts = None
        self.tau = data["tr_tau"][0][0]
        # self.fpaths = [[l_fpath.strip(".jpg") + "_tau={0}.pdf".format(self.tau)
        #                 for l_fpath in d_fpaths]
        #                for d_fpaths in self.fpaths]

        self.surprises = self.calc_surprise()

    def calc_layer_surprises(self, d, l):
        if self.test_counts is None:
            return None
        return surprise_soft_bins(torch.from_numpy(self.train_counts[l]),
                                  torch.from_numpy(self.test_counts[d][l]), self.surprise_type)


class GaussianPlotter(BasePlotter):
    def __init__(self, run_name, output_dir, data, test_dist_names, surprise_type=None):
        super(GaussianPlotter, self).__init__(run_name, output_dir, data, test_dist_names, "gaussian", surprise_type)
        self.train_means = data["tr_mean"]
        self.train_vars = data["tr_var"]
        self.test_means = data["tst_mean"]
        self.test_vars = data["tst_var"]
        self.surprises = self.calc_surprise()

    def calc_layer_surprises(self, d, l):
        return surprise_gaussian(torch.from_numpy(self.train_means[l]), torch.from_numpy(self.train_vars[l]),
                                 torch.from_numpy(self.test_means[d][l]), torch.from_numpy(self.test_vars[d][l]),
                                 score_type=self.surprise_type)

    def plot_layer_dists(self, d, l):
        means = [self.train_means[l], self.test_means[d][l]]
        vars = [self.train_vars[l], self.test_vars[d][l]]
        layer_surprises = self.surprises[d][l] if self.surprises is not None else None
        plot_gaussians(means, vars, self.titles[d][l], self.labels, self.fpaths[d][l],
                       layer_surprises, self.surprise_type)


class MomentsPlotter(BasePlotter):
    def __init__(self, run_name, output_dir, data, test_dist_names, surprise_type=None):
        super(MomentsPlotter, self).__init__(run_name, output_dir, data, test_dist_names, "gaussian", surprise_type)
        self.train_moments = data["tr_moments"]
        self.test_moments = data["tst_moments"]
        self.mins = data["tr_m_mins"]
        self.maxs = data["tr_m_maxs"]
        self.surprises = self.calc_surprise()

    def calc_layer_surprises(self, d, l):
        tr_ms = [torch.from_numpy(m) for m in self.train_moments[l]]
        tst_ms = [torch.from_numpy(m) for m in self.train_moments[d][l]]
        return surprise_moments(tr_ms, tst_ms, self.mins, self.maxs)

    def plot_layer_dists(self, d, l):
        means = [self.train_moments[l][0], self.test_moments[d][l][0]]
        vars = [self.train_moments[l][1], self.test_moments[d][l][1]]
        layer_surprises = self.surprises[d][l] if self.surprises is not None else None
        plot_gaussians(means, vars, self.titles[d][l], self.labels, self.fpaths[d][l],
                       layer_surprises, self.surprise_type)


class MoGsPlotter(BasePlotter):
    def __init__(self, run_name, output_dir, data, test_dist_names, surprise_type=None):
        super(MoGsPlotter, self).__init__(run_name, output_dir, data, test_dist_names, "mogs", surprise_type)
        self.centres = data["tr_centres"]
        self.stds = data["tr_stds"]
        self.hs = data["tr_hs"]
        self.train_weights = data["tr_weights"]
        self.test_weights = data["tst_weights"]
        self.surprises = self.calc_surprise()

    def calc_layer_surprises(self, d, l):
        return surprise_mogs(torch.from_numpy(self.train_weights[l]), torch.from_numpy(self.test_weights[d][l]),
                             score_type=self.surprise_type, fast=False)

    def plot_layer_dists(self, d, l):
        all_weights = [self.train_weights[l], self.test_weights[d][l]]
        layer_surprises = self.surprises[d][l] if self.surprises is not None else None
        plot_mogs(self.centres[l], self.stds[l], all_weights, self.titles[d][l], self.labels, self.fpaths[d][l],
                  layer_surprises, self.surprise_type)


class SamplesPlotter(BasePlotter):
    def __init__(self, run_name, output_dir, data, test_dist_names, surprise_type=None):
        super(SamplesPlotter, self).__init__(run_name, output_dir, test_dist_names, "samples", surprise_type)
        self.train_samples = data["tr_samples"]
        self.test_samples = data["tst_samples"]
        self.chosen_indices = data["tr_indices"]

    def calc_layer_surprises(self, d, l):
        raise NotImplementedError

    def plot_layer_dists(self, d, l):
        all_samples = [self.train_samples[l], self.test_samples[d][l]]
        plot_samples(all_samples, self.chosen_indices[l], self.titles[d][l], self.labels, self.fpaths[d][l])


def get_plotter(stats_layer):
    if stats_layer == "bins":
        return BinsPlotter
    elif stats_layer == "gaussian":
        return GaussianPlotter
    elif stats_layer == "mogs":
        return MoGsPlotter
    elif stats_layer == "moments":
        return MomentsPlotter
    elif stats_layer == "soft_bins":
        return SoftBinsPlotter
    elif stats_layer == "samples":
        return SamplesPlotter
    else:
        raise ValueError("Invalid stats layer type {0}".format(stats_layer))


def get_module_stats(module_list):
    stats = {}
    for m in module_list:
        m_dict = m.get_stats()
        for k, v in m_dict.items():
            if k not in stats:
                stats[k] = []
            stats[k].append(v.copy())
    return stats


def main():
    #  Setup args, seed and logger ------------------------------------------
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

    # Create folders --------------------------------------------------------
    ckpt_dir = os.path.join(args.output_dir, "ckpts", data_config["dataset_name"])
    figs_dir = os.path.join(args.output_dir, "figs", data_config["dataset_name"], "unit_activations")
    mkdir_p(ckpt_dir)
    mkdir_p(figs_dir)

    seed = 123
    shift_name = "stripe"
    exp_name = "{0}_{1}".format("appendix", alg_config["stats_layer"])
    reset_rngs(seed=seed, deterministic=alg_config["deterministic"])
    assert shift_name in data_config["shifts"]

    # Get datasets ---------------------------------------------------
    if data_config["dataset_name"] == 'emnist':
        # Properties ------------------------------
        model = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
        adapt_classes = list(range(data_config["total_n_classes"]))
        ds_path = os.path.join(args.data_root, "EMNIST", shift_name)
        _, _, dataloader = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"], False,
                                                         args.n_workers, args.pin_mem)

    elif "cifar" in data_config["dataset_name"]:
        # Setup CIFAR-10/CIFAR-100 parameters
        if data_config["dataset_name"] == 'cifar10':
            get_dataloaders_fn = get_cifar10c_dataloaders
        elif data_config["dataset_name"] == 'cifar100':
            get_dataloaders_fn = get_cifar100c_dataloaders
        else:
            raise ValueError("Invalid dataset name {0}".format(data_config["dataset_name"]))

        # Setup CIFAR model
        model = ResNet18(n_classes=data_config["total_n_classes"])
        modules_to_track = ['layer4', 'linear']
        module_features_out = [512, data_config["total_n_classes"]]
        module_features_in = [None, 512]

        # Setup CIFAR dataloaders
        dataloader, _ = get_dataloaders_fn(args.data_root, shift_name, 5, alg_config["batch_size"], False,
                                           args.n_workers, args.pin_mem, train_split=1., normalize=True)

    else:
        raise NotImplementedError("Dataset {0} not implemented.".format(data_config["dataset_name"]))

    # Add stats layers
    if alg_config["stats_layer"] == "all":
        stats_layers = ["gaussian", "bins", "mogs", "soft_bins"]
    elif alg_config["stats_layer"] is None:
        stats_layers = []
    else:
        stats_layers = [alg_config["stats_layer"]]

    if len(stats_layers) > 0:
        for stats_layer in stats_layers:
            if "resnet" in data_config["network"]:
                add_stats_layer_to_resnet_named_modules(model, modules_to_track, module_features_out,
                                                        module_features_in, stats_layer_type=stats_layer,
                                                        surprise_score=alg_config["surprise_score"],
                                                        tau=alg_config["tau"])
                for model_stats_layer in model.stats_layers:
                    model_stats_layer.calc_surprise = True
                model_stats_layers = model.stats_layers
            else:
                add_stats_layers_to_cnn_classifier(model, stats_layer, alg_config["surprise_score"],
                                                   alg_config["tau"])
                for model_stats_layer in model.stats_layers:
                    model_stats_layer.calc_surprise = True
                model_stats_layers = model.stats_layers

    # Load model ------------------------------------------------------------#
    model = model.to(dev)
    shift_ckpt_name = "adapted-learner-30_{}_sgd_1.0_BUFR_fc_0.01_{}.pth.tar".format(shift_name, seed)
    model_ckpt_path = os.path.join(ckpt_dir, shift_ckpt_name)

    if os.path.exists(model_ckpt_path):
        if "pretrain" in shift_ckpt_name:
            _, model = load_ckpt('pretrain-learner', model, model_ckpt_path, dev)
        else:
            _, model = load_ckpt('adapted-learner', model, model_ckpt_path, dev)
    else:
        raise ValueError("No checkpoint found at {0}!".format(model_ckpt_path))

    # Pull out the saved train (p) stats ------------------------------------
    train_stats = get_module_stats(model_stats_layers)

    # Accumulate the test (q) stats -----------------------------------------
    print("Accumulating unit statistics on the test data (i.e. forming \'q\')...")
    all_test_stats = {}
    test_dist_names = [shift_name]
    model.eval()
    for dist_name in test_dist_names:
        # Reset stats
        for m in model_stats_layers:
            m.reset_stats()
            m.track_stats = True
        print("\t-{0}".format(dist_name))
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                x_tr, y_tr = batch
                x_tr = x_tr.to(dev)
                _ = model(x_tr)
        dist_test_stats = get_module_stats(model_stats_layers)
        for k in dist_test_stats:  # Accumulate per-test-dist stats in "outer" dict
            if k not in all_test_stats:
                all_test_stats[k] = []
            all_test_stats[k].append(dist_test_stats[k].copy())
    print("Unit statistics accumulated (q formed).")

    # all_test_stats = {}
    # test_dist_names = []

    # Combine train and test data -------------------------------------------
    train_stats = add_prefix_to_dict_keys(train_stats, "tr_")
    all_test_stats = add_prefix_to_dict_keys(all_test_stats, "tst_")
    all_stats = {**train_stats, **all_test_stats}

    # Plot ------------------------------------------------------------------
    for stats_layer in stats_layers:
        plotter = get_plotter(stats_layer)
        plotter = plotter(exp_name, figs_dir, all_stats, test_dist_names, alg_config["surprise_score"])
        plotter.plot()


if __name__ == '__main__':
    main()
