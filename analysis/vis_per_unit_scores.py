"""
Visualize per-unit scores / KL divergences / surprise.

"""
from __future__ import division, print_function, absolute_import
import argparse
import yaml
# from nets import MNISTCNNBase, MNISTCNNBase_NoBN, add_stats_layers_to_cnn_everywhere # EMNISTCNNBase, add_stats_layers_to_net, add_batchnorm_layers_to_net
from nets import MNISTCNNBase, add_stats_layers_to_cnn_everywhere
from lib.utils import *
from lib.stats_layers import *
from lib.data_utils import get_static_emnist_dataloaders, get_static_emnist_dataloaders_fewshot
# from lib.data_transforms import CORRUPTIONS
from lib.custom_vis_nmv import NNV
# from adapt_single_target_exps import get_ckpt_name
# from icbinb import get_ckpt_name

FLAGS = argparse.ArgumentParser()

# Data parameters
# FLAGS.add_argument('--dataset_name', default="emnist", choices=['miniimagenet', 'mnist', 'emnist'],
#                    help="Name of dataset")
# FLAGS.add_argument('--shift-name', default="grass", help="Name of the shift (test distribution).",
#                    choices=["hlvl-1", "hlvl-2", "hlvl-3", "grass", "bricks", "sky", "sparks", "crystals"] + CORRUPTIONS)
# FLAGS.add_argument('--n-shot', type=int, default=10,
#                    help="Examples per class for training (k, n_support), i.e. training learner.")
# FLAGS.add_argument('--train-dist-classes', default="[0,37]",
#                    help="Classes used when (pre)training the model. 'all' or [start,end], e.g. [0,36].")
# FLAGS.add_argument('--n-high-lvl-lbls-per-task', type=int, default=5,
#                    help="Number of labels to use for each high level (label) shift.")
# FLAGS.add_argument('--which-set', type=str, default="train", choices=["train", "valid"])
#
# # Model parameters
# FLAGS.add_argument('--act-fn', choices=['relu', 'sigmoid'], default='relu')
# FLAGS.add_argument('--use-batchnorm', action="store_true")
#
# # Surprise parameters
# FLAGS.add_argument('--stats-layer', choices=['gaussian', 'bins', 'mogs', None, 'all', "moments"], default='bins')
# FLAGS.add_argument('--surprise-score', default="PSI", choices=['None', "SI", "SI_Z", "SI_norm",
#                                                                    "KL_Q_P", "KL_P_Q", "PSI", "JS"])
# FLAGS.add_argument('--n-batches', type=int, default=1,
#                    help="Validation batches. -1 = use whole validation set.")
# FLAGS.add_argument('--bn-momentum', type=float, default=0.1,
#                    help="Momentum parameter in BatchNorm2d")
# FLAGS.add_argument('--bn-eps', type=float, default=1e-5,
#                    help="Eps parameter in BatchNorm2d")
# FLAGS.add_argument('--seed', type=int, default=123,
#                    help="Random seed")
#
# # Paths
# FLAGS.add_argument('--data-root', type=str, default='datasets/',
#                    help="Location of data")
# FLAGS.add_argument('--pretrain-ckpt-name', type=str,
#                    help="Name of pretraining checkpoint, ending in pth.tar")
# FLAGS.add_argument('--output-dir', type=str, default='./',
#                    help="Location to logs and ckpts")
#
# # GPU / speed
# FLAGS.add_argument('--cpu', action='store_false',
#                    help="Set this to use CPU, default use CUDA")
# FLAGS.add_argument('--n-workers', type=int, default=0,
#                    help="How many processes for preprocessing")
# FLAGS.add_argument('--pin-mem', action='store_true',
#                    help="DataLoader pin_memory")

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


def get_plot_defaults(score):
    # [min, max, linthres, clip, logscale]
    if score == "SI":
        return -0.5, 0.5, 1e-3, True, True
    if "KL" in score:
        return 0., 0.1, 1e-3, True, False
    if score == "PSI":
        return 0., 2., 1e-3, True, False
    if score == "CMD":
        return 0., 0.2, 1e-3, True, False
    if score == "SI_norm":
        return 0., 0.05, 1e-3, True, False
    raise ValueError("Invalid score {0}".format(score))


def visualise_dists(vals, fpath="figs/conv", title=None, layer_names=None, logscale=False,
                    alpha=0.01, mn=None, mx=None, clip=False, linthres=1e-3):
    if layer_names is None:
        layer_names = ["Conv1", "Conv2", "Conv3", "FC1", "FC2"]

    layers_list = []

    for i, name in enumerate(layer_names):
        layer_vals = vals[i]
        layers_list.append({"title": name, "units": len(layer_vals), "color": layer_vals})

    # nmv = NNV(layers_list, spacing_layer=150, spacing_nodes=0.5, max_num_nodes_visible=34,
    #           node_radius=10, logscale=logscale, alpha=alpha, vmin=mn, vmax=mx, clip=clip,
    #           linthres=linthres)
    # nmv = NNV(layers_list, spacing_layer=150, spacing_nodes=10, max_num_nodes_visible=10,
    #           node_radius=40, logscale=logscale, alpha=alpha, vmin=mn, vmax=mx, clip=clip,
    #           linthres=linthres)
    # nmv.render(fig_title=title, save_to_file=fpath + '.pdf',
    #            do_not_show=True, shrink=1.0)

    nmv = NNV(layers_list, spacing_layer=75, spacing_nodes=3, max_num_nodes_visible=10,
              node_radius=10, font_size=14, logscale=logscale, alpha=alpha, vmin=mn, vmax=mx, clip=clip,
              linthres=linthres)
    nmv.render(fig_title=title, save_to_file=fpath + '.pdf',
               do_not_show=True, shrink=0.6)


def vis_unit_scores(shift_name, data_config, alg_config, data_root="datasets/", ckpt_dir="ckpts/",
                    outputs_dir="outputs/", n_workers=0, pin_mem=False, dev=torch.device('cpu'), seed=123):

    # Get data -----------------------------------------------------------------------
    if data_config["dataset_name"] == 'emnist':
        if shift_name == "H1":
            ds_path = os.path.join(data_root, "EMNIST", "identity")
            adapt_classes = [37, 38, 39, 40, 41]
        elif shift_name == "H2":
            ds_path = os.path.join(data_root, "EMNIST", "identity")
            adapt_classes = [42, 43, 44, 45, 46]
        elif shift_name == "H3":
            ds_path = os.path.join(data_root, "EMNIST", "identity")
            adapt_classes = [37, 39, 41, 43, 45]
        else:
            ds_path = os.path.join(data_root, "EMNIST", shift_name)
            # start, end = str(alg_config["pretr_classes"]).strip("[](){}").split(",")
            # adapt_classes = list(range(int(start), int(end)))  # use a subset of classes
            adapt_classes = list(range(data_config["total_n_classes"]))  # for rebuttal

        if alg_config["shots_per_class"] > 0:  # few-shot experiments
            if alg_config["shots_per_class"] < 5:
                alg_config["batch_size"] = alg_config["shots_per_class"] * len(adapt_classes)
            else:
                alg_config["batch_size"] = 5 * len(adapt_classes) #  5 * data_config["total_n_classes"]
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders_fewshot(ds_path, alg_config["shots_per_class"],
                                                                          adapt_classes, alg_config["batch_size"], True,
                                                                          n_workers, pin_mem)
            dl = tr_dl

            # tr_dl = tst_dl   # Use tr_dl to get a fully separate tr and tst set. (For low-shot is okay either way)

            # Also need tst_dl_full to evaluate few-shot training performance on whole dataset
            # _, _, tst_dl_full = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"], True,
            #                                                   n_workers, pin_mem)

        elif alg_config["shots_per_class"] <= 0:  # default behaviour
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                                  True, n_workers, pin_mem)
            dl = tr_dl
            # tr_dl = tst_dl   # Use whole tr_dl to get a separate tr and tst set

        # # High-level shifts -----------------------
        # if "hlvl" in shift_name:
        #     hlvl_shift_num = int(shift_name.split("-")[1])
        #     ds_path = os.path.join(ds_path, "identity")
        #     unused_classes = [i for i in range(total_n_classes) if i not in train_dist_classes]
        #     n_unused = len(unused_classes)
        #     n_to_choose = n_lbls_per_high_lvl_task
        #     if n_unused < n_to_choose:
        #         raise ValueError("Cannot choose {0} classes from {1} unused classes".format(n_to_choose, n_unused))
        #     if n_to_choose == -1 or n_unused == n_to_choose:
        #         print("Only a single high-level shift is possible as there is only a single way to choose {0} classes "
        #               "from {1} unused classes".format(n_to_choose, n_unused))
        #         test_dist_classes = unused_classes
        #     else:
        #         for _ in range(hlvl_shift_num):
        #             test_dist_classes = np.random.choice(unused_classes, n_to_choose, replace=False)
        #     print(shift_name)
        #     print(test_dist_classes)
        # else:
        #     ds_path = os.path.join(ds_path, shift_name)
        #     test_dist_classes = train_dist_classes

        # Create iterator dataloaders --------------
        # tr_dl, val_dl, tst_dl = get_balanced_emnist_dataloaders(ds_path, test_dist_classes, n_shot, 0,
        #                                                         False, n_workers, pin_mem)
        # if which_set == "train":
        #     dl = tr_dl
        # else:
        #     dl = val_dl
        #
        # # Calculate num batches per task for meta-validation
        # possible_n_batches = len(dl)
        # if n_batches == -1:
        #     n_batches = possible_n_batches
        # else:
        #     n_batches = min(n_batches, possible_n_batches)
    else:
        raise NotImplementedError("Only EMNIST pretraining dataloader available")

    # Create networks---
    if data_config["network"] == "DigitCNN":
        learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
        # TODO: is this weights init required?
        learner.apply(weights_init)
    elif data_config["network"] == "DigitCNN_NoBN":
        learner = MNISTCNNBase_NoBN(data_config["image_shape"], data_config["total_n_classes"])
        learner.apply(weights_init)
    else:
        raise ValueError("Invalid network name {}".format(data_config["network"]))

    # Add stats layers in here to get the values required for the plotting
    if alg_config["stats_layer"] == "all":
        stats_layers = ["gaussian", "bins", "mogs", "soft_bins"]
    elif alg_config["stats_layer"] is None:
        stats_layers = []
    else:
        stats_layers = [alg_config["stats_layer"]]

    # TODO: I think this new way of adding stats layers is broken if we add multiple types of stats layer
    # TODO: because both these functions overwrite the learner.stats_layers with an empty list
    # TODO: is the solution to add an attribute learner.stats_layers to the networks themselves that can be cleared?
    # TODO: how would this work with add stats layers for example? Best is to check if the stats_layer attr exists and if not create empty list
    # TODO: Alternatively remove the option to add many types of stats layer
    if len(stats_layers) > 0:
        for stats_layer in stats_layers:
            add_stats_layers_to_cnn_everywhere(learner, stats_layer, alg_config["surprise_score"],
                                               alg_config["tau"])
            for learner_stats_layer in learner.stats_layers:
                learner_stats_layer.calc_surprise = True
            learner_stats_layers = learner.stats_layers

    # Load base learner parameters (pre-trained)--------------
    # Used in ICBINB stuff
    # learner = learner.to(dev)
    # print(learner)
    # if alg_config["alg_name"] == 'scores':
    #     pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed,
    #                                     alg_config["stats_layer"], alg_config["tau"],
    #                                     keep_classes=alg_config["pretr_classes"])
    # else:
    #     pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed,
    #                                     keep_classes=alg_config["pretr_classes"])
    # _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)

    # Load base learner parameters (pre-trained)--------------
    # Used in rebuttal stuff
    learner = learner.to(dev)
    print(learner)

    # Before adapting
    from adapt import get_ckpt_name  # yes this is terrible, but fast for now
    pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed,
                                    alg_config["stats_layer"], alg_config["tau"])
    print(pretr_ckpt_name)
    _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)

    # During BUFR
    # adapt_ckpt_name = "adapted-learner-0_crystals_sgd_1.0_BUFR_fc_0.01_12345.pth.tar"
    # _, learner = load_ckpt('adapted-learner', learner, os.path.join(ckpt_dir+"rebuttal/", adapt_ckpt_name), dev)


    # Create model ------------------------------------------------
    # if act_fn_name == "relu":
    #     act_fn = nn.ReLU()
    # else:  # args parsing of "choices" ensures this is indeed "sigmoid"
    #     act_fn = nn.Sigmoid()
    # learner = learner(input_shape, total_n_classes, act_fn=act_fn)
    # if use_batchnorm:
    #     add_batchnorm_layers_to_net(learner, bn_eps, bn_momentum)
    # learner.apply(weights_init)
    #
    # # Add stats layers to model (*before* loading weights and units stats)---
    # for stats_layer in stats_layers:
    #     add_stats_layers_to_net(learner, stats_layer, surprise_score)
    # learner.calc_surprise = True if len(stats_layers) > 0 else False
    # learner = learner.to(dev)

    # Load base learner parameters (pre-trained), if available --------------
    # if pretr_ckpt_name is not None:
    #     pretrain_ckpt_path = os.path.join(ckpt_dir, pretr_ckpt_name)
    #     ckpt = torch.load(pretrain_ckpt_path, map_location=dev)
    #     learner.load_state_dict(ckpt['pretrain-learner'])
    #     print("Loaded pre-trained learner from {0}".format(pretrain_ckpt_path))
    # else:
    #     print("No pre-training checkpoint found. Training from scratch.")

    """
    Uncomment the below to find the surprise after updating the batch norm statistics with the new ones
    May also wish to change the filepath the image is saved to
    """
    # def set_dropout_to_eval(learner):
    #     for module in learner.modules():
    #         if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
    #             module.eval()
    #
    # # Momentum=None --> calculates simple average
    # for nm, m in learner.named_modules():
    #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
    #         m.reset_running_stats()
    #         m.momentum = None
    #
    # # Train mode --> track running stats
    # learner.train()
    # set_dropout_to_eval(learner)
    #
    # # Track batch stats. 5 epochs to be sure stats are accurate.
    # for _ in range(5):
    #     for x_tr, y_r in dl:
    #         x_tr = x_tr.to(dev)
    #         _ = learner(x_tr)



    # Get zero-shot accuracy and scores -------------------------------------
    learner.eval()
    running_scores = None

    n_batches = 0
    for i, (x, y) in enumerate(dl):  # set number of shots to e.g. 50 to not use whole dataset and get results faster
        x, y = x.to(dev), y.to(dev)
        with torch.no_grad():
            _ = learner(x)
            # scores = learner.get_surprise()
            scores = [sl.surprise for sl in learner_stats_layers]  # current unit surprise
            if running_scores is None:
                running_scores = scores
            else:
                running_scores = [run_l_ss.abs() + btch_l_ss.abs()
                                  for run_l_ss, btch_l_ss in zip(running_scores, scores)]
            n_batches +=1

    avg_scores = [l_ss.numpy() / float(n_batches) for l_ss in running_scores]
    maxs = [l_ss.max() for l_ss in avg_scores]
    mins = [l_ss.min() for l_ss in avg_scores]
    print(n_batches)
    print(maxs)
    print(mins)

    # Create score visualisations
    mn, mx, linthres, clip, logscale = get_plot_defaults(alg_config["surprise_score"])
    # fpath = os.path.join(outputs_dir, "score_vis_{0}_{1}_{2}".format(alg_config["surprise_score"], stats_layers[0],
    #                                                                  shift_name))
    fpath = os.path.join(outputs_dir + "/rebuttal/", "score_vis_{0}_{1}_{2}".format(alg_config["surprise_score"],
                                                                                    stats_layers[0], shift_name))

    print(fpath)
    visualise_dists(vals=avg_scores, fpath=fpath, title=None,     # title=shift_name, also possible
                    clip=clip, logscale=logscale, mn=mn, mx=mx, linthres=linthres)


if __name__ == '__main__':
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
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")
        dev = torch.device('cuda')

    # if args.train_dist_classes is not None:
    #     start, end = args.train_dist_classes.strip("[](){}").split(",")
    #     args.train_dist_classes = list(range(int(start), int(end)))
    # if args.stats_layer == "all":
    #     stats_layers = ["gaussian", "bins", "mogs"]
    # elif args.stats_layer is None:
    #     stats_layers = []
    # else:
    #     stats_layers = [args.stats_layer]
    # if args.pretrain_ckpt_name is None:
    #     args.pretrain_ckpt_name = get_ckpt_name(args.train_dist_classes, args.act_fn, args.use_batchnorm,
    #                                             stats_layers[0])

    # Create folders -----------------------------------------------------------------
    # ckpt_dir = os.path.join(args.output_dir, "ckpts", args.dataset_name)
    # outputs_dir = os.path.join(args.output_dir, "outputs", args.dataset_name)
    # mkdir_p(outputs_dir)

    ckpt_dir = os.path.join(args.output_dir, "ckpts", data_config["dataset_name"])
    outputs_dir = os.path.join(args.output_dir, "figs", data_config["dataset_name"], "network_surprises")
    mkdir_p(ckpt_dir)
    mkdir_p(outputs_dir)

    seed = 12345  # 123
    shift = "crystals"
    # shift = "H1"

    reset_rngs(seed=seed, deterministic=alg_config["deterministic"])
    vis_unit_scores(shift, data_config, alg_config, args.data_root, ckpt_dir,
                    outputs_dir, n_workers=args.n_workers,
                    pin_mem=args.pin_mem, dev=dev, seed=seed)

    # vis_unit_scores(args.dataset_name, args.shift_name, args.train_dist_classes, args.n_shot,
    #                 args.n_high_lvl_lbls_per_task,
    #                 args.act_fn, args.use_batchnorm, args.bn_eps, args.bn_momentum,
    #                 args.pretrain_ckpt_name, args.data_root, ckpt_dir, outputs_dir,
    #                 args.n_batches, args.which_set,
    #                 stats_layers, args.surprise_score,
    #                 args.n_workers, args.pin_mem, dev)
