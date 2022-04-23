"""
Adapt a model on a shifted (target) dataset using bottom-up feature restoration.
"""

from __future__ import division, print_function, absolute_import
import argparse
import yaml
import time
from nets import MNISTCNNBase, ResNet18, learner_distances, add_stats_layer_to_resnet_named_modules, \
    add_stats_layers_to_cnn_classifier, add_stats_layers_to_cnn_everywhere
import nets_wilds
from lib.utils import *
from lib.stats_layers import *
from lib.data_utils import get_static_emnist_dataloaders, get_static_emnist_dataloaders_fewshot, \
    get_cifar10c_dataloaders, get_cifar100c_dataloaders, per_hospital_wilds_dataloader, \
    per_hospital_wilds_dataloader_fewshot
from data.digits import *

FLAGS = argparse.ArgumentParser()

FLAGS.add_argument('--data-root', type=str, default='datasets/',
                   help="path to data")
FLAGS.add_argument('--output-dir', type=str, default='./',
                   help="path to logs and ckpts")
FLAGS.add_argument('--alg-configs-dir',
                   help="path to directory containing yaml config files for algorithm settings")
FLAGS.add_argument('--data-config',
                   help="path to yaml config file for dataset settings")
FLAGS.add_argument('--seed', type=int,
                   help="random seed, should match a pretraining seed")
FLAGS.add_argument('--save-adapted-model', action='store_true',
                   help="Set this to save the network after adaptation")
FLAGS.add_argument('--deterministic', action='store_true',
                   help="Set this to make everything deterministic")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', action='store_true',
                   help="DataLoader pin_memory")
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")


def get_trainable_params(module_list, network_name):
    params = []
    names = []
    if network_name == "resnet18" or network_name == "resnet18wilds":
        for nm, m in module_list.named_modules():
            if not isinstance(m, nn.Linear):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
    elif network_name == "DigitCNN":
        # Hardcoded for MNISTCNNBase structure with Linear/BN1D/Linear as final layers
        freeze_layers = False
        for nm, m in module_list.named_modules():
            if isinstance(m, nn.BatchNorm1d):
                freeze_layers = True
            elif isinstance(m, nn.Linear) and freeze_layers:
                pass
            else:
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
    else:
        raise ValueError("Invalid network name: {}".format(network_name))

    return params, names


def get_trainable_params_list(model, network_name):
    if network_name == "DigitCNN":
        modules_list = []
        for m in model.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear) or
              isinstance(m, nn.BatchNorm1d)):
                modules_list.append(m)

        # Bottom up goes 2 layers at a time ([conv1, bn1], [conv2, bn2], ..., [linear, bn])
        # Note the zip drops the last linear layer which we do not want to train anyway
        modules_list = list(zip(modules_list[::2], modules_list[1::2]))
        modules_list = [nn.ModuleList(ms) for ms in modules_list]
    elif network_name == "resnet18":
        # Bottom up goes 1 block at a time ([conv1, bn1], block1, block2, ..., linear)
        modules_list = list(model.children())
        conv1, bn1 = modules_list[:2]
        modules_list = [nn.ModuleList([conv1, bn1])] + modules_list[2:-1]
    elif network_name == "resnet18wilds":
        # Bottom up goes 1 block at a time ([conv1, bn1], block1, block2, ..., linear)
        modules_list = list(model.children())
        conv1, bn1 = modules_list[:2]
        # Cuts out relu, max pool 2d and adaptive avg pool
        modules_list = [nn.ModuleList([conv1, bn1])] + modules_list[4:-2]
    else:
        raise ValueError("Invalid network choice {0}".format(network_name))
    return modules_list


def get_scores_layer_from_param_layer(l, using_batchnorm=False):
    if not using_batchnorm:
        return l
    if l == 0:
        return l
    return math.ceil(l / 2)


def adapt_bu(shift_name, data_config, alg_config, data_root="datasets/", ckpt_dir="ckpts/", logs_dir="logs/",
             n_workers=0, pin_mem=False, dev=torch.device('cpu'), seed=123):

    if shift_name not in data_config["shifts"]:
        raise ValueError("Invalid shift, {}, for dataset {}".format(shift_name, data_config["dataset_name"]))
    
    if alg_config["epochs_per_block"] % alg_config["val_freq"] != 0:
        raise ValueError("Epochs per block must be divisible by validation frequency")

    # Set up data loading---
    if data_config["dataset_name"] == 'emnist':
        ds_path = os.path.join(data_root, "EMNIST", shift_name)
        adapt_classes = list(range(data_config["total_n_classes"]))

        # Standard
        # For EMNIST-DA we follow existing UDA works adapting on corrupted samples from the separate test set
        # (without labels) and then report accuracy on this same set (with labels).
        if alg_config["shots_per_class"] > 0:  # few-shot experiments
            alg_config["batch_size"] = 5 * data_config["total_n_classes"]
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders_fewshot(ds_path, alg_config["shots_per_class"],
                                                                          adapt_classes, alg_config["batch_size"],
                                                                          True,
                                                                          n_workers, pin_mem)
            tr_dl = tst_dl
            # Also need tst_dl_full to evaluate few-shot training performance on whole dataset
            _, _, tst_dl_full = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                              True,
                                                              n_workers, pin_mem)

        else:  # default behaviour
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                                  True, n_workers, pin_mem)
            tr_dl = tst_dl

    elif data_config["dataset_name"] == 'cifar10':
        # For cifar the process is the same as EMNIST-DA
        tr_dl, val_dl = get_cifar10c_dataloaders(data_root, shift_name, 5, alg_config["batch_size"], True,
                                                 n_workers, pin_mem, train_split=1., normalize=True)
        tst_dl = tr_dl

    elif data_config["dataset_name"] == 'cifar100':
        # For cifar the process is the same as EMNIST-DA
        tr_dl, val_dl = get_cifar100c_dataloaders(data_root, shift_name, 5, alg_config["batch_size"], True,
                                                  n_workers, pin_mem, train_split=1., normalize=True)
        tst_dl = tr_dl

    elif data_config["dataset_name"] == "camelyon17":
        if alg_config["shots_per_class"] > 0:  # few-shot experiments
            alg_config["batch_size"] = min(alg_config["shots_per_class"] * data_config["total_n_classes"],
                                           alg_config["batch_size"])
            tr_dl = per_hospital_wilds_dataloader_fewshot(data_root, shift_name, alg_config["shots_per_class"],
                                                           alg_config["batch_size"], n_workers, pin_mem)
            tst_dl = tr_dl
            # Also need tst_dl_full to evaluate few-shot training performance on whole dataset
            tst_dl_full = per_hospital_wilds_dataloader(data_root, shift_name, 200, n_workers, pin_mem)
        else:
            tr_dl = per_hospital_wilds_dataloader(data_root, shift_name, alg_config["batch_size"], n_workers, pin_mem)
            tst_dl = tr_dl

    elif data_config["dataset_name"] == "mnist":
        # For mnist we follow previous works by adapting on corrupted samples from same training set used for
        # pretraining and evaluating on the mnist test set that has been corrupted
        if shift_name == "mnistm":
            tr_dl, val_dl, tst_dl = get_mnistm_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                           n_workers, pin_mem, split_seed=12345)
        else:
            tr_dl, val_dl, tst_dl = get_mnist_c_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                            shift_name, n_workers, pin_mem, split_seed=12345)

    else:
        raise NotImplementedError("Dataset {} not implemented".format(data_config["dataset_name"]))

    # Create networks---
    if data_config["network"] == "DigitCNN":
        learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
    elif data_config["network"] == "resnet18":
        learner = ResNet18(n_classes=data_config["total_n_classes"])
        modules_to_track = ['linear']
        module_features_out = [data_config["total_n_classes"]]
        module_features_in = [512]
    elif data_config["network"] == "resnet18wilds":
        learner = nets_wilds.ResNet18(num_classes=data_config["total_n_classes"])
        modules_to_track = ['fc']
        module_features_out = [data_config["total_n_classes"]]
        module_features_in = [512]
    else:
        raise ValueError("Invalid network name {}".format(data_config["network"]))

    # Add stats layers to model (*before* loading weights and units stats)---
    if alg_config["stats_layer"] == "all":
        stats_layers = ["gaussian", "bins", "mogs", "soft_bins"]
    elif alg_config["stats_layer"] is None:
        stats_layers = []
    else:
        stats_layers = [alg_config["stats_layer"]]

    if len(stats_layers) > 0:
        for stats_layer in stats_layers:
            if "resnet" in data_config["network"]:
                add_stats_layer_to_resnet_named_modules(learner, modules_to_track, module_features_out,
                                                        module_features_in, stats_layer_type=stats_layer,
                                                        surprise_score=alg_config["surprise_score"],
                                                        tau=alg_config["tau"])
                for learner_stats_layer in learner.stats_layers:
                    learner_stats_layer.calc_surprise = True
                learner_stats_layers = learner.stats_layers
            else:
                add_stats_layers_to_cnn_classifier(learner, stats_layer, alg_config["surprise_score"],
                                                   alg_config["tau"])
                # add_stats_layers_to_cnn_everywhere(learner, stats_layer, alg_config["surprise_score"],
                #                                    alg_config["tau"])
                for learner_stats_layer in learner.stats_layers:
                    learner_stats_layer.calc_surprise = True
                learner_stats_layers = learner.stats_layers

    # Load base learner parameters (pre-trained)--------------
    learner = learner.to(dev)
    pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed,
                                    alg_config["stats_layer"], alg_config["tau"])
    _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)

    criterion = nn.CrossEntropyLoss()

    # Experiment settings and logger ---------------------------------------------------
    exp_settings = [shift_name, alg_config["optimizer"], alg_config["lr"], alg_config["alg_name"],
                    alg_config["fr_type"], alg_config["tau"], seed]
    exp_setting_names = ["Shift", "Opt.", "LR", "Algorithm", "FR type", "tau", "seed"]
    logger = GOATLogger("train", logs_dir, alg_config["log_freq"], "adapt-single-ds", alg_config["epochs_per_block"], 0,
                        *exp_settings)
    logger.loginfo(learner)
    exp_settings_tabular = [(n, s) for n, s in zip(exp_setting_names, exp_settings)]
    exp_settings_tabular.sort(key=lambda r: r[0])
    exp_settings_table = tabulate(exp_settings_tabular, headers=["Name", "Value"], tablefmt="rst")
    logger.loginfo("Experiment settings:\n" + exp_settings_table + "\n")

    # Specify trainable params ----------------------------------------------
    trainable_modules_list = get_trainable_params_list(learner, data_config["network"])

    # Final setup to track learning curves and distances moved. -------------
    init_learner = clone_module(learner)  # clone and detach init model to compute distances moved (final - init)
    detach_module(init_learner)
    tr_accs, val_accs, val_top_k_accs = [], [], []

    # Train -----------------------------------------------------------------
    epoch_times = []
    logger.loginfo("Beginning training...")
    learner.train()
    set_dropout_to_eval(learner)
    trainable_params = []
    for m_idx, trainable_module in enumerate(trainable_modules_list):
        logger.loginfo("Block {0}".format(m_idx))
        if data_config["network"] == "DigitCNN":
            trainable_module_params, param_names = get_trainable_params(trainable_module, data_config["network"])
            lr_ = alg_config["lr"] / (1.5 ** m_idx)  # decay based on the block being trained
        elif data_config["network"] == "resnet18" or data_config["network"] == "resnet18wilds":
            trainable_module_params, param_names = get_trainable_params(trainable_module, data_config["network"])
            lr_ = alg_config["lr"] / (5 ** m_idx)  # decay based on the block being trained
        else:
            raise ValueError("Invalid network name: {}".format(data_config["network"]))

        # Add trainable module params to running list ("unfreeze" strategy) -
        trainable_params.extend(trainable_module_params)
        logger.loginfo("Block LR {0}".format(lr_))

        if alg_config["optimizer"] == "sgd":
            optim = torch.optim.SGD(trainable_params, lr_, momentum=alg_config["momentum"],
                                    weight_decay=alg_config["weight_decay"])
        elif alg_config["optimizer"] == "adam":
            optim = torch.optim.Adam(trainable_params, lr_)
        else:
            raise NotImplementedError("Optimizer {} not available".format(alg_config["optimizer"]))

        for epoch in range(1, alg_config["epochs_per_block"] + 1):  # epochs
            before_epoch_t = time.time()
            train_loss = 0.0
            train_acc = 0.0
            train_ece = 0.0
            for batch_idx, data_tuple in enumerate(tr_dl, 1):
                x, y = data_tuple[0].to(dev), data_tuple[1].to(dev)
                optim.zero_grad()
                predictions = learner(x)
                acc = accuracy(predictions, y)
                scores = [sl.surprise for sl in learner_stats_layers]
                loss = get_fr_loss(scores, alg_config["fr_type"])
                loss.backward()
                optim.step()
                train_loss += loss.item()
                train_acc += acc
                train_ece += batch_ece(predictions, y)

            after_epoch_t = time.time()
            epoch_times.append(after_epoch_t - before_epoch_t)

            if epoch % alg_config["log_freq"] == 0:
                results = [epoch, train_loss / len(tr_dl), train_acc / len(tr_dl)]
                tr_accs.append(train_acc / len(tr_dl))
                logger.loginfo("Epoch {}. Avg tr loss {:6.4f}. Avg tr acc {:6.3f}.".format(*results))

                if len(stats_layers) > 0:
                    scores = [sl.surprise for sl in learner_stats_layers]
                    log_scores(scores, None, logger, None, None)

            if epoch % alg_config["val_freq"] == 0:
                learner.eval()
                with torch.no_grad():
                    valid_loss = 0.0
                    valid_acc = 0.0
                    n_val_samples = 0
                    for data_tuple in tst_dl:
                        x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                        predictions = learner(x_val)
                        loss = criterion(predictions, y_val)  # This is cross entropy and not e.g. FR loss
                        acc = accuracy(predictions, y_val)

                        # Weighted sum (final batch may be smaller with drop_last=False)
                        n_samples = len(y_val)
                        n_val_samples += n_samples
                        valid_loss += loss.item() * n_samples
                        valid_acc += acc * n_samples

                val_accs.append(valid_acc / n_val_samples)
                logger.loginfo("Validation loss {:6.4f}".format(valid_loss / n_val_samples))
                logger.loginfo("Validation accuracy {:6.3f}".format(valid_acc / n_val_samples))

                d_moved_per_layer = learner_distances(init_learner, learner, distance_type="all", is_tracked_net=False)
                mean_d, max_d, frac_moved = list(zip(*d_moved_per_layer))
                # logger.loginfo("Avg dist. moved:\n{0}".format(["{0:.2f}".format(d) for d in mean_d]))
                logger.loginfo("Max dist. moved:\n{0}".format(["{0:.3f}".format(d) for d in max_d]))
                # logger.loginfo("Frac. of features who moved:\n{0}".format(["{0:.2f}".format(d) for d in frac_moved]))

                learner.train()
                set_dropout_to_eval(learner)

        # For visualising how the scores change during training
        # ckpt_path = save_ckpt(ckpt_dir + "rebuttal/", "adapted-learner", learner, optim, m_idx,
        #                       *exp_settings)
        # logger.loginfo("Saved model ckpt to {0}".format(ckpt_path))

    logger.loginfo("Finished Training.")
    logger.loginfo("###Timing###")
    logger.loginfo(epoch_times)
    logger.loginfo("Avg time per epochh {}".format(np.mean(epoch_times)))
    logger.loginfo("############")

    ece = expected_calibration_error(tst_dl, learner, dev)

    # Few-shot datasets only - get final few-shot performance on whole dataset. Will print in the finals table.
    if data_config["dataset_name"] == 'emnist' or data_config["dataset_name"] == "camelyon17":
        if alg_config["shots_per_class"] > 0:
            learner.eval()
            with torch.no_grad():
                valid_loss = 0.0
                valid_acc = 0.0
                n_val_samples = 0
                for data_tuple in tst_dl_full:
                    x_val, y_val = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    predictions = learner(x_val)
                    loss = criterion(predictions, y_val)
                    acc = accuracy(predictions, y_val)

                    # Weighted sum (final batch may be smaller with drop_last=False)
                    n_samples = len(y_val)
                    n_val_samples += n_samples
                    valid_loss += loss.item() * n_samples
                    valid_acc += acc * n_samples

            val_accs.append(valid_acc / n_val_samples)
            logger.loginfo("Full dataset validation loss {:6.4f}".format(valid_loss / n_val_samples))
            logger.loginfo("Full dataset validation accuracy {:6.3f}".format(valid_acc / n_val_samples))

            ece = expected_calibration_error(tst_dl_full, learner, dev)

    # Save model ----------------------------------------------
    if args.save_adapted_model:
        # Save model ---------------------------------------------
        ckpt_path = save_ckpt(ckpt_dir, "adapted-learner", learner, optim, epoch, *exp_settings)
        logger.loginfo("Saved model ckpt to {0}".format(ckpt_path))

        # Save results ---------------------------------------------
        exp_affix = "_".join(str(arg) for arg in exp_settings)
        output_file_path = os.path.join("./", "outputs", data_config["dataset_name"], "analysis_data_{0}".format(exp_affix))
        np.savez(output_file_path,
                 tr_accs=np.array(tr_accs), val_accs=np.array(val_accs),
                 mean_ds=np.array(mean_d), max_ds=np.array(max_d))

    logger.shutdown()

    return max(val_accs), val_accs[-1], ece


if __name__ == '__main__':
    #  Setup args, seed and logger -----------------------------------------------------
    args, unparsed = FLAGS.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {0} not recognized".format(unparsed))
    with open(args.data_config) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
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
    logs_dir = os.path.join(args.output_dir, "logs", data_config["dataset_name"])
    mkdir_p(ckpt_dir)
    mkdir_p(logs_dir)
    mkdir_p(outputs_dir)

    # Set up seeds and algorithms to run experiments over ----------------------------
    seeds = [args.seed]  # Can put multiple seeds in here if desired
    alg_names = ["bufr"]
    shift_names = data_config["shifts"]
    shift_names.sort()

    # Experiments --------------------------------------------------------------------
    seed_maxs, seed_finals, seed_eces = [], [], []
    for seed in seeds:
        alg_name_maxs, alg_name_finals, alg_name_eces = [], [], []
        for alg_name in alg_names:
            with open(args.alg_configs_dir + alg_name + ".yml") as f:
                alg_config = yaml.load(f, Loader=yaml.FullLoader)
            shift_maxs, shift_finals, shift_eces = [], [], []
            for shift_name in shift_names:
                reset_rngs(seed=seed, deterministic=args.deterministic)
                max_acc, final_acc, ece = adapt_bu(shift_name, data_config, alg_config, args.data_root,
                                                   ckpt_dir, logs_dir, n_workers=args.n_workers,
                                                   pin_mem=args.pin_mem, dev=dev, seed=seed)
                shift_maxs.append(max_acc)
                shift_finals.append(final_acc)
                shift_eces.append(ece)
            alg_name_maxs.append(shift_maxs)
            alg_name_finals.append(shift_finals)
            alg_name_eces.append(shift_eces)
        seed_maxs.append(alg_name_maxs)
        seed_finals.append(alg_name_finals)
        seed_eces.append(alg_name_eces)

    # Save results
    fname = os.path.join(outputs_dir, "results_all.npz")
    np.savez(fname, maxes=seed_maxs, finals=seed_finals, eces=seed_eces)

    # Load results and print tables
    fname = os.path.join(outputs_dir, "results_all.npz")
    data = np.load(fname)

    for k in data:
        experiment_names, experiment_results = [], []

        for i, seed_results in enumerate(data[k]):
            for j, alg_name_results in enumerate(seed_results):
                experiment_names.append("{}. Seed {}.".format(alg_names[j], seeds[i]))
                experiment_results.append(alg_name_results)

        print_table(experiment_results, shift_names, experiment_names, k)
