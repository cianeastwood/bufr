"""
Adapt a model on a shifted (target) dataset using various different algorithms such as feature restoration, SHOT,
pseudo-labelling.
"""

from __future__ import division, print_function, absolute_import
import argparse
import yaml
import time
from nets import MNISTCNNBase, ResNet18, learner_distances, \
    add_stats_layer_to_resnet_named_modules, add_stats_layers_to_cnn_classifier, \
    add_stats_layers_to_cnn_everywhere
import nets_wilds
from lib.utils import *
from lib.stats_layers import *
from lib.data_utils import get_static_emnist_dataloaders, get_static_emnist_idx_dataloaders, \
    get_static_emnist_dataloaders_fewshot, get_static_emnist_dataloaders_oracle, \
    get_cifar10c_dataloaders, get_cifar100c_dataloaders, \
    per_hospital_wilds_dataloader, per_hospital_wilds_dataloader_fewshot
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
FLAGS.add_argument('--alg-name',
                   help="which algorithm to run, can also be set to all or fewshot")
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


def get_trainable_params(model, alg_name, network_name):
    params = []
    names = []
    if alg_name == "tent" or alg_name == "tent_online":
        for nm, m in model.named_modules():
            # Note: if using DigitCNN may wish to not include the BatchNorm1d
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    elif alg_name in ["IM", "FR", "IM_online", "FR_online", "BNM", "BNM_IM", "JOINT_GAUSSIAN"]:
        if network_name == "resnet18" or network_name == "resnet18wilds":
            for nm, m in model.named_modules():
                if not isinstance(m, nn.Linear):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
        elif network_name == "DigitCNN":
            # Hardcoded for MNISTCNNBase structure with Linear/BN1D/Linear as final layers
            freeze_layers = False
            for nm, m in model.named_modules():
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
    elif alg_name == "SHOT":  # The last batch norm is trainable in SHOT
        if network_name == "DigitCNN":
            # Hardcoded for a network structure with Linear/BN1D/Linear as final layers and no earlier BN1D
            freeze_layers = False
            for nm, m in model.named_modules():
                if isinstance(m, nn.BatchNorm1d):
                    freeze_layers = True
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
                elif isinstance(m, nn.Linear) and freeze_layers:
                    pass
                else:
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
        else:
            raise ValueError("Invalid network name for SHOT: {}".format(network_name))
        return params, names
    elif alg_name in ["label", "PL"]:
        return model.parameters(), []
    else:
        raise ValueError("Invalid algorithm name: {}".format(alg_name))


def adapt(shift_name, data_config, alg_config, data_root="datasets/", ckpt_dir="ckpts/", logs_dir="logs/",
          n_workers=0, pin_mem=False, dev=torch.device('cpu'), seed=123):

    # Error catching
    if shift_name not in data_config["shifts"]:
        raise ValueError("Invalid shift, {}, for dataset {}".format(shift_name, data_config["dataset_name"]))

    # Set up data loading---
    if data_config["dataset_name"] == 'emnist':
        ds_path = os.path.join(data_root, "EMNIST", shift_name)
        adapt_classes = list(range(data_config["total_n_classes"]))

        # Standard
        # For EMNIST-DA we follow existing UDA works adapting on corrupted samples from the separate test set
        # (without labels) and then report accuracy on this same set (with labels).
        if alg_config["alg_name"] == 'SHOT':
            # Need indexed data for pseudo labelling loss
            tr_dl, val_dl, tst_dl = get_static_emnist_idx_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                                      True, n_workers, pin_mem)
            tr_dl = tst_dl
            # Need unshuffled data for pseudo labelling method
            _, _, pl_dl = get_static_emnist_idx_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                            False, n_workers, pin_mem)

            # Need non-indexed data for ece
            _, _, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                         True, n_workers, pin_mem)
        elif alg_config["alg_name"] != 'label' and alg_config["shots_per_class"] > 0:  # few-shot experiments
            alg_config["batch_size"] = 5 * data_config["total_n_classes"]
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders_fewshot(ds_path, alg_config["shots_per_class"],
                                                                          adapt_classes, alg_config["batch_size"], True,
                                                                          n_workers, pin_mem)
            tr_dl = tst_dl
            # Also need tst_dl_full to evaluate few-shot training performance on whole dataset
            _, _, tst_dl_full = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"], True,
                                                              n_workers, pin_mem)

        elif alg_config["alg_name"] != 'label':  # default behaviour
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders(ds_path, adapt_classes, alg_config["batch_size"],
                                                                  True, n_workers, pin_mem)
            tr_dl = tst_dl

        elif alg_config["alg_name"] == 'label':  # Oracle/label
            tr_dl, val_dl, tst_dl = get_static_emnist_dataloaders_oracle(ds_path, adapt_classes,
                                                                         alg_config["batch_size"], True,
                                                                         n_workers, pin_mem)
        else:
            raise ValueError("Invalid algorithm name: {}".format(alg_name))

    elif data_config["dataset_name"] == 'cifar10':
        shuffle = False if "online" in alg_config["alg_name"] else True  # For reproducibilty for online experiments

        if alg_config["alg_name"] == "label":
            # Train split is 0.8 here as we need to get a true val_dl and tst_dl split if using labelled data
            tr_dl, val_dl, tst_dl = get_cifar10c_dataloaders(data_root, shift_name, 5, alg_config["batch_size"],
                                                             shuffle, n_workers, pin_mem, train_split=0.8,
                                                             normalize=True)
        else:
            # For cifar the process is the same as EMNIST-DA
            tr_dl, val_dl = get_cifar10c_dataloaders(data_root, shift_name, 5, alg_config["batch_size"], shuffle,
                                                     n_workers, pin_mem, train_split=1., normalize=True)
            tst_dl = tr_dl

    elif data_config["dataset_name"] == 'cifar100':
        shuffle = False if "online" in alg_config["alg_name"] else True  # For reproducibilty for online experiments

        if alg_config["alg_name"] == "label":
            # Train split is 0.8 here as we need to get a true val_dl and tst_dl split if using labelled data
            tr_dl, val_dl, tst_dl = get_cifar100c_dataloaders(data_root, shift_name, 5, alg_config["batch_size"],
                                                              shuffle, n_workers, pin_mem, train_split=0.8,
                                                              normalize=True)
        else:
            # For cifar the process is the same as EMNIST-DA
            tr_dl, val_dl = get_cifar100c_dataloaders(data_root, shift_name, 5, alg_config["batch_size"], shuffle,
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
            if alg_config["alg_name"] == 'SHOT':
                # Need indexed data for pseudo labelling loss
                tr_dl, val_dl, tst_dl = get_mnistm_idx_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                                   n_workers, pin_mem, split_seed=12345)
                # Need unshuffled data for pseudo labelling method
                pl_dl, _, _ = get_mnistm_idx_dataloaders(data_root, alg_config["batch_size"], False, False,
                                                         n_workers, pin_mem, split_seed=12345)
                # Need non-indexed data for ece
                _, _, tst_dl = get_mnistm_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                      n_workers, pin_mem, split_seed=12345)
            else:
                tr_dl, val_dl, tst_dl = get_mnistm_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                               n_workers, pin_mem, split_seed=12345)
        else:
            if alg_config["alg_name"] == 'SHOT':
                # Need indexed data for pseudo labelling loss
                tr_dl, val_dl, tst_dl = get_mnist_c_idx_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                                    shift_name, n_workers, pin_mem, split_seed=12345)
                # Need unshuffled data for pseudo labelling method
                pl_dl, _, _ = get_mnist_c_idx_dataloaders(data_root, alg_config["batch_size"], False, False,
                                                          shift_name, n_workers, pin_mem, split_seed=12345)
                # Need non-indexed data for ece
                _, _, tst_dl = get_mnist_c_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                       shift_name, n_workers, pin_mem, split_seed=12345)
            else:
                tr_dl, val_dl, tst_dl = get_mnist_c_dataloaders(data_root, alg_config["batch_size"], True, False,
                                                                shift_name, n_workers, pin_mem, split_seed=12345)

    else:
        raise NotImplementedError("Dataset {} not implemented".format(data_config["dataset_name"]))

    # Create networks---
    if data_config["network"] == "DigitCNN":
        if alg_config["alg_name"] == "SHOT":
            learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"], weight_norm=True)
        else:
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
    if alg_config["alg_name"] == "FR" or alg_config["alg_name"] == "FR_online":
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

        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed,
                                        alg_config["stats_layer"], alg_config["tau"])
    elif alg_config["alg_name"] == "SHOT":
        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed, shot=True)
    else:
        pretr_ckpt_name = get_ckpt_name(alg_config["pretr_epochs"], data_config["network"], seed)

    # Load base learner parameters (pre-trained)--------------
    learner = learner.to(dev)
    if alg_config["alg_name"] == "SHOT":
        _, learner = load_ckpt('pretrain-learner-shot', learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)
    else:
        _, learner = load_ckpt('pretrain-learner', learner, os.path.join(ckpt_dir, pretr_ckpt_name), dev)
    criterion = nn.CrossEntropyLoss()

    # Baselines that do not require SGD -----------------------------------
    if alg_config["alg_name"] == 'AdaBN':
        # Momentum=None --> calculates simple average
        for nm, m in learner.named_modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.reset_running_stats()
                m.momentum = None

        # Train mode --> track running stats
        learner.train()
        set_dropout_to_eval(learner)

        # Track batch stats. 2 epochs to be sure stats are accurate.
        for _ in range(2):
            for data_tuple in tr_dl:
                x_tr = data_tuple[0].to(dev)
                _ = learner(x_tr)

        if args.save_adapted_model:
            ckpt_path = save_ckpt(ckpt_dir, "adapted-learner", learner, None, 0, shift_name, alg_config["alg_name"],
                                  seed)
            print("Saved model ckpt to {0}".format(ckpt_path))

    if alg_config["alg_name"] in ['AdaBN', 'AdaBN_online', 'source_only']:
        if alg_config["alg_name"] == 'AdaBN_online':
            learner.train()
            set_dropout_to_eval(learner)
        else:
            learner.eval()
        test_loss = 0.
        test_acc = 0.

        if (data_config["dataset_name"] == 'emnist' or data_config["dataset_name"] == "camelyon17") and \
                alg_config["shots_per_class"] > 0:  # few-shot
            with torch.no_grad():
                for data_tuple in tst_dl_full:
                    x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    predictions = learner(x_tst)
                    loss = criterion(predictions, y_tst)
                    acc = accuracy(predictions, y_tst)

                    test_loss += loss.item()
                    test_acc += acc

            ece = expected_calibration_error(tst_dl_full, learner, dev)

            return test_acc / len(tst_dl_full), test_acc / len(tst_dl_full), ece
        else:
            with torch.no_grad():
                for data_tuple in tst_dl:
                    x_tst, y_tst = data_tuple[0].to(dev), data_tuple[1].to(dev)
                    predictions = learner(x_tst)
                    loss = criterion(predictions, y_tst)
                    acc = accuracy(predictions, y_tst)

                    test_loss += loss.item()
                    test_acc += acc

            ece = expected_calibration_error(tst_dl, learner, dev)

            return test_acc / len(tst_dl), test_acc / len(tst_dl), ece

    # Further error catching - after baselines have returned
    if alg_config["epochs"] % alg_config["val_freq"] != 0:
        raise ValueError("Total epochs must be divisible by validation frequency to get correct final epoch results")

    if "online" in alg_config["alg_name"] and alg_config["epochs"] != 1:
        raise ValueError("Online experiments should have epochs==1")

    # Experiment settings and logger ---------------------------------------------------
    if alg_config["alg_name"] == "FR" or alg_config["alg_name"] == "FR_online":
        exp_settings = [shift_name, alg_config["optimizer"], alg_config["lr"], alg_config["alg_name"],
                        alg_config["fr_type"], alg_config["tau"], seed]
        exp_setting_names = ["Shift", "Opt.", "LR", "Algorithm", "FR type", "tau", "seed"]
    else:
        exp_settings = [shift_name, alg_config["optimizer"], alg_config["lr"], alg_config["alg_name"], seed]
        exp_setting_names = ["Shift", "Opt.", "LR", "Algorithm", "seed"]

    logger = GOATLogger("train", logs_dir, alg_config["log_freq"], "adapt-single-ds", alg_config["epochs"], 0,
                        *exp_settings)
    logger.loginfo(learner)
    exp_settings_tabular = [(n, s) for n, s in zip(exp_setting_names, exp_settings)]
    exp_settings_tabular.sort(key=lambda r: r[0])
    exp_settings_table = tabulate(exp_settings_tabular, headers=["Name", "Value"], tablefmt="rst")
    logger.loginfo("Experiment settings:\n" + exp_settings_table + "\n")

    # Specify trainable params ----------------------------------------------
    trainable_params, param_names = get_trainable_params(learner, alg_config["alg_name"], data_config["network"])
    logger.loginfo("Trainable parameter names:\n" + str(param_names) + "\n")

    # Specify optimizer ----------------------------------
    if alg_config["optimizer"] == "sgd":
        optim = torch.optim.SGD(trainable_params, alg_config["lr"], momentum=alg_config["momentum"],
                                weight_decay=alg_config["weight_decay"])
    elif alg_config["optimizer"] == "adam":
        optim = torch.optim.Adam(trainable_params, alg_config["lr"])
    else:
        raise NotImplementedError("Optimizer {} not available".format(alg_config["optimizer"]))

    # Final setup to track learning curves and distances moved. -------------
    init_learner = clone_module(learner)  # clone and detach init model to compute distances moved (final - init)
    detach_module(init_learner)
    tr_accs, val_accs, val_top_k_accs = [], [], []

    # Get zero-shot accuracy and scores. Quick check for visual debugging. -------------------------------------
    tr_batch = next(iter(tr_dl))
    x_1, y_1 = tr_batch[0].to(dev), tr_batch[1].to(dev)
    learner.eval()
    with torch.no_grad():
        predictions = learner(x_1)
        zs_acc = accuracy(predictions, y_1)
    logger.loginfo("Zero-shot acc, single batch: {0:.2f}".format(zs_acc))

    if "BNM" in alg_config["alg_name"]:
        # Get batch norm statistics from penultimate layer
        pretr_bn_mean, pretr_bn_var = get_last_bn_stats(learner)
        hooked_modules = hook_linears(learner)

    if alg_config["alg_name"] == "JOINT_GAUSSIAN":
        # Get batch norm statistics from penultimate layer
        pretr_mean = load_ckpt_tensor(os.path.join(ckpt_dir, "pretrain-learner-{}_{}_{}_joint_gaussian_mean.pt".format(
                                      alg_config["pretr_epochs"], data_config["network"], seed)))
        pretr_cov = load_ckpt_tensor(os.path.join(ckpt_dir, "pretrain-learner-{}_{}_{}_joint_gaussian_cov.pt".format(
                                      alg_config["pretr_epochs"], data_config["network"], seed)))
        pretr_inv = torch.inverse(pretr_cov)
        # pretr_det = torch.det(pretr_cov)
        pretr_log_det = torch.slogdet(pretr_cov).logabsdet
        hooked_modules = hook_linears(learner)

    # Train -----------------------------------------------------------------
    epoch_times = []
    logger.loginfo("Beginning training...")
    learner.train()
    if alg_config["alg_name"] != 'label':
        set_dropout_to_eval(learner)
    for epoch in range(1, alg_config["epochs"] + 1):  # epochs
        before_epoch_t = time.time()
        # train step
        train_loss = 0.0
        train_acc = 0.0
        train_ece = 0.0

        if alg_config["alg_name"] == "SHOT" and alg_config["pl_weight"] > 0:
            if data_config["dataset_name"] in ["mnist", "emnist"]:
                mem_label = obtain_label_shot(pl_dl, learner)  # return is same size as dataset, 1 label per sample
            else:
                raise NotImplementedError("Shot not implemented for dataset {}".format(data_config["dataset_name"]))
            set_dropout_to_eval(learner)

        for batch_idx, data_tuple in enumerate(tr_dl, 1):
            x, y = data_tuple[0].to(dev), data_tuple[1].to(dev)
            if alg_config["alg_name"] == "SHOT": idx = data_tuple[2]
            optim.zero_grad()
            predictions = learner(x)
            acc = accuracy(predictions, y)

            # Calculate loss
            if alg_config["alg_name"] == "label":
                loss = criterion(predictions, y)

            elif alg_config["alg_name"] == "PL":
                _, pseudo_labels = torch.max(predictions, 1)
                loss = criterion(predictions, pseudo_labels)

            elif alg_config["alg_name"] == "BNM":  # https://arxiv.org/pdf/2101.10842.pdf, BNM only, marginal gaussians
                if data_config["network"] != "DigitCNN":
                    raise NotImplementedError("Selecting correct trainable params only implemented for EMNIST")
                # This is very hardcoded for DigitCNN, taking idx 0 only works with linear->BN->linear classifier
                bn_input_feats = hooked_modules[0].output
                batch_mean = torch.mean(bn_input_feats, dim=0)
                batch_var = torch.var(bn_input_feats, dim=0)
                loss = BNM_loss(pretr_bn_mean, pretr_bn_var, batch_mean, batch_var)

            elif alg_config["alg_name"] == "BNM_IM":  # https://arxiv.org/pdf/2101.10842.pdf, full loss
                if data_config["network"] != "DigitCNN":
                    raise NotImplementedError("Selecting correct trainable params only implemented for EMNIST")
                # This is very hardcoded for DigitCNN, taking idx 0 only works with linear->BN->linear classifier
                bn_input_feats = hooked_modules[0].output
                batch_mean = torch.mean(bn_input_feats, dim=0)
                batch_var = torch.var(bn_input_feats, dim=0)
                bnm = BNM_loss(pretr_bn_mean, pretr_bn_var, batch_mean, batch_var)
                im = IM_loss(predictions)
                loss = im + alg_config["lambda"] * bnm

            elif alg_config["alg_name"] == "JOINT_GAUSSIAN":
                if data_config["network"] != "DigitCNN":
                    raise NotImplementedError("Selecting correct trainable params only implemented for EMNIST")
                # This very hardcoded for DigitCNN, taking idx 0 only works with linear->BN->linear classifier
                bn_input_feats = hooked_modules[0].output
                batch_mean = torch.mean(bn_input_feats, dim=0)
                batch_cov = torch.mean(torch.einsum('bi,bj->bij', bn_input_feats - batch_mean,
                                                     bn_input_feats - batch_mean), dim=0) * (
                            len(bn_input_feats) / (len(bn_input_feats) - 1))  # /n-1 for unbiased

                # In this equation sigma_1 is from the batch, sigma_2 is saved on pretraining data
                # This way round means we only have to invert a covariance matrix at the start rather than every batch
                # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
                loss = 0.5 * (pretr_log_det - torch.slogdet(batch_cov).logabsdet -
                              len(batch_mean) +
                              torch.trace(torch.matmul(pretr_inv, batch_cov)) +
                              torch.matmul(torch.matmul((pretr_mean - batch_mean), pretr_inv),
                                           (pretr_mean - batch_mean))
                              )

            elif alg_config["alg_name"] == "IM" or alg_config["alg_name"] == "IM_online":
                loss = IM_loss(predictions)

            elif alg_config["alg_name"] == "SHOT":
                if alg_config["pl_weight"] > 0:
                    pred = mem_label[idx]
                    loss = criterion(predictions, pred)
                    loss *= alg_config["pl_weight"]
                else:
                    loss = 0

                im = IM_loss(predictions)
                loss = loss + im

            elif alg_config["alg_name"] == "tent" or alg_config["alg_name"] == "tent_online":
                softmax_preds = nn.Softmax(dim=1)(predictions)
                loss_ent = torch.mean(entropy(softmax_preds, dim=1))
                loss = loss_ent

            elif alg_config["alg_name"] == "FR" or alg_config["alg_name"] == "FR_online":
                scores = [sl.surprise for sl in learner_stats_layers]
                loss = get_fr_loss(scores, alg_config["fr_type"])

            else:
                raise ValueError("Invalid algorithm name {}".format(alg_config["alg_name"]))

            loss.backward()
            optim.step()

            # Accumulate loss and accuracy
            train_loss += loss.item()
            train_acc += acc
            train_ece += batch_ece(predictions, y)

        after_epoch_t = time.time()
        epoch_times.append(after_epoch_t - before_epoch_t)

        if epoch % alg_config["log_freq"] == 0:
            results = [epoch, train_loss / len(tr_dl), train_acc / len(tr_dl)]
            tr_accs.append(train_acc / len(tr_dl))
            logger.loginfo("Epoch {}. Avg tr loss {:6.4f}. Avg tr acc {:6.3f}.".format(*results))

            if alg_config["alg_name"] == "FR" or alg_config["alg_name"] == "FR_online":
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
            if alg_config["alg_name"] != 'label':
                set_dropout_to_eval(learner)

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
    if "online" in alg_config["alg_name"]:
        # We return the same acc twice for consistency with other return statement
        return train_acc / len(tr_dl), train_acc / len(tr_dl), train_ece / len(tr_dl)
    else:
        return max(val_accs), val_accs[-1], ece  # digits


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

    if data_config["dataset_name"] == 'emnist':
        possible_algs = ["adabn", "bnm", "bnm_im", "fr", "im", "jg", "pl", "shot", "label", "source_only"]
        few_shot_algs = ["bnm", "bnm_im", "fr", "im", "jg", "pl"]
    elif data_config["dataset_name"] == 'cifar10' or  data_config["dataset_name"] == 'cifar100':
        possible_algs = ["adabn_online", "im_online", "fr_online", "tent_online", "adabn", "fr", "im", "pl", "label",
                         "source_only", "tent"]
    elif data_config["dataset_name"] == 'camelyon17':
        possible_algs = ["adabn", "fr", "im", "pl", "source_only"]
        few_shot_algs = ["adabn", "fr", "im", "pl", "source_only"]
    elif  data_config["dataset_name"] == 'mnist':
        possible_algs = ["adabn", "fr", "im", "pl", "shot", "label", "source_only"]
    else:
        raise ValueError("Dataset {} not implemented".format(data_config["dataset_name"]))

    if args.alg_name == "all":
        alg_names = possible_algs
    elif args.alg_name == "few-shot" or args.alg_name == "fewshot":
        alg_names = few_shot_algs
    elif args.alg_name in possible_algs:
        alg_names = [args.alg_name]
    else:
        raise ValueError("Algorithm {} not implemented for dataset {}".format(args.alg_name,
                                                                              data_config["dataset_name"]))

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
                max_acc, final_acc, final_ece = adapt(shift_name, data_config, alg_config, args.data_root, ckpt_dir,
                                                      logs_dir, n_workers=args.n_workers,
                                                      pin_mem=args.pin_mem, dev=dev, seed=seed)

                shift_maxs.append(max_acc)
                shift_finals.append(final_acc)
                shift_eces.append(final_ece)
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
