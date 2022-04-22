"""
Pre-train a model on the training (source) distribution.
"""

from __future__ import division, print_function, absolute_import
import argparse
import yaml
from nets import MNISTCNNBase, ResNet18
import nets_wilds
import torch.optim
from lib.utils import *
from lib.stats_layers import *
from lib.data_utils import get_dynamic_emnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, \
    per_hospital_wilds_dataloader_split
from data.digits import *


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
                   help="random seed")
FLAGS.add_argument('--test-accuracy', action='store_true',
                   help="Set this to print test set accuracy")
FLAGS.add_argument('--deterministic', action='store_true',
                   help="Set this to make everything deterministic")
FLAGS.add_argument('--n-workers', type=int, default=4,
                   help="How many processes for preprocessing")
FLAGS.add_argument('--pin-mem', action='store_true',
                   help="DataLoader pin_memory")
FLAGS.add_argument('--cpu', action='store_true',
                   help="Set this to use CPU, default use CUDA")


def main():
    #  Setup args, seed and logger -----------------------------------------------------
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

    # Create folders -----------------------------------------------------------------
    ckpt_dir = os.path.join(args.output_dir, "ckpts", data_config["dataset_name"])
    mkdir_p(ckpt_dir)

    # Experiment flags for name (affixes) -----------------------------------

    exp_affixes = [data_config["network"], args.seed]

    # Get data -----------------------------------------------------------------------
    if data_config["dataset_name"] == 'emnist':
        # Use DynamicEMNIST for pre-training (loads entire dataset into memory at once, slightly faster)
        train_classes = list(range(data_config["total_n_classes"]))  # use all classes
        tr_loader, val_loader, tst_loader = get_dynamic_emnist_dataloaders(args.data_root, train_classes,
                                                                           alg_config["batch_size"], True,
                                                                           args.n_workers, args.pin_mem)
    elif data_config["dataset_name"] == 'cifar10':
        tr_loader, val_loader, tst_loader = get_cifar10_dataloaders(args.data_root, alg_config["batch_size"], True,
                                                                    args.n_workers, args.pin_mem, train_split=1.)
    elif data_config["dataset_name"] == 'cifar100':
        tr_loader, val_loader, tst_loader = get_cifar100_dataloaders(args.data_root, alg_config["batch_size"], True,
                                                                     args.n_workers, args.pin_mem, train_split=1.)
    elif data_config["dataset_name"] == 'camelyon17':
        tr_loader, val_loader, tst_loader = per_hospital_wilds_dataloader_split(args.data_root,
                                                                                alg_config["hospital_idx"],
                                                                                alg_config["batch_size"],
                                                                                args.n_workers, args.pin_mem)
    elif data_config["dataset_name"] == "mnist":
        tr_loader, val_loader, tst_loader = get_mnist_dataloaders(args.data_root, alg_config["batch_size"], True, True,
                                                                  args.n_workers, args.pin_mem, split_seed=12345)
    else:
        raise NotImplementedError("Dataset {} not implemented".format(data_config["dataset_name"]))

    if data_config["network"] == "DigitCNN":  # MNIST, EMNIST
        if alg_config["shot_pretrain"]:
            learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"], weight_norm=True)
        else:
            learner = MNISTCNNBase(data_config["image_shape"], data_config["total_n_classes"])
    elif data_config["network"] == "resnet18":  # Cifar
        learner = ResNet18(n_classes=data_config["total_n_classes"])
    elif data_config["network"] == "resnet18wilds":  # Camelyon
        learner = nets_wilds.ResNet18(num_classes=data_config["total_n_classes"])
    else:
        raise ValueError("Invalid network name {}".format(data_config["network"]))

    learner = learner.to(args.dev)

    # Specify optimizer and loss ----------------------------------
    optim = torch.optim.SGD(learner.parameters(), alg_config["lr"], momentum=alg_config["momentum"],
                            weight_decay=alg_config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=alg_config["epochs"])
    print(learner)

    if data_config["network"] == "DigitCNN" and alg_config["shot_pretrain"]:
        criterion = CrossEntropyLabelSmooth(data_config["total_n_classes"])  # for SHOT baseline
    else:
        criterion = nn.CrossEntropyLoss()

    # Pre-train ---------------------------------------------------
    print("Beginning pre-training...")
    learner.train()
    for epoch in range(1, alg_config["epochs"] + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for data_tuple in tr_loader:
            x_tr, y_tr = data_tuple[0].to(args.dev), data_tuple[1].to(args.dev)
            # train step
            optim.zero_grad()
            output = learner(x_tr)
            loss = criterion(output, y_tr)
            acc = accuracy(output, y_tr)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            epoch_acc += acc

        results = [epoch, epoch_loss / len(tr_loader), epoch_acc / len(tr_loader)]
        print("Epoch {}. Avg train loss {:6.4f}. Avg train acc {:6.3f}.".format(*results))

        # Although for EMNIST we implement a true train/val split, for cifar we wish to train on the whole training set
        # (that is, there is no validation set).
        #
        # If setting hyperparameters, change the train_split argument in cifar dataloaders to get a true split
        # using a validation set
        if epoch % alg_config["val_freq"] == 0:
            learner.eval()
            valid_loss = 0.0
            valid_acc = 0.0
            with torch.no_grad():
                for data_tuple in val_loader:
                    x_val, y_val = data_tuple[0].to(args.dev), data_tuple[1].to(args.dev)
                    output = learner(x_val)
                    loss = criterion(output, y_val)
                    acc = accuracy(output, y_val)
                    valid_loss += loss.item()
                    valid_acc += acc
            print("Validation loss {:6.4f}".format(valid_loss / len(val_loader)))
            print("Validation accuracy {:6.3f}".format(valid_acc / len(val_loader)))
            learner.train()
        scheduler.step()
    print("Finished Training.")

    # Save model ----------------------------------------------
    if alg_config["shot_pretrain"]:
        if data_config["dataset_name"] not in ["mnist", "emnist"]:
            raise ValueError("Shot pretraining not available for dataset name {}".format(data_config["dataset_name"]))
        else:
            ckpt_path = save_ckpt(ckpt_dir, "pretrain-learner-shot", learner, optim, epoch, *exp_affixes)
    else:
        ckpt_path = save_ckpt(ckpt_dir, "pretrain-learner", learner, optim, epoch, *exp_affixes)
    print("Saved model ckpt to {0}".format(ckpt_path))

    # Print pre-training test set accuracy --------------------
    if args.test_accuracy:
        print("Testing pre-trained network on test set.")
        learner.eval()
        learner.track_stats = False
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for data_tuple in tst_loader:
                x_te, y_te = data_tuple[0].to(args.dev), data_tuple[1].to(args.dev)

                output = learner(x_te)
                loss = criterion(output, y_te)
                acc = accuracy(output, y_te)

                test_loss += loss.item()
                test_acc += acc
        print("Test loss {:6.4f}".format(test_loss / len(tst_loader)))
        print("Test accuracy {:6.3f}".format(test_acc / len(tst_loader)))
        print("====================")

    print("Test ECE: {:6.4f}".format(expected_calibration_error(tst_loader, learner, args.dev)))


if __name__ == '__main__':
    main()
