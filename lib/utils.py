from __future__ import division, print_function, absolute_import
from scipy.spatial.distance import cdist
import matplotlib
# must select appropriate backend before importing any matplotlib functions
# matplotlib.use("TkAgg")

import math
import os
import logging
import errno
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import tarfile
from matplotlib.image import imread
from matplotlib.lines import Line2D
import scipy.stats
import time
from tabulate import tabulate
from collections import OrderedDict

plt.interactive(False)

EPS = 1e-10

###################################################################
# Logging Functions ###############################################
###################################################################
class GOATLogger:

    def __init__(self, mode, save_root, log_freq, base_name, n_iterations, n_eval_iterations, *argv):
        self.mode = mode
        self.save_root = save_root
        self.log_freq = log_freq
        self.n_iterations = n_iterations
        self.n_eval_iterations = n_eval_iterations

        if self.mode == 'train':
            mkdir_p(self.save_root)
            affix = "_".join(str(a) for a in argv)
            filename = os.path.join(self.save_root, '{0}_{1}.log'.format(base_name, affix))
            # https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            logging.basicConfig(level=logging.INFO,  # DEBUG causes: https://github.com/camptocamp/pytest-odoo/issues/15
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S',
                                filename=filename,
                                filemode='w')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            self.log = logging.getLogger('')
            self.log.addHandler(console)

            logging.info("Logger created at {}".format(filename))
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S')

        self.stats = {}
        self.reset_stats()
        self.time = time.time()

    def reset_stats(self):
        if self.mode == 'train':
            self.stats = {'train': {'loss': [], 'acc': []},
                          'eval': {'loss': [], 'acc': []},
                          'eval-per-task': {}}
        else:
            self.stats = {'eval': {'loss': [], 'acc': []},
                          'eval-per-task': {}}

    def batch_info(self, **kwargs):
        if kwargs['phase'] == 'train':
            self.stats['train']['loss'].append(kwargs['loss'])
            self.stats['train']['acc'].append(kwargs['acc'])

            if kwargs['iteration'] % self.log_freq == 0:
                # mean on the last n samples, where n=self.log_freq
                batch_loss = np.mean(self.stats['train']['loss'][-self.log_freq:])
                batch_acc = np.mean(self.stats['train']['acc'][-self.log_freq:])

                self.loginfo("[{0}/{1}] Loss:{2:.3f}, Acc:{3:.2f}".format(kwargs['iteration'], self.n_iterations,
                                                                          batch_loss, batch_acc))
                self.loginfo("Time for {0} iteration(s): {1}s".format(self.log_freq, int(time.time() - self.time)))
                self.time = time.time()

        elif kwargs['phase'] == 'eval':
            self.stats['eval']['loss'].append(kwargs['loss'])
            self.stats['eval']['acc'].append(kwargs['acc'])

        elif kwargs['phase'] == 'eval-done':
            loss_mean = np.mean(self.stats['eval']['loss'])
            loss_std = np.std(self.stats['eval']['loss'])
            acc_mean = np.mean(self.stats['eval']['acc'])
            acc_std = np.std(self.stats['eval']['acc'])

            self.loginfo("[{:5d}] Eval ({:3d} episode) - "
                         "Loss: {:6.4f} +- {:6.4f},"
                         "Acc: {:6.3f} +- {:5.3f}%. "
                         "Low CI {:6.3f}, "
                         "High CI {:6.3f}.".format(kwargs['iteration'], self.n_eval_iterations, loss_mean, loss_std,
                                                   acc_mean, acc_std,
                                                   acc_mean - 1.96 * (acc_std/np.sqrt(self.n_eval_iterations)),
                                                   acc_mean + 1.96 * (acc_std/np.sqrt(self.n_eval_iterations))))
            self.reset_stats()
            return acc_mean

        elif kwargs['phase'] == 'eval-per-task':
            attrs = ['loss', 'acc', 'per_step_accs', 'layer_distances', 'c_thlds', 'p_thlds', 'c_ss', 'p_ss']
            if kwargs['task_name'] in self.stats['eval-per-task']:      # append to list
                for attr in attrs:
                    append_to_dict_entry(self.stats, kwargs, attr)
            else:                                                       # create list
                self.stats['eval-per-task'][kwargs['task_name']] = {}
                for attr in attrs:
                    add_new_dict_entry(self.stats, kwargs, attr)

        elif kwargs['phase'] == 'eval-per-task-done':
            #  Gather info -- lists used over arrays to allow different batches sizes for different tasks (slower)
            task_names = list(self.stats['eval-per-task'].keys())
            task_losses = np.array([self.stats['eval-per-task'][task_name]['loss'] for task_name in task_names])
            task_accs = np.array([self.stats['eval-per-task'][task_name]['acc'] for task_name in task_names])

            #  Calc mean loss/acc *per task* -- could print within-task std over batches, but not so insightful
            mean_loss_per_task = task_losses.mean(1)
            mean_acc_per_task = task_accs.mean(1)

            #  Log per-task results
            tabular_results = [(t_name, "{0:.3f}".format(t_loss), "{0:.2f}".format(t_acc))
                               for t_name, t_loss, t_acc in zip(task_names, mean_loss_per_task, mean_acc_per_task)]
            tabular_results.sort(key=lambda r: r[0])             # sort by task name
            results_table = tabulate(tabular_results, headers=["Task", "Loss", "Accuracy"], tablefmt="rst")
            self.loginfo("[{0}/{1}] Meta-validation".format(kwargs['iteration'], self.n_iterations) + "\n"
                         "Per-task results (query samples):\n" + results_table + "\n")

            #  Log/print per-step, per-task results (if available)
            if 'per_step_accs' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'per_step_accs', task_names,
                                                  'Per-step accuracies [s_1, s_1, ...]')
                self.loginfo("Per-task, per-step results (avg. across support samples):\n" + res_table + "\n")

            #  Log/print layer distances, if available
            if 'layer_distances' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'layer_distances', task_names,
                                                  'Per-layer dist. moved [l_1, l_2, ...]')
                self.loginfo("Distances moved by each layer after n inner steps:\n" + res_table + "\n")

            #  Log/print current thresholds, if available
            if 'c_thlds' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'c_thlds', task_names,
                                                  'Current Thresholds [Conv, Linear] / [l_1, l_2, ...]')
                self.loginfo("Thresholds for current surprises:\n" + res_table + "\n")

            #  Log/print parent thresholds, if available
            if 'p_thlds' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results(self.stats, 'p_thlds', task_names,
                                                  'Parent Thresholds [Conv, Linear] / [l_1, l_2, ...]')
                self.loginfo("Thresholds for parent surprises:\n" + res_table + "\n")

            #  Log/print current surprises, if available
            if 'c_ss' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results_per_layer(self.stats, 'c_ss', task_names,
                                                            'Current Surprises [s_1, s_2, ...]')
                self.loginfo("Current surprises (avg. across support samples):\n" + res_table + "\n")

            #  Log/print current surprises, if available
            if 'p_ss' in self.stats['eval-per-task'][task_names[0]]:
                res_table = tabulate_mean_results_per_layer(self.stats, 'p_ss', task_names,
                                                            'Parent Surprises [s_1, s_2, ...]')
                self.loginfo("Parent surprises (avg. across support samples):\n" + res_table + "\n")

            #  Calc average and std *across tasks*
            loss_mean = mean_loss_per_task.mean()
            loss_std = mean_loss_per_task.std()
            acc_mean = mean_acc_per_task.mean()
            acc_std = mean_acc_per_task.std()

            #  Log avg results
            self.loginfo("Avg task loss: {0:.3f} +- {1:.3f}, "
                         "Avg task acc: {2:.2f} +- {3:.2f}.\n".format(loss_mean, loss_std, acc_mean, acc_std))

            self.reset_stats()
            return acc_mean

        else:
            raise ValueError("phase {} not supported".format(kwargs['phase']))

    def logdebug(self, strout):
        logging.debug(strout)

    def loginfo(self, strout):
        logging.info(strout)

    def shutdown(self):
        handlers = self.log.handlers[:]
        for handler in handlers:
            handler.close()
            self.log.removeHandler(handler)


def log_scores(c_surprises, p_surprises, logger, per_l_c_thlds, per_l_p_thlds):
    logger.loginfo("Scores:")
    c_s_means = [l_s.abs().mean() for l_s in c_surprises]
    c_s_maxs = [l_s.abs().max() for l_s in c_surprises]
    c_s_medians = [l_s.abs().median() for l_s in c_surprises]
    logger.loginfo("C. Avg. Surprise {0}".format(["{0:.2g}".format(l_s_m) for l_s_m in c_s_means]))
    logger.loginfo("C. Max. Surprise {0}".format(["{0:.2g}".format(l_s_m) for l_s_m in c_s_maxs]))
    logger.loginfo("C. Median. Surprise {0}".format(["{0:.2g}".format(l_s_m) for l_s_m in c_s_medians]))

    if per_l_c_thlds is not None:
        c_s_frac = [torch.sum(torch.where(s >= thld, 1., 0.)) / s.size()[0]
                    for s, thld in zip(c_surprises, per_l_c_thlds)]
        logger.loginfo("C. Frac. above thld {0}".format(["{0:.2f}".format(l_s_m) for l_s_m in c_s_frac]))

    if p_surprises is None:
        return

    p_s_means = [l_s.abs().mean() for l_s in p_surprises]
    p_s_maxs = [l_s.abs().max() for l_s in p_surprises]
    p_s_medians = [l_s.abs().median() for l_s in p_surprises]
    p_s_frac = [torch.sum(torch.where(s >= thld, 1., 0.)) / s.size()[0]
                for s, thld in zip(p_surprises, per_l_p_thlds)]

    logger.loginfo("P. Avg. Surprise {0}".format(["{0:.2f}".format(l_s_m) for l_s_m in p_s_means]))
    logger.loginfo("P. Max. Surprise {0}".format(["{0:.2f}".format(l_s_m) for l_s_m in p_s_maxs]))
    logger.loginfo("P. Median. Surprise {0}".format(["{0:.2f}".format(l_s_m) for l_s_m in p_s_medians]))
    logger.loginfo("P. Frac. above thld {0}".format(["{0:.2f}".format(l_s_m) for l_s_m in p_s_frac]))


def append_to_dict_entry(d_running, d_input, attr_name):
    if attr_name in d_input and d_input[attr_name] is not None:
        d_running['eval-per-task'][d_input['task_name']][attr_name].append(d_input[attr_name])


def add_new_dict_entry(d_running, d_input, attr_name):
    if attr_name in d_input and d_input[attr_name] is not None:
        d_running['eval-per-task'][d_input['task_name']][attr_name] = [d_input[attr_name]]


def tabulate_mean_results(d_running, attr_name, task_names, attr_name_pretty):
    per_step_accs = [d_running['eval-per-task'][task_name][attr_name] for task_name in task_names]
    mean_acc_per_task_per_step = np.mean(per_step_accs, 1)  # mean over n_batches
    tabular_results = [(t_name, "{0}".format(["{0:.2f}".format(t_a) for t_a in t_accs]))
                       for t_name, t_accs in zip(task_names, mean_acc_per_task_per_step)]
    tabular_results.sort(key=lambda r: r[0])  # sort by task name
    results_table = tabulate(tabular_results, headers=["Task", attr_name_pretty], tablefmt="rst")
    return results_table


def tabulate_mean_results_per_layer(d_running, attr_name, task_names, attr_name_pretty):
    per_step_values = [d_running['eval-per-task'][task_name][attr_name] for task_name in task_names]
    mean_value_per_task_per_step = np.mean(per_step_values, 1).transpose((0, 2, 1))     # [n_tasks, n_layers, n_steps]
    tabular_results = [[(t_name, l, "{0}".format(["{0:.2f}".format(l_v) for l_v in l_values]))
                       for l, l_values in enumerate(t_values)]
                       for t_name, t_values in zip(task_names, mean_value_per_task_per_step)]
    tabular_results = flatten(tabular_results)
    tabular_results.sort(key=lambda r: r[0])  # sort by task name
    results_table = tabulate(tabular_results, headers=["Task", "Layer", attr_name_pretty], tablefmt="rst")
    return results_table


###################################################################
# Evaluation functions ############################################
###################################################################
def topk_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0].item() if len(res) == 1 else [r.item() for r in res]


def accuracy(outputs, targets):
    _, predictions = torch.max(outputs, 1)
    return 100 * torch.sum(torch.squeeze(predictions).float() == targets).item() / float(targets.size(0))


def binary_accuracy_with_sigmoid(outputs, targets):
    outputs = torch.sigmoid(outputs)
    predictions = torch.where(outputs >= 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
    return 100 * torch.sum(torch.squeeze(predictions).float() == targets).item() / float(targets.size(0))


def batch_ece(outputs, targets, num_bins=10):
    bin_indices = np.linspace(0, 1, num_bins + 1)
    bin_confidences_sum = np.zeros(num_bins)
    bin_accuracies_sum = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    with torch.no_grad():
        outputs = torch.nn.Softmax(dim=1)(outputs)
        confidences, predictions = torch.max(outputs, 1)
    confidences = confidences.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    confidence_bins = np.digitize(confidences, bin_indices, right=True)
    if np.any(confidence_bins == 0) or np.any(confidence_bins > num_bins):
        raise RuntimeError("It should be impossible for the softmax outputs to be <=0 or >1")
    # Every sample now has a bin index between 1 and num_bins (inclusive)

    for i in range(len(predictions)):  # Probably can be vectorised but this is clearer for now
        bin_confidences_sum[confidence_bins[i] - 1] += confidences[i]
        bin_counts[confidence_bins[i] - 1] += 1
        bin_accuracies_sum[confidence_bins[i] - 1] += predictions[i] == targets[i]

    # print("After ECE: avg test accuracy = {:6.3f}%".format(100 * np.sum(bin_accuracies_sum) / np.sum(bin_counts)))

    total_count = np.sum(bin_counts)
    bin_counts[np.where(bin_counts == 0)] += 1  # as accuracy or conf sum is zero anyway this just stops a nan error
    per_bin_accuracies = bin_accuracies_sum / bin_counts
    per_bin_confidences = bin_confidences_sum / bin_counts

    # Eqn 3 in https://arxiv.org/pdf/1706.04599.pdf
    # Multiply by 100 to get a percent
    ece = 100 * np.sum((bin_counts / total_count) * np.abs(per_bin_accuracies - per_bin_confidences))

    return ece


def expected_calibration_error(dataloader, learner, dev, num_bins=10):
    bin_indices = np.linspace(0, 1, num_bins + 1)
    bin_confidences_sum = np.zeros(num_bins)
    bin_accuracies_sum = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    learner.eval()
    with torch.no_grad():
        # for x_src, y_src in dataloader:
        #     x_src, y_src = x_src.to(dev), y_src.to(dev)

        for data_tuple in dataloader:
            x_src, y_src = data_tuple[0].to(dev), data_tuple[1].to(dev)

            outputs = learner(x_src)
            outputs = torch.nn.Softmax(dim=1)(outputs)
            confidences, predictions = torch.max(outputs, 1)
            confidences = confidences.detach().cpu().numpy()
            predictions = predictions.detach().cpu().numpy()
            targets = y_src.detach().cpu().numpy()
            confidence_bins = np.digitize(confidences, bin_indices, right=True)
            if np.any(confidence_bins == 0) or np.any(confidence_bins > num_bins):
                raise RuntimeError("It should be impossible for the softmax outputs to be <=0 or >1")
            # Every sample now has a bin index between 1 and num_bins (inclusive)

            for i in range(len(predictions)):  # Probably can be vectorised but this is clearer for now
                bin_confidences_sum[confidence_bins[i] - 1] += confidences[i]
                bin_counts[confidence_bins[i] - 1] += 1
                bin_accuracies_sum[confidence_bins[i] - 1] += predictions[i] == targets[i]

        print("After ECE: avg test accuracy = {:6.3f}%".format(100 * np.sum(bin_accuracies_sum) / np.sum(bin_counts)))

        total_count = np.sum(bin_counts)
        bin_counts[np.where(bin_counts == 0)] += 1  # as accuracy or conf sum is zero anyway this just stops a nan error
        per_bin_accuracies = bin_accuracies_sum / bin_counts
        per_bin_confidences = bin_confidences_sum / bin_counts

        # Eqn 3 in https://arxiv.org/pdf/1706.04599.pdf
        # Multiply by 100 to get a percent
        ece = 100 * np.sum((bin_counts / total_count) * np.abs(per_bin_accuracies - per_bin_confidences))
        # print("Expected Calibration Error = {:6.3f}%".format(ece))

        # Eqn 5 in https://arxiv.org/pdf/1706.04599.pdf
        # mce = np.max(np.abs(per_bin_accuracies - per_bin_confidences))
        # print("Maximum Calibration Error = {:6.3f}%".format(100 * mce))
    return ece


###################################################################
# Ckpt saving and loading #########################################
###################################################################

def get_ckpt_name(epochs, network, seed, stats_layer=None, tau=None, shot=False):
    if shot:
        pretr_ckpt_name = "pretrain-learner-shot-{}_{}_{}".format(epochs, network, seed)
    else:
        pretr_ckpt_name = "pretrain-learner-{}_{}_{}".format(epochs, network, seed)
    if stats_layer is not None:
        pretr_ckpt_name += "_{}".format(stats_layer)
    if tau is not None:
        pretr_ckpt_name += "_{}".format(tau)
    pretr_ckpt_name += ".pth.tar"
    return pretr_ckpt_name


def save_ckpt(ckpt_dir, model_name, model, optim, episode=0, *args):
    # Join exp settings to form affix
    affix = "_".join(str(arg) for arg in args)

    # Remove model name and file extension from exp affix (in case full ckpt passed in for exp settings)
    if "pretrain" in affix:
        affix = affix.replace("{0}-{1}_".format(model_name, episode), '').replace(".pth.tar", "")

    # Setup
    mkdir_p(ckpt_dir)
    ckpt_name = '{0}-{1}_{2}.pth.tar'.format(model_name, episode, affix)
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    opt_state_dict = optim.state_dict() if optim is not None else None

    torch.save({
        'episode': episode,
        '{0}'.format(model_name): model.state_dict(),
        'optim': opt_state_dict
    }, ckpt_path)

    return ckpt_path


def save_ckpt_tensor(ckpt_dir, model_name, tensor, episode=0, *args):
    # Join exp settings to form affix
    affix = "_".join(str(arg) for arg in args)

    # Remove model name and file extension from exp affix (in case full ckpt passed in for exp settings)
    if "pretrain" in affix:
        affix = affix.replace("{0}-{1}_".format(model_name, episode), '').replace(".pth.tar", "")

    # Setup
    mkdir_p(ckpt_dir)
    tensor_name = '{0}-{1}_{2}.pt'.format(model_name, episode, affix)
    tensor_path = os.path.join(ckpt_dir, tensor_name)

    torch.save(tensor, tensor_path)

    return tensor_path


def resume_ckpt(model_name, model, optim, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        last_episode = ckpt['episode']
        model.load_state_dict(ckpt[model_name])
        optim.load_state_dict(ckpt['optim'])
        print("Resuming model from {}".format(ckpt_path))
        return last_episode, model, optim
    else:
        raise ValueError("No checkpoint found at {}".format(ckpt_path))


def load_ckpt(model_name, model, ckpt_path, device):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        last_episode = ckpt['episode']
        model.load_state_dict(ckpt[model_name])
        print("Loaded model from {}".format(ckpt_path))
        return last_episode, model
    else:
        raise ValueError("No checkpoint found at {}".format(ckpt_path))


def load_ckpt_tensor(tensor_path):
    if os.path.exists(tensor_path):
        tensor = torch.load(tensor_path)
        print("Loaded tensor from {}".format(tensor_path))
        return tensor
    else:
        raise ValueError("No tensor found at {}".format(tensor_path))


###################################################################
# SFDA useful funcions ###########################################
###################################################################

class FwdHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


def hook_linears(model):
    hooked_modules = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            print("Hooked module: {}.".format(m))
            hooked_modules.append(FwdHook(m))
    return hooked_modules


def set_dropout_to_eval(learner):
    for module in learner.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.eval()


def entropy(p, dim=0):
    return torch.sum(-p * torch.log(p + 1e-5), dim)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


def get_last_bn_stats(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            bn_mean = m.running_mean
            bn_var = m.running_var
    # Need to clone, otherwise just get updated running_mean and running_var
    bn_mean = torch.clone(bn_mean)
    bn_var = torch.clone(bn_var)
    return bn_mean, bn_var


def BNM_loss(pretr_mean, pretr_var, batch_mean, batch_var):
    return (1. / (2 * len(pretr_mean))) * torch.sum(torch.log(batch_var / pretr_var) +
            ((pretr_var + torch.square(pretr_mean - batch_mean)) / batch_var) - 1)


def IM_loss(predictions):
    # Takes in pre-softmax predictions
    softmax_preds = nn.Softmax(dim=1)(predictions)
    loss_ent = torch.mean(entropy(softmax_preds, dim=1))
    mean_softmax_pred = softmax_preds.mean(0)
    loss_div = -entropy(mean_softmax_pred)
    return loss_ent + loss_div


def get_fr_loss(scores, fr_type):
    if fr_type == 'features':
        return scores[-2].mean()
    if fr_type == 'logits':
        return scores[-1].mean()
    if fr_type == 'fc':
        return scores[-2].mean() + scores[-1].mean()
    raise ValueError("Invalid choice for FR type: {0}", format(fr_type))


def obtain_label_shot(dataloader, learner, threshold=0, distance="cosine"):
    """
    Based on https://github.com/tim-learn/SHOT
    """
    start_test = True
    learner.eval()
    with torch.no_grad():
        for inputs, labels, idx in dataloader:
            # print(idx)
            inputs = inputs.cuda()
            feas = learner.features(inputs)
            feas = torch.flatten(feas, 1)
            outputs = learner.classifier(feas)
            # hardcode where to extract features for pseudo-labelling
            for module in learner.classifier.modules():
                if isinstance(module, nn.Linear):
                    feas = module(feas)
                if isinstance(module, nn.BatchNorm1d):
                    feas = module(feas)
                    break
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # cosine distance
    if distance == "cosine":
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)  # num_classes
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>threshold)
    labelset = labelset[0]

    dd = cdist(all_fea, initc[labelset], distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    print(log_str+'\n')

    learner.train()

    return torch.from_numpy(pred_label.astype('int')).cuda()


def clone_module(module, memo=None):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """

    if memo is None:
        # Maps original data_ptr to the cloned tensor.
        # Useful when a Module uses parameters from another Module; see:
        # https://github.com/learnables/learn2learn/issues/174
        memo = {}

    # First, create a copy of the module.
    # Adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    # Finally, rebuild the flattened parameters for RNNs
    # See this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    clone = clone._apply(lambda x: x)
    return clone


def detach_module(module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Detaches all parameters/buffers of a previously cloned module from its computational graph.

    Note: detach works in-place, so it does not return a copy.

    **Arguments**

    * **module** (Module) - Module to be detached.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    detach_module(clone)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate on clone, not net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return
    # First, re-write all parameters
    for param_key in module._parameters:
        if module._parameters[param_key] is not None:
            detached = module._parameters[param_key].detach_()

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        if module._buffers[buffer_key] is not None and \
                module._buffers[buffer_key].requires_grad:
            module._buffers[buffer_key] = module._buffers[buffer_key].detach_()

    # Then, recurse for each submodule
    for module_key in module._modules:
        detach_module(module._modules[module_key])


###################################################################
# Misc Useful Functions ###########################################
###################################################################

def print_table(shift_surprises, column_titles, row_titles, title):
    tabular_mean_ss = [(r_title, *["{0:.1f}".format(shift_ss) for shift_ss in row_ss])
                       for r_title, row_ss in zip(row_titles, shift_surprises)]
    tabular_mean_ss.sort(key=lambda r: r[0])  # sort by task name
    results_table = tabulate(tabular_mean_ss, headers=["Surp. Type", *column_titles], tablefmt="rst")
    print(title)
    print(results_table)
    print()


def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)


def to_device(tensors, device):
    for t in tensors:
        t.to(device)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.uniform_(m.weight.data)


def seed_torch(seed=404, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Multi-GPU
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Sacrifice speed for exact reproducibility
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def reset_rngs(rng=None, seed=404, deterministic=False, cpu=True):
    if rng is not None:
        rng.seed(seed)
    seed_torch(seed, deterministic)


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

# Vectorized bincount
def bincount2D_vectorized(a, minlength=None):
    if minlength is None:
        minlength = a.max() + 1
    a_offs = a + torch.arange(a.shape[0], device=a.device)[:, None] * minlength
    return torch.bincount(a_offs.flatten(), minlength=a.shape[0] * minlength).reshape(-1, minlength)


###################################################################
# Probability dists ###############################################
###################################################################

EPS = 1e-8
_norm_pdf_C = math.sqrt(2*math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)


def calc_z_scores(xs, means, variances):
    return torch.abs(xs - means) / torch.sqrt(variances)


def gaussian_logpdf(x, mu, sigma):
    return - _norm_pdf_logC - torch.log(sigma + EPS) - 0.5*((x - mu) / (sigma + EPS))**2


def gaussian_pdf_np(x, mu, sigma):
    return np.exp(gaussian_logpdf(x, mu, sigma))


def gaussian_pdf(x, mu, sigma):
    return torch.exp(gaussian_logpdf(x, mu, sigma))


def gmm_pdf(x, locs, scales, weights):
    """
    Calculate the pdf of a Gaussian mixture model (GMM).
    """
    d = np.zeros_like(x)
    for loc, scale, weight in zip(locs, scales, weights):
        d += weight * scipy.stats.norm.pdf(x, loc=loc, scale=scale)
    return d


def gmm_pdf_torch(x, locs, scales, weights):
    """
    Calculate the pdf of a Gaussian mixture model (GMM).
    """
    d = torch.zeros_like(x)
    for loc, scale, weight in zip(locs, scales, weights):
        d += weight * gaussian_pdf(x, loc, scale)
    return d


def update_mixture_weights(x, weights, means, stds):
    """
    Calculate the updated mixture weights via:

    \hat{w}_k = 1/n * \sum_i r_{ki},
    r_{ki} = (w_k * p_k(x_i)) / (\sum_j w_j * p_j(x_i)),

    where r_{ki} is the "responsibility" or posterior prob of sample i under gaussian k, and
    p_k(x_i)=N(x_i; \mu_k, \sigma_k) for this case of gaussian distributions.

    :param x: Array of n samples.
    :param weights: Array of k weights.
    :param means: Array of k means.
    :param stds: Array of k standard deviations.
    :return: Array of k updated weights.
    """
    probs_ki = torch.zeros((weights.size()[0], len(x)))  # init matrix of shape [num dists, num samples]
    weights = weights.reshape(-1, 1)
    for k, w in enumerate(weights):
        probs_ki[k] = gaussian_pdf(x, means[k], stds[k])
    # probs_ki = gaussian_pdf(x, means, stds)
    norm = torch.mm(weights.T, probs_ki)   # normaliser / denominator
    probs_ki = probs_ki * weights  # each entry ki is unnormed resp_ki, via row-wise multiplication (broadcasting)
    probs_ki = probs_ki / (norm + EPS)     # each entry ki is resp_ki, via column-wise division (broadcasting)
    return 1./float(len(x)) * torch.sum(probs_ki, 1)


###################################################################
# Plotting functions ##############################################
###################################################################

DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']


def arg_condition(arr, cond):
    """
    Return first and last entries that satisfy the given condition in a 1-D array, i.e. indicies
    required to remove leading and trailing entries that do not satisfy the condition.
    :param arr: array
    :return: tuple: (first idx, final idx)
    """
    first = 0
    for e in arr:
        if cond(e):
            break
        else:
            first += 1
    last = len(arr)
    for e in arr[::-1]:
        if cond(e):
            break
        else:
            last -= 1
    return first, last


def arg_trim_zeros(arr):
    """
    Return first and last non-zero entries in a 1-D array, i.e. indicies
    required to remove leading and trailing zeros.
    :param arr: array values
    :return: tuple: (first idx, final idx)
    """
    return arg_condition(arr, lambda x: x != 0.)


def arg_trim_close_zeros(arr, tol=1e-8):
    return arg_condition(arr, lambda x: x > tol)


def select_inds(n_units, max_units, shuffle=False, rng=None):
    rng = np.random.RandomState(404) if rng is None else rng
    if n_units <= max_units:
        inds = np.array(list(range(n_units)))
        if shuffle:
            rng.shuffle(inds)
    else:
        inds = rng.choice(range(n_units), max_units, replace=False)
        if not shuffle:
            inds.sort()
    return inds


def add_prefix_to_dict_keys(d, prefix):
    d = dict(d)            # accepts e.g. .npz objects
    ks = list(d.keys())    # pull out keys in advance to prevent recursion
    for k in ks:
        d[prefix + k] = d.pop(k)
    return d


def remove_prefix_from_dict_keys(d, prefix):
    d = dict(d)            # accepts e.g. .npz objects
    ks = list(d.keys())    # pull out keys in advance to prevent recursion
    for k in ks:
        d[k.replace(prefix + ".", "")] = d.pop(k)
    return d


def get_fig_params(n_r):
    w = 4                                         # default for matplotlib
    row_ext_ratio = (n_r / 3.)                      # default = 3 rows
    h = w * row_ext_ratio + 1.5
    t_b_adjust = 0.1 * (1. / row_ext_ratio) + 0.01
    suptitle_y = 0.995 + (0.0025 * n_r / 40.)       # annoying but necessary tweaking -- title at the top of a long fig
    return w, h, t_b_adjust, suptitle_y


def get_indices(input_shape, max_units, max_locs=0, diag=True):
    b, c, h, w = input_shape
    max_units = c if max_units is None else max_units
    max_locs = h*w if max_locs is None else max_locs

    if w > 0:  # conv layer
        # select indices of units to plot
        n_units = min(c, max_units)
        c_is = select_inds(c, n_units)

        if diag:
            n_locs = min(h, w, max_locs)  # ensures len(h_is) = len(w_is), i.e. valid (h_i, w_i) pairs
            h_is = np.array(list(range(n_locs)))
            w_is = np.array(list(range(n_locs)))
        else:
            h_w_pairs = np.array(list(np.ndindex(h, w)))  # [(0,0), (0,1), (0,2) ...]
            h_w_pair_inds = select_inds(h*w, max_locs)    # [1, 4, 7, ...]
            h_w_inds = h_w_pairs[h_w_pair_inds]           # [(0,0), (1,2), (3,1) ...]
            h_is, w_is = list(zip(*h_w_inds))             # (0,1,3,...), (0,2,1,...)]

        inds = [c_is, h_is, w_is]

    else:   # fc layer
        # select indices of units to plot
        n_units = min(c, max_units)
        inds = select_inds(c, n_units)

    return inds


def plot_samples(all_samples, indices, title, labels=None, fpath=None,
                 scores=None, score_symbol="s", ax_title="Unit", n_columns=None):
    # all_samples: list of distribution samples to compare
    if labels is None:
        labels = ["D{0}".format(i) for i in range(len(all_samples))]

    # Setup fig dims: n_columns, n_rows, width, height.
    if n_columns is None:
        n_columns = 2
    n_rows = len(indices) // n_columns
    w, h, t_b_adjust, suptitle_y = get_fig_params(n_rows)

    fig, _ = plt.subplots(n_rows, n_columns, figsize=(w, h))
    for i, ax in enumerate(fig.axes):
        if scores is not None:
            ax.set_title("{0} {1} | ${2}={3:.2f}$".format(ax_title, indices[i], score_symbol, scores[i]))
        else:
            ax.set_title("{0} {1}".format(ax_title, indices[i]))

        for samples, label in zip(all_samples, labels):
            ax.hist(samples[i], density=True, alpha=0.7, label=label, histtype="step")

    # # Blank plots for unused/overflow axes
    # n_blank = len(axs) - len(test_samples)
    # if n_blank > 0:
    #     for i in range(n_blank):
    #         axs[-(i + 1)].axis("off")

    # Adjust figures
    fig.tight_layout()
    fig.suptitle(title, fontsize=14, y=suptitle_y)
    fig.subplots_adjust(top=(1. - t_b_adjust), bottom=t_b_adjust)

    # Add a single legend, centre bottom of the figure
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in by_label.values()]
    fig.legend(new_handles, by_label.keys(), 'lower center', ncol=2)

    # Save or show
    if fpath is None:
        fig.show()
    else:
        fig.savefig(fpath)

    plt.close(fig)


def plot_bins(bin_edges, all_counts, title, labels=None, fpath=None,
              scores=None, score_symbol="s", ax_title="Unit", n_columns=None, trim_zero_bins=False):
    """
    Plot a nested list of bin counts.
    :param bin_edges: array of bin edges with shape [n_features, n_bins+1] (shared by all dists).
    :param all_counts: array of bin counts with shape [n_dists, n_features, n_bins].
    :param title: str title for plot.
    :param labels: list of labels with shape [n_dists].
    :param fpath: filepath at which to save the plot.
    :param scores:
    :param score_symbol:
    :param ax_title:
    :param n_columns:
    """
    # all_counts: list of distribution bin counts to compare
    if labels is None:
        labels = ["D{0}".format(i) for i in range(len(all_counts))]

    # Setup fig dims: n_columns, n_rows, width, height.
    if n_columns is None:
        n_columns = 2
    n_rows = len(bin_edges) // n_columns
    w, h, t_b_adjust, suptitle_y = get_fig_params(n_rows)

    fig, _ = plt.subplots(n_rows, n_columns, figsize=(w, h), sharex=True, sharey=True)
    for i, ax in enumerate(fig.axes):                       # iterate over features
        if scores is not None:
            ax.set_title("${0}_{{{1}}}$ | {2}$={3:.1f}$".format(ax_title, i, "$D_{SKL}$", scores[i]))
        else:
            ax.set_title("{0} {1}".format(ax_title, i))

        # if np.allclose(bin_edges[i], 0.):  # skip 'dead' relu units, all activations ~= 0!
        #     continue

        # # Remove leading and trailing zero bins
        if trim_zero_bins:
            total_counts = all_counts[0][i]
            for counts in all_counts[1:]:
                total_counts += counts[i]
            first, last = arg_trim_zeros(total_counts)
            trimmed_b_edges = bin_edges[i][first:last + 1]
        else:
            trimmed_b_edges = bin_edges[i]
            center_edges = bin_edges[i][1:-1]
            center_edges = center_edges - center_edges[0]
            trimmed_b_edges = np.array([-0.125] + list(center_edges) + [1.125])
            print(trimmed_b_edges)

        # Plot bin counts on training data, if exists
        for counts, label in zip(all_counts, labels):
            if trim_zero_bins:
                trimmed_counts = counts[i][first:last] + EPS
            else:
                trimmed_counts = counts[i]
            ax.hist(trimmed_b_edges[:-1], trimmed_b_edges, weights=trimmed_counts, density=True,
                    alpha=1., label=label, histtype="step", linewidth=1.5)

    # Adjust figure to have readable grid and titles
    plt.gcf().tight_layout()
    # fig.suptitle(title, fontsize=14, y=suptitle_y)
    # fig.subplots_adjust(top=(1. - t_b_adjust), bottom=t_b_adjust)
    fig.subplots_adjust(bottom=t_b_adjust)

    # Construct a single figure legend for all subplots, located at the bottom
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in by_label.values()]
    fig.legend(new_handles, by_label.keys(), 'lower center', ncol=2, fontsize=12)

    print(fpath)
    print(n_rows)

    if fpath is None:
        fig.show()
    else:
        fig.savefig(fpath)

    plt.close(fig)


def plot_gaussians(means, vars, title, labels=None, fpath=None, scores=None,
                  score_symbol="s", ax_title="Unit", n_columns=None):

    # List of Gaussian distributions (means and vars) to compare
    assert len(means) == len(vars)
    if labels is None:
        labels = ["D{0}".format(i) for i in range(len(means))]

    # Setup fig dims: n_columns, n_rows, width, height.
    if n_columns is None:
        n_columns = 2
    n_rows = len(means[0]) // n_columns
    w, h, t_b_adjust, suptitle_y = get_fig_params(n_rows)
    fig, _ = plt.subplots(n_rows, n_columns, figsize=(w, h))

    #  Plot fig
    for i, ax in enumerate(fig.axes):
        if scores is not None:
            ax.set_title("{0} {1} | {2}$={3:.1f}$".format(ax_title, i, score_symbol, scores[i]))
        else:
            ax.set_title("{0} {1}".format(ax_title, i))

        # Plot Gaussian distributions from mean and var
        for ms, vs, label in zip(means, vars, labels):
            mu = ms[i]
            sigma = math.sqrt(vs[i])
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), label=label)

    # Adjust figure to have readable grid and titles
    fig.tight_layout()
    fig.suptitle(title, fontsize=14, y=suptitle_y)
    fig.subplots_adjust(top=(1. - t_b_adjust), bottom=t_b_adjust)

    # Construct a single figure legend for all subplots, located at the bottom
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    new_handles = [Line2D([], [], c=h.get_color()) for h in by_label.values()]
    fig.legend(new_handles, by_label.keys(), 'lower center', ncol=2)

    if fpath is None:
        fig.show()
    else:
        fig.savefig(fpath)

    plt.close(fig)


def plot_mogs(centres, stds, all_weights, title, labels=None, fpath=None, scores=None,
              score_symbol="s", ax_title="Unit", n_columns=None):
    # all_weights: list of distributions (weighted gaussians with shared means/centres and stds) to compare
    if labels is None:
        labels = ["D{0}".format(i) for i in range(len(all_weights))]

    # Setup fig dims: n_columns, n_rows, width, height.
    if n_columns is None:
        n_columns = 2
    n_rows = len(centres) // n_columns
    w, h, t_b_adjust, suptitle_y = get_fig_params(n_rows)
    fig, _ = plt.subplots(n_rows, n_columns, figsize=(w, h))

    #  Plot fig
    for i, ax in enumerate(fig.axes):
        if scores is not None:
            ax.set_title("{0} {1} | {2}$={3:.1f}$".format(ax_title, i, score_symbol, scores[i]))
        else:
            ax.set_title("{0} {1}".format(ax_title, i))

        # Plot Gaussian mixture distributions for fixed centres and stds (only weights/coefficients differ)
        cs = centres[i]
        ss = stds[i]
        x = np.linspace(cs[0], cs[-1], 100)  # assume ordered centres and stds
        for weights, label in zip(all_weights, labels):
            ws = weights[i]
            ax.plot(x, gmm_pdf(x, cs, ss, ws), label=label)

    # Adjust figure to have readable grid and titles
    fig.tight_layout()
    fig.suptitle(title, fontsize=14, y=suptitle_y)
    fig.subplots_adjust(top=(1. - t_b_adjust), bottom=t_b_adjust)

    # Construct a single figure legend for all subplots, located at the bottom
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    new_handles = [Line2D([], [], c=h.get_color()) for h in by_label.values()]
    fig.legend(new_handles, by_label.keys(), 'lower center', ncol=2)

    if fpath is None:
        fig.show()
    else:
        fig.savefig(fpath)

    plt.close(fig)


def square_grid(data):
    grid_size = math.ceil(math.sqrt(data.shape[0]))
    return grid_size, data[:grid_size ** 2]


def unnorm_imgs(imgs):
    return (imgs + 1.) / 2.0


def imshow(imgs, filepath="test.png", unnorm=True, fig_title=None, img_titles=None, max_imgs=None):
    # init square figure
    max_imgs = 64 if max_imgs is None else max_imgs
    imgs = np.array(imgs[:max_imgs])  # plot up to 8x8 grid
    grid_size, imgs = square_grid(imgs)
    plt.figure(figsize=(grid_size*2, grid_size*2))

    # unnormalize
    if unnorm:
        imgs = unnorm_imgs(imgs)

    # (B, C, H, W) --> (B, H, W, C)
    imgs = imgs.transpose([0, 2, 3, 1])

    # plot
    for i, img in enumerate(imgs):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img, interpolation='none')
        plt.axis('off')
        if img_titles is not None:
            plt.title(img_titles[i], fontsize=8)

    if fig_title is not None:
        plt.suptitle(fig_title, fontsize=18)

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    else:
        plt.show()

    plt.close()


def pad(a, length, neg_pad=False):
    if neg_pad:
        b = -np.ones(length)
    else:
        b = np.zeros(length)
    b[:len(a)] = a
    return b



def display_data(images, labels, use_int_targets=True):
    """
    Display the data in a grid, i.e. images and corresponding labels.
    :param use_int_targets: bool: integer targets rather than one-of-k.
    """
    images = images[:64]  # plot up to 8x8 grid
    labels = labels[:64]

    imshow(images)
    if not use_int_targets:
        labels = np.argmax(labels, 1)  # from one-of-k to int
    grid_size, labels = square_grid(labels)
    labels = pad(labels, grid_size**2, neg_pad=True)
    print(labels.reshape((grid_size, grid_size)))


def get_bsr_backgrounds(bst_path):
    f = tarfile.open(bst_path)
    f_paths = []
    for name in f.getnames():
        if name.startswith('BSR/BSDS500/data/images/train/'):
            f_paths.append(name)
    print('Loading BSR training images')

    bg_imgs = []
    bg_names = []
    for f_path in f_paths:
        try:
            fp = f.extractfile(f_path)
            bg_img = imread(fp)
            bg_imgs.append(bg_img)
            bg_name = f_path.split('/')[-1].split('.')[0]  # '...data/images/splash.png' --> 'splash'
            print(f_path)
            print(bg_name)
            bg_names.append(bg_name)
        except IOError:
            continue
    return bg_imgs, bg_names