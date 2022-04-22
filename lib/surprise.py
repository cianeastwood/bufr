import math
import numpy as np
import torch
from torch import nn
# from pyemd import emd
from lib.utils import update_mixture_weights
import time

EPS = 1e-8
NORM = False


def euclidean_pairwise_distance_matrix(x):
    """Calculate the Euclidean pairwise distance matrix for a 1D array."""
    distance_matrix = np.abs(np.repeat(x, len(x)) - np.tile(x, len(x)))
    return distance_matrix.reshape(len(x), len(x))


def prep_bins_accurate(ps, qs, edges=None, alpha=0.01, value="probs", smooth_q=True):
    """
    Prepare bins. Includes removing end bins with zero counts and add-one smoothing.
    TODO:
        1) Compare performance of removing zero bins to leaving them (add-one smoothing)
            -i) Speed (array processing given same number of bins)
            -ii) Accuracy (change in scores for all but dead units)
            -iii) How big do N and M have to be in order to have no significant difference?
        2) Add-one smoothing: p only?
            -i) Issue: 1 count for p, 0 for q --> large Opp! Reduces surprise *increase*
            -ii) Solution: Either remove zero bins and add-one to p, or leave zero bins and add-one to p AND q.

    :param ps: train dist bin counts.
    :param qs: test dist bin counts.
    :param edges: bin edges.
    :param alpha: smoothing "pseudocount". 0 corresponds to no smoothing, while 1 corresponds to add-one smoothing.
    :param value: value to return.
    :return: array: counts, probs or densities.
    """
    valid_values = ["counts", "probs", "densities"]
    if value not in valid_values:
        raise ValueError("Invalid value selected {0}.\nChoose one of {1}".format(value, valid_values))

    # get total counts (assuming they are the same for each feature)
    n_ps = ps[0].sum()
    n_qs = qs[0].sum()

    # remove bins where both obs and exp counts are zero. New name to avoid unexpected behaviour from mutable ps,qs
    nonzero_inds = (ps + qs) != 0
    ps_ = [p[nonz_is] for p, nonz_is in zip(ps, nonzero_inds)]
    qs_ = [q[nonz_is] for q, nonz_is in zip(qs, nonzero_inds)]

    # add-one smoothing to prevent extreme values (p only?)
    ks = [len(p) for p in ps_]                          # num non-empty bins for each feature
    ps_ = [p + alpha for p in ps_]
    n_ps = np.array([n_ps + k*alpha for k in ks])       # update total counts
    if smooth_q:
        qs_ = [q + alpha for q in qs_]
        n_qs = np.array([n_qs + k * alpha for k in ks])  # update total counts
    else:
        n_qs = np.array([n_qs for _ in ks])

    # calculate desired value
    if value == "counts":
        return ps_, qs_, n_ps, n_qs

    if value == "probs":
        ps_ = [p / float(n_p) for p, n_p in zip(ps_, n_ps)]
        qs_ = [q / float(n_q) for q, n_q in zip(qs_, n_qs)]
        return ps_, qs_, n_ps, n_qs

    if value == "densities":
        if edges is None:
            raise ValueError("Need bin edges (widths) to calculate densities!")
        hs = edges[:, 1:] - edges[:, :-1]  # bin widths
        hs = [h[nonz_is] for h, nonz_is in zip(hs, nonzero_inds)]  # bin widths for non-zero bins
        ps_ = [p / h / float(n_p) for p, h, n_p in zip(ps_, hs, n_ps)]
        qs_ = [q / h / float(n_q) for q, h, n_q in zip(qs_, hs, n_qs)]
        return ps_, qs_, n_ps, n_qs


def prep_bins_fast(p_cs, q_cs, edges=None, alpha=0.01, value="probs", smooth_q=True):
    valid_values = ["counts", "probs", "densities"]
    if value not in valid_values:
        raise ValueError("Invalid value selected {0}.\nChoose one of {1}".format(value, valid_values))

    p_cs_ = p_cs.copy()                                     # mutable object
    q_cs_ = q_cs.copy()                                     # mutable object
    nonzero_inds = (p_cs + q_cs) != 0
    n_bins = nonzero_inds.sum(1).reshape(-1, 1)

    # smoothing using "pseudocount" alpha. Prevents extreme values (p only for surprise). alpha=0 means no smoothing.
    p_cs_ += alpha
    if smooth_q:
        q_cs_ += alpha

    # get total counts (assuming they are the same for each feature)
    n_ps = p_cs_.sum(1).reshape(-1, 1)
    n_qs = q_cs_.sum(1).reshape(-1, 1)

    # calculate desired value
    if value == "counts":
        return p_cs_, q_cs_, n_ps, n_qs, n_bins

    if value == "probs":
        p_cs_ = p_cs_ / n_ps
        q_cs_ = q_cs_ / n_qs
        return p_cs_, q_cs_, n_ps, n_qs, n_bins

    if value == "densities":
        if edges is None:
            raise ValueError("Need bin edges (widths) to calculate densities!")
        hs = edges[:, 1:] - edges[:, :-1] + EPS  # non-zero bin widths
        p_cs_ = p_cs_ / hs / n_ps
        q_cs_ = q_cs / hs / n_qs
        return p_cs_, q_cs_, n_ps, n_qs, n_bins


def bin_variance(ps, n_qs, norm=NORM):
    """
    Calculate the Gaussian approx for the confidence interval under the null hypothesis that the q_i's are
    obtained as q_i = x_i/n by sampling n times from the multinomial with probs p = (p_1, ..., p_k) to obtain
    counts(x_1, ..., x_k).

    :param ps: (f x r) array of probs p = (p_1, ..., p_k), where f is the num of features of k is the num of bins.
    :param n_qs: array containing the number of q "samples" for each feature (usually the same...)
    :param norm: "normalised" surprise score is being used (dividing score by h_p)
    :return: array of variances that define confidence intervals around the surprise score.
    """
    vs = []
    for p_is, n in zip(ps, n_qs):
        if len(p_is) == 1:
            # single bin with prob = 1. for both p and q
            vs.append(EPS)
            continue
        w = - np.log(p_is + EPS)  # surprise
        var_di = p_is * (1. - p_is) / n
        cov_di_dj = - np.outer(p_is, p_is) / n
        cov_d = cov_di_dj - np.diag(np.diag(cov_di_dj)) + np.diag(var_di)
        var_stat = w.T.dot(cov_d).dot(w)
        if norm:
            o_pp = (p_is.dot(w) + EPS)
            var_stat /= ((o_pp ** 2) + EPS)
        vs.append(var_stat)

    return np.array(vs)


def get_bin_entropies(p_cs, q_cs, fast=True, alpha=0.01):
    if fast:
        # Prep bins: remove bins where p_count AND q_count = 0, smooth p and q, counts  --> probabilities
        ps, qs, _, n_qs, n_bins = prep_bins_fast(p_cs, q_cs, alpha=alpha)

        # Change log base = num bins (put h_p and h_q in range [0,1])
        log_base_change_divisor = np.log(n_bins)
        single_bin_inds = (log_base_change_divisor == 0.)  # single bin with prob = 1. for both p and q
        log_base_change_divisor[single_bin_inds] = 1.

        # Calculate entropies
        h_ps = -np.sum(ps * np.log(ps + EPS) / log_base_change_divisor, 1)
        h_qs = -np.sum(qs * np.log(qs + EPS) / log_base_change_divisor, 1)
        h_q_ps = -np.sum(qs * np.log(ps + EPS) / log_base_change_divisor, 1)
        h_p_qs = -np.sum(ps * np.log(qs + EPS) / log_base_change_divisor, 1)

        # Set entropy = exactly zero for units/features with a single bin prob = 1. for both p and q
        h_ps[single_bin_inds[:, 0]] = 0.
        h_qs[single_bin_inds[:, 0]] = 0.
        h_q_ps[single_bin_inds[:, 0]] = 0.
        h_p_qs[single_bin_inds[:, 0]] = 0.

    else:  # cannot do array-based operations as each feature has a diff num of bins
        # Prep bins: remove bins where p_count AND q_count = 0, smooth p (length of code), counts  --> probabilities
        ps, qs, _, n_qs = prep_bins_accurate(p_cs, q_cs, alpha=alpha)
        n_bins = np.array([len(p) for p in ps])

        # Calculate entropies
        h_ps = []
        h_qs = []
        h_q_ps = []
        h_p_qs = []
        for i, (p_is, q_is, k) in enumerate(zip(ps, qs, n_bins)):
            if len(p_is) == 1:  # single bin with prob = 1. for both p and q. Zero entropy/randomness!
                h_ps.append(0.), h_qs.append(0.), h_q_ps.append(0.), h_p_qs.append(0.),
                continue
            h_p = -np.sum(p_is * np.log(p_is + EPS) / np.log(k))
            h_q = -np.sum(q_is * np.log(q_is + EPS) / np.log(k))
            h_q_p = -np.sum(q_is * np.log(p_is + EPS) / np.log(k))
            h_p_q = -np.sum(p_is * np.log(q_is + EPS) / np.log(k))
            h_ps.append(h_p), h_qs.append(h_q), h_q_ps.append(h_q_p), h_p_qs.append(h_p_q)
        h_ps, h_qs, h_q_ps, h_p_qs = np.array(h_ps), np.array(h_qs), np.array(h_q_ps), np.array(h_p_qs)

    return h_ps, h_qs, h_q_ps, h_p_qs, ps, qs, n_qs


def get_soft_bin_entropies(p_cs, q_cs, alpha=0.0):
    # Smoothing
    ps = p_cs + alpha
    qs = q_cs + alpha
    # ps = p_cs + alpha / torch.sum(p_cs, 1, keepdim=True)
    # qs = q_cs + alpha / torch.sum(q_cs, 1, keepdim=True)

    # Re-normalize
    n_ps = ps.sum(1).reshape(-1, 1)
    n_qs = qs.sum(1).reshape(-1, 1)
    ps = ps / n_ps
    qs = qs / n_qs

    # Calculate entropies
    n_bins = ps.shape[1]
    h_ps = -torch.sum(ps * torch.log(ps + EPS) / math.log(n_bins), 1)
    h_qs = -torch.sum(qs * torch.log(qs + EPS) / math.log(n_bins), 1)
    h_q_ps = -torch.sum(qs * torch.log(ps + EPS) / math.log(n_bins), 1)
    h_p_qs = -torch.sum(ps * torch.log(qs + EPS) / math.log(n_bins), 1)

    return h_ps, h_qs, h_q_ps, h_p_qs, ps, qs, n_qs


def get_gaussian_entropies(p_means, p_vars, q_means, q_vars):
    h_p = 0.5 * torch.log(2. * math.pi * math.e * p_vars)
    h_q = 0.5 * torch.log(2. * math.pi * math.e * q_vars)
    h_p_q = 0.5 * torch.log(2. * math.pi * q_vars) + (p_vars + torch.square(p_means - q_means)) / (2. * q_vars)
    h_q_p = 0.5 * torch.log(2. * math.pi * p_vars) + (q_vars + torch.square(q_means - p_means)) / (2. * p_vars)
    return h_p, h_q, h_p_q, h_q_p


def kl_gaussian(p_means, p_vars, q_means, q_vars):
    """ KL(P||Q) for univariate Normal distributions P and Q.
    See https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians for derivation.
    """
    return torch.log(torch.sqrt(q_vars) / torch.sqrt(p_vars)) + (p_vars + (p_means - q_means)**2) / (2.*q_vars) - 0.5


def surprise_bins(p_cs, q_cs, score_type="KL_Q_P", fast=True, alpha=0.01, bin_edges=None):
    """
    Calculate desired surprise score given bin counts.
    :param p_cs: bin counts from P.
    :param q_cs: bin counts from Q.
    :param score_type: desired score type (str).
    :param fast: sacrifice some accuracy for speed (bool). Depends on method for bins that are empty for both p and q.
    :param alpha: smoothing 'pseudocount' (float).
    :return: array of scores.

    Note: counts for *both* p and q are currently smoothed using the same alpha, regardless of which defines the
    code length (log(p) / log(q)). This makes calculations easier and entropy calculations stable. However, there may
    be reason not to smooth p and/or q. Would involve get_bin_entropies(q_cs, p_cs).

    TODO:
        1) Can we reduce computation by storing ps (probs) and h_p? Definitely if we can use "empirical" H(q,p) --
        see surprise_MoGs fn.
    """
    if score_type == "JS":                  # Jensen–Shannon divergence, M is the average of distributions P and Q.
        m_cs = 0.5 * (p_cs + q_cs)
        h_p, _, _, h_p_m, _, _, _ = get_bin_entropies(p_cs, m_cs, fast=fast, alpha=alpha)
        h_q, _, _, h_q_m, _, _, _ = get_bin_entropies(q_cs, m_cs, fast=fast, alpha=alpha)  # slight redundancy in calcs
        return 0.5 * (h_p_m - h_p) + 0.5 * (h_q_m - h_q)

    h_p, h_q, h_q_p, h_p_q, ps, qs, n_qs = get_bin_entropies(p_cs, q_cs, fast=fast, alpha=alpha)

    if score_type == "SI":
        return h_q_p - h_p
    elif score_type == "SI_norm":
        return (h_q_p - h_p) / (np.abs(h_p) + EPS)
    elif score_type == "SI_Z":
        sampling_std = np.sqrt(np.abs(bin_variance(ps, n_qs)) + EPS)
        return (h_q_p - h_p) / (sampling_std + EPS)
    elif score_type == "KL_Q_P":                # KL(Q||P)
        return h_q_p - h_q
    elif score_type == "KL_P_Q":                # KL(P||Q)
        return h_p_q - h_p
    elif score_type == "PSI":                   # Population stability index (Symmetric KL divergence)
        return (h_p_q - h_p) + (h_q_p - h_q)
    elif score_type == "EMD":
        raise NotImplementedError("Coming soon from https://github.com/wmayner/pyemd")
        #  TODO:
        #   1) bin edges need to match counts when zero-bins have been removed.
        #   2) Unit testing (speed + accuracy).
        # if bin_edges is None:
        #     raise ValueError("Bin edges are required to compute the distance matrix for the EMD(P,Q).")
        # emds = np.zeros(len(ps))
        # bin_locations = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)                        # centre of bins
        # for i, (unit_ps, unit_qs, unit_bin_locations) in enumerate(zip(ps, qs, bin_locations)):
        #     distance_matrix = euclidean_pairwise_distance_matrix(unit_bin_locations)            # k x k matrix
        #     emds[i] = emd(unit_ps, unit_qs, distance_matrix)
        # return np.array(emds)
    else:
        raise ValueError("Invalid surprise score choice {0} for bins.".format(score_type))


def surprise_soft_bins(p_cs, q_cs, score_type="PSI", alpha=0.):

    """
    Calculate desired surprise score given bin counts.
    :param p_cs: bin counts from P.
    :param q_cs: bin counts from Q.
    :param score_type: desired score type (str).
    :param alpha: smoothing 'pseudocount' (float).
    :return: array of scores.
    Note: counts for *both* p and q are currently smoothed using the same alpha, regardless of which defines the
    code length (log(p) / log(q)). This makes calculations easier and entropy calculations stable. However, there may
    be reason not to smooth p and/or q. Would involve get_bin_entropies(q_cs, p_cs).
    TODO:
        1) Can we reduce computation by storing ps (probs) and h_p? Definitely if we can use "empirical" H(q,p) --
        see surprise_MoGs fn.
    """
    if score_type == "JS":                  # Jensen–Shannon divergence, M is the average of distributions P and Q.
        m_cs = 0.5 * (p_cs + q_cs)
        h_p, _, _, h_p_m, _, _, _ = get_soft_bin_entropies(p_cs, m_cs, alpha=alpha)
        h_q, _, _, h_q_m, _, _, _ = get_soft_bin_entropies(q_cs, m_cs, alpha=alpha)  # slight redundancy in calcs
        return 0.5 * (h_p_m - h_p) + 0.5 * (h_q_m - h_q)

    if score_type == "Wass":
        ps = p_cs / p_cs.sum(1).reshape(-1, 1)
        qs = q_cs / q_cs.sum(1).reshape(-1, 1)
        return torch_wasserstein_loss(ps, qs)

    h_p, h_q, h_q_p, h_p_q, ps, qs, n_qs = get_soft_bin_entropies(p_cs, q_cs, alpha=alpha)

    if score_type == "SI":
        return h_q_p - h_p
    elif score_type == "SI_norm":
        return (h_q_p - h_p) / (torch.abs(h_p) + EPS)
    elif score_type == "SI_Z":
        raise NotImplementedError
    elif score_type == "KL_Q_P":                # KL(Q||P)
        return h_q_p - h_q
    elif score_type == "KL_P_Q":                # KL(P||Q)
        return h_p_q - h_p
    elif score_type == "PSI":                   # Population stability index (Symmetric KL divergence)
        return (h_p_q - h_p) + (h_q_p - h_q)
    elif score_type == "EMD":
        raise NotImplementedError
    else:
        raise ValueError("Invalid surprise score choice {0} for bins.".format(score_type))


def surprise_gaussian(p_means, p_vars, q_means, q_vars, xs=None, score_type="KL_Q_P"):
    """
    Calculate desired surprise score, where P and Q are 1D Gaussians.
    :param p_means: array of P means.
    :param p_vars: array of P variances.
    :param q_means: array of Q means.
    :param q_vars: array of Q variances.
    :param xs: array of samples from Q. Only needed to construct sampling distribution under Q for SI_Z score.
    :param score_type: desired score (str). Default = "KL_Q_P".
    :return: array of scores.
    """
    h_p, h_q, h_p_q, h_q_p = get_gaussian_entropies(p_means, p_vars, q_means, q_vars)

    if score_type == "SI":
        return h_q_p - h_p
    elif score_type == "SI_norm":
        return (h_q_p - h_p) / (h_p + EPS)
    elif score_type == "SI_Z":
        if xs is None:
            raise ValueError("Need samples from Q to calculate sampling distribution for SI_Z score.")
        sampling_std = torch.sqrt(p_vars / xs.size()[1])    # var = sigma^2/n, where n = n_qs = num samples per unit
        return (h_q_p - h_p) / (sampling_std + EPS)         # zero-mean scores
    elif score_type == "KL_Q_P":  # KL(Q||P)
        return h_q_p - h_q
    elif score_type == "KL_P_Q":  # KL(P||Q)
        return h_p_q - h_p
    elif score_type == "PSI":                               # Population stability index (Symmetric KL divergence)
        return (h_p_q - h_p) + (h_q_p - h_q)
    elif score_type == "JS":                                # Jensen–Shannon divergence, M = avg of dists P and Q
        raise NotImplementedError("Jensen–Shannon divergence not current supported for Gaussians as the average or "
                                  "mixture distribution causes complications when calculating differential entropies.")
    elif score_type == "EMD":
        raise NotImplementedError
    else:
        raise ValueError("Invalid surprise score choice {0} for Gaussians.".format(score_type))


def surprise_moments(p_moments, q_moments, mins, maxs, normed_range=True, score_type="CMD"):
    cmds = torch.zeros_like(mins)
    for n, (p_m, q_m) in enumerate(zip(p_moments, q_moments), 1):
        if normed_range:
            normalizer = 1.
        else:
            normalizer = 1. / torch.abs(maxs - mins)**n
        cmds = cmds + normalizer * torch.abs(p_m - q_m)                 # vector norm --> abs for scalars
    return cmds


def surprise_mogs(p_weights, q_weights, score_type="KL", fast=True, alpha=0.0001):
    """
    Calculate desired surprise score, where P and Q are 1D GMMs with fixed means and variances.

    As in Hershey and Olsen (2007), an upper bound on the KL divergence between two GMMs with the same
    number of components is given by: D(P||Q) <= D(w^p||w^q) + \sum_i w^p_i*D(P_i||Q_i), where w^p, w^q are the
    mixture weights for p and q respectively, and P_i, Q_i are the ith Gaussians in the mixtures of P and Q.

    In our setup, D(P_i||Q_i) will be zero for all i as P_i and Q_i are Gaussians with the same mean and variance.
    Thus, the \sum_i term is zero in the bound above, leaving only the mixture weights for P and Q. I.e., the upper
    bound for our setup is given by: D(P||Q) <= D(w^p||w^q).

    Hershey, John R., and Peder A. Olsen. "Approximating the Kullback Leibler divergence between Gaussian mixture
    models." 2007 IEEE International Conference on Acoustics, Speech and Signal Processing-ICASSP'07. Vol.4. IEEE, 2007.

    TODO:
        1) Complete unit testing for i)known values, ii)n_steps and iii) alphas.
        2) Write a new surprise_bins method in torch, allowing the surprise calculation for MoGs to be differentiable.

    :param p_weights: array of weights.
    :param q_weights: array of weights.
    :param score_type: desired score (str). Default = "KL_Q_P".
    :param fast: sacrifice some accuracy for speed (bool). Depends on method for bins that are empty for both p and q.
    :param alpha: smoothing 'pseudocount' (float).
    :return: array of scores.
    """
    p_weights = p_weights.detach().clone().cpu().numpy()
    q_weights = q_weights.detach().clone().cpu().numpy()

    # Get entropies using the weights only (upper bound for matched components, fixed means and centres)
    return surprise_bins(p_weights, q_weights, score_type, fast=fast, alpha=alpha)


def aggregate_parent_surprises(learner_surprises, learner, dev, norm_weights=True):
    """
    Aggregate parent surprises using the parent-->child weights.

    TODO:
        0) Test!
            i) Random and uniform weights
            ii) Scaling weights with / without norm

    :param learner_surprises: list of arrays, each containing the surprises for a layer of the learner.
    :param learner: learner itself (needed for clr_in dimensions, for now).
    :param norm_weights: bool, whether or not to normalize the weights s.t. they sum to 1.
    :return: array of surprises with shape=[n_parameters, 2], where 2 = (unit surprise, parent surprise).
    """
    # Setup -------------------------------------------------------
    stats_layer_idx = 0                                 # idx links stats_layers to corresponding learner module
    h, w = 0, 0                                         # kernel height and width for conv layers
    aggr_parent_surprises = [(torch.zeros_like(learner_surprises[0], device=dev))]  # zero for 1st layer

    # Go through all modules/layers in the learner ----------------
    for m in learner.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # Common actions for a stats layer --------------------
            if stats_layer_idx == 0:
                stats_layer_idx += 1
                continue
            weights = torch.abs(m._parameters['weight'].detach())   # detach from comp. graph to save compute+mem
            parent_surprises = learner_surprises[stats_layer_idx - 1]
            extra_axes = list(range(len(weights.shape)))[1:]        # [1,2,3] for conv, [1] for linear
            ones_like_extra_axes = [1] * len(extra_axes)            # [1,1,1] for conv, [1] for linear
            if norm_weights:
                weights /= torch.sum(weights, extra_axes).view(-1, *ones_like_extra_axes)

            # Conv layer ------------------------------------------
            if isinstance(m, nn.Conv2d):
                _, _, h, w = weights.shape
                parent_surprises = parent_surprises.view(1, -1, 1, 1).repeat(1, 1, h, w)
                weighted_parent_surprises = torch.sum(torch.mul(weights, parent_surprises), extra_axes)

            # Linear layer ----------------------------------------
            else:
                if h > 0:   # linear layer directly after conv layer
                    parent_surprises = parent_surprises.view(1, -1, 1, 1).repeat(1, 1, learner.clr_in, learner.clr_in)
                    h = 0
                parent_surprises = parent_surprises.view(1, -1)
                weighted_parent_surprises = torch.matmul(weights, parent_surprises.t()).view(-1)

            # Accumulate ------------------------------------------
            aggr_parent_surprises.append(weighted_parent_surprises)
            stats_layer_idx += 1

    return aggr_parent_surprises


def get_surprise_thresholds(surprise_score, n_layer_types=2):
    # Assume basic setup of conv + linear for now (easy to change)
    if surprise_score == "PSI":
        c_thresholds, p_thresholds = [0.0, 0.], [50.0001, 50.0001]
    elif "KL" in surprise_score:
        c_thresholds, p_thresholds = [0.0, 0.0], [50., 50.]
    elif surprise_score == "JS":
        c_thresholds, p_thresholds = [0.05, 0.25], [0.05, 0.01]
    elif surprise_score == "SI_Z":
        c_thresholds, p_thresholds = [0.1, 0.5], [0.1, 0.5]
    elif "SI" in surprise_score:
        c_thresholds, p_thresholds = [0.2, 0.2], [0.2, 0.2]
    else:
        raise ValueError("Invalid surprise score {0}".format(surprise_score))

    if n_layer_types > 2:                           # batchnorm
        c_thresholds = [0.1] + c_thresholds         # layer types are sorted alphabetically, so b < c < l
        p_thresholds = [0.1] + p_thresholds

    return c_thresholds, p_thresholds


def get_surprise_thresholds_bu_full(surprise_score, n_layer_types=2):
    # Assume basic setup of conv + linear for now (easy to change)
    if surprise_score == "CMD":
        c_thresholds, p_thresholds = [0.001, 0.001], [0.01, 0.01]
    elif surprise_score == "PSI":
        c_thresholds, p_thresholds = [0.2, 0.2], [0.5, 0.5]
    elif "KL" in surprise_score:
        c_thresholds, p_thresholds = [0.05, 0.05], [0.5, 0.5]
    elif surprise_score == "JS":
        c_thresholds, p_thresholds = [0.05, 0.25], [0.05, 0.01]
    elif surprise_score == "SI_Z":
        c_thresholds, p_thresholds = [0.1, 0.5], [0.1, 0.5]
    elif "SI" in surprise_score:
        c_thresholds, p_thresholds = [0.2, 0.2], [0.2, 0.2]
    else:
        raise ValueError("Invalid surprise score {0}".format(surprise_score))

    if n_layer_types > 2:                           # batchnorm
        c_thresholds = [0.1] + c_thresholds         # layer types are sorted alphabetically, so b < c < l
        p_thresholds = [0.1] + p_thresholds

    return c_thresholds, p_thresholds


def get_surprise_thresholds_bu(surprise_score, stats_layer_type="bins"):
    if stats_layer_type == "bins":
        if surprise_score == "PSI":
            return [0.0, 0.]
        elif "KL" in surprise_score:
            return [0.0, 0.]
        elif surprise_score == "JS":
            return [0.0, 0.]
        elif surprise_score == "SI_norm":
            return [0.0, 0.]
        elif surprise_score == "SI":
            return [0.0, 0.]
        else:
            raise ValueError("Invalid surprise score {0}".format(surprise_score))
    elif stats_layer_type == "gaussian":
        if surprise_score == "PSI":
            return [0.0, 0.]
        elif "KL" in surprise_score:
            return [0.0, 0.]
        elif surprise_score == "JS":
            return [0.0, 0.]
        elif surprise_score == "SI_norm":
            return [0.0, 0.]
        elif surprise_score == "SI":
            return [0.0, 0.]
        else:
            raise ValueError("Invalid surprise score {0}".format(surprise_score))
    elif stats_layer_type == "moments":
        return [0., 0.]
    else:
        raise ValueError("Invalid stats-layer type {0}".format(stats_layer_type))


def get_surprise_thresholds_meta(surprise_score):
    if surprise_score == "PSI":
        current_threshold, parent_threshold = 1., -0.5
    elif "KL" in surprise_score:
        current_threshold, parent_threshold = 2., -0.5
    elif surprise_score == "JS":
        current_threshold, parent_threshold = 0.25, 0.25
    elif surprise_score == "SI_Z":
        current_threshold, parent_threshold = 5., 5.
    elif "SI" in surprise_score:
        current_threshold, parent_threshold = 0.5, -0.5
    else:
        raise ValueError("Invalid surprise score {0}".format(surprise_score))

    return current_threshold, parent_threshold


def torch_wasserstein_loss(tensor_a, tensor_b):
    #Compute the first Wasserstein distance between two 1D distributions.
    return(torch_cdf_loss(tensor_a,tensor_b, p=1))


def torch_energy_loss(tensor_a,tensor_b):
    # Compute the energy distance between two 1D distributions.
    return((2**0.5)*torch_cdf_loss(tensor_a,tensor_b,p=2))


def torch_cdf_loss(tensor_a,tensor_b,p=1):
    # last-dimension is weight distribution
    # p is the norm of the distance, p=1 --> First Wasserstein Distance
    # to get a positive weight with our normalized distribution
    # we recommend combining this loss with other difference-based losses like L1

    # normalize distribution, add 1e-14 to divisor to avoid 0/0
    tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
    tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)
    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(tensor_a,dim=-1)
    cdf_tensor_b = torch.cumsum(tensor_b,dim=-1)

    # choose different formulas for different norm situations
    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_tensor_a-cdf_tensor_b)),dim=-1)
    elif p == 2:
        cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_tensor_a-cdf_tensor_b),2),dim=-1))
    else:
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_tensor_a-cdf_tensor_b),p),dim=-1),1/p)

    cdf_loss = cdf_distance.mean()
    return cdf_loss


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
