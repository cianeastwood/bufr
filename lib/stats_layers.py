from __future__ import division
from torch.nn.modules import Module
from collections import deque
import random
from lib.surprise import *
from lib.utils import bincount2D_vectorized, calc_z_scores, add_prefix_to_dict_keys
import torch.nn.functional as F

SEED = 404

def get_stats_layer_by_name(layer_name):
    if layer_name == "bins":
        return BinStats2D, BinStats1D
    elif layer_name == "gaussian":
        return GaussianStats2D, GaussianStats1D
    elif layer_name == "mogs":
        return MoGStats2D, MoGStats1D
    elif layer_name == "moments":
        return MomentStats2D, MomentStats1D
    elif layer_name == "soft_bins":
        return SoftBinStats2D, SoftBinStats1D
    else:
        raise NotImplementedError("Invalid stats layer choice: {0}".format(layer_name))


def select_inds(n_units, max_units, shuffle=False, rng=None):
    rng = np.random.RandomState(SEED) if rng is None else rng
    if n_units <= max_units:
        inds = np.array(list(range(n_units)))
        if shuffle:
            rng.shuffle(inds)
    else:
        inds = rng.choice(range(n_units), max_units, replace=False)
        if not shuffle:
            inds.sort()
    return inds


def nmoment(x, c, n, dim=0):
    return torch.mean((x-c)**n, dim=dim)


def soft_bin(x, bin_edges, temperature=0.1):
    """
    TODO:
        1) Test function
        2) Vectorized version possible?
        3) Pass in W and b, not edges (they are constant so should be stored

    :param x: N-by-1 matrix (column vector)
    :param bin_edges: montonically-increasing D-dim vector (D is the number of bin-edges)
    :param temperature:
    :return: N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    """

    D = bin_edges.shape[0]
    W = torch.reshape(torch.linspace(1, D + 1, D + 1), [1, -1])
    b = torch.cumsum(torch.cat([torch.zeros([1]), -bin_edges], 0), 0)
    z = torch.matmul(x, W) + b
    soft_assignments = torch.nn.Softmax(dim=0)(z / temperature)
    return soft_assignments


def softly_smooth_bins(bin_counts, bin_edges, W, b, tau=0.1):
    """
    "Soft" smoothing of bins means that some bins counts are dispersed into nearby bins, in contrast to uniform /
    add-one smoothing.
    :param bin_counts: array of bins counts of dimension [num_features, n_bins].
    :param bin_edges: array of bins edges of dimension [num_features, n_bins + 2].
    """
    # Get bins centres -- these will act as our "samples"
    bin_centres = (bin_edges[:, :-1] + bin_edges[:, 1:]) / 2.
    n_features = bin_centres.shape[0]

    # Reshape our "samples"
    bin_centres = bin_centres.reshape((n_features, -1, 1))  # reshape for make-shift batch outer prod via matmul

    # Calculate "logits" per sample via batch outer-product.
    # x:[n_features, n_samples, 1] x W:[n_features, 1, n_bins] = [n_features, n_samples, n_bins]
    z = torch.matmul(bin_centres, W) + b

    # Calculate soft allocations per sample ("soft" --> sum to 1)
    soft_counts = torch.nn.Softmax(dim=2)(z / tau)  # [n_features, n_samples, n_bins]

    # Reweight these soft allocations by bin counts
    bin_counts_ = bin_counts.unsqueeze(-1)
    soft_counts = soft_counts * bin_counts_

    # Sum over samples to get total soft counts ("soft" --> real number)
    soft_counts = soft_counts.sum(1)

    return soft_counts


class StatsLayer(Module):
    """
    Base module for all Stats layers.
    """

    def __init__(self, n_features, eps=1e-5, momentum=0.9, rng=None, track_stats=False, calc_surprise=False,
                 surprise_score="KL_Q_P", cpu=True):
        super(StatsLayer, self).__init__()
        self.n_features = n_features
        self.eps = eps
        self.momentum = momentum
        self.rng = np.random.RandomState(SEED) if rng is None else rng
        self.track_stats = track_stats
        self.calc_surprise = calc_surprise
        self.cpu = cpu
        self.register_buffer('surprise', torch.zeros(self.n_features))
        self.parent_scores = None
        self.surprise_score = surprise_score

    def get_stats(self):
        raise NotImplementedError

    def reset_stats(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def reset_rng(self):
        self.rng.seed(SEED)
        torch.manual_seed(SEED)
        random.seed(SEED)

    def extra_repr(self):
        return '{n_features}'.format(**self.__dict__)


class Layer1D(Module):
    """
    Base module for all 1D layers, e.g. FC layers.
    """
    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(x.dim()))


class Layer2D(Module):
    """
    Base module for all 2D layers, e.g. conv layers.
    """
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))


class GaussianStats(StatsLayer):
    """ Module to track the unit activation distributions of a layer with a Gaussian/Normal distribution. """

    def __init__(self, n_features, eps=1e-5, momentum=0.1, rng=None, track_stats=False,
                 calc_surprise=False, surprise_score="KL_Q_P", use_direct_z_score=False, cpu=True):
        super(GaussianStats, self).__init__(n_features, eps, momentum, rng, track_stats, calc_surprise, surprise_score,
                                            cpu)
        # What to track: buffers = savable params with no grads. Q: do we need grad for running stats?
        self.register_buffer('running_mean', torch.zeros(self.n_features))
        self.register_buffer('running_var', torch.ones(self.n_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.use_direct_z_score = use_direct_z_score

    def get_stats(self):
        return {"mean": self.running_mean.detach().numpy(), "var": self.running_var.detach().numpy()}

    def reset_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def calc_batch_stats(self, x):
        raise NotImplementedError

    def update_stats(self, x):
        self.num_batches_tracked += 1
        exponential_average_factor = self._exp_avg_factor()
        batch_mean, batch_var = self.calc_batch_stats(x)

        self.running_mean = batch_mean * exponential_average_factor + \
                            self.running_mean * (1. - exponential_average_factor)
        self.running_var = batch_var * exponential_average_factor + \
                           self.running_var * (1. - exponential_average_factor)

    def update_surprise(self, x):
        if self.use_direct_z_score:  # use direct z-score instead of calibrated surprised score
            zs = calc_z_scores(x, self.running_mean, self.running_var)
            self.surprise = torch.mean(zs, 0)
        else:
            q_means, q_vars = self.calc_batch_stats(x)
            self.surprise = surprise_gaussian(self.running_mean, self.running_var,
                                              q_means, q_vars, x, self.surprise_score)

    def forward(self, x):
        self._check_input_dim_if_implem(x)
        if self.calc_surprise:
            self.update_surprise(x)

        if self.track_stats:
            self.update_stats(x)

        return x

    def _check_input_dim_if_implem(self, x):
        _check_input_dim = getattr(self, "_check_input_dim", None)
        if callable(_check_input_dim):   # i.e. object instance has a method called _check_input_dim
            _check_input_dim(x)

    def _exp_avg_factor(self):
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
        return exponential_average_factor


class GaussianStats1D(GaussianStats, Layer1D):
    def calc_batch_stats(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        return batch_mean, batch_var


class GaussianStats2D(GaussianStats, Layer2D):
    """
    Within-channel "spatial samples" can be dealt with in a number of ways.
        -Tracking: default = treat these spatial samples as normal samples, i.e. flatten to B*H*W. Other
                 options include tracking each location separately.
        -Surprise: default = mean surprise across spatial locations, using a common distribution. Other
                 options include counting the number of locations surprised (i.e. thresholding
                 based on quality/mean or quantity/counts).

    TODO: track_per_loc flag
    """
    def calc_batch_stats(self, x):
        batch_mean = torch.mean(x, dim=(0, 2, 3))
        batch_var = torch.var(x, dim=(0, 2, 3))
        return batch_mean, batch_var

    def update_surprise(self, x):
        # Conv: [B, C, H, W] --> [BxHxW, C]
        # x = x.transpose(0, 1).flatten(1).transpose(0, 1)
        super(GaussianStats2D, self).update_surprise(x)


class MomentStats(StatsLayer):
    def __init__(self, n_features, eps=1e-5, momentum=0.1, rng=None, track_stats=False, track_range=False,
                 calc_surprise=False, surprise_score="KL_Q_P", n_moments=4, cpu=True, norm_range=True):
        super(MomentStats, self).__init__(n_features, eps, momentum, rng, track_stats, calc_surprise, surprise_score,
                                          cpu)
        assert n_moments >= 2
        self.n_moments = n_moments
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('m1', torch.zeros(self.n_features))
        for n in range(2, n_moments + 1):
            self.register_buffer('m{0}'.format(n), torch.ones(self.n_features))

        self.register_buffer('mins', 1e6*torch.ones(self.n_features))
        self.register_buffer('maxs', -1e6*torch.ones(self.n_features))
        self.register_buffer('feature_ranges', torch.ones(self.n_features, 1))
        self.track_range = track_range
        self.norm_range = norm_range
        self.feature_ranges = None

    def get_stats(self):
        moments = [getattr(self, 'm{0}'.format(n)).detach().cpu().numpy() for n in range(1, self.n_moments + 1)]
        return {"moments": moments, "m_mins": self.mins.detach().cpu().numpy(),
                "m_maxs": self.maxs.detach().cpu().numpy()}

    def reset_stats(self):
        self.num_batches_tracked.zero_()
        self.m1.zero_()
        for n in range(2, self.n_moments + 1):
            getattr(self, 'm{0}'.format(n)).fill_(1)

    def reset_range(self):
        self.min.fill_(1e6)
        self.max.fill_(-1e6)

    def update_range(self, x):
        """
        Update activation range, if inputs fall outside current range. Flatten conv "spatial samples" by default.
        """
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        x = x.transpose(0, 1).flatten(1).detach()

        # Get min over dim=1, i.e. over samples, one per feature/channel
        mns, mxs = x.min(1)[0], x.max(1)[0]
        self.mins = torch.where(self.mins < mns, self.mins, mns).detach()
        self.maxs = torch.where(self.maxs > mxs, self.maxs, mxs).detach()

    def calc_batch_stats(self, x):
        raise NotImplementedError

    def _norm_inputs(self, x):
        """ x.shape = [C, -1]"""
        if self.norm_range and not self.track_range:
            return x / self.feature_ranges
        return x

    def update_stats(self, x):
        if self.feature_ranges is None and self.norm_range:
            feature_ranges = self.maxs - self.mins
            self.feature_ranges = feature_ranges.unsqueeze(1)

        self.num_batches_tracked += 1
        exponential_average_factor = self._exp_avg_factor()
        batch_moments = self.calc_batch_stats(x)

        for n, b_moment in enumerate(batch_moments, 1):
            running_moment = getattr(self, 'm{0}'.format(n))
            running_moment = b_moment * exponential_average_factor + running_moment * (1. - exponential_average_factor)
            setattr(self, 'm{0}'.format(n), running_moment)

    def update_surprise(self, x):
        q_moments = self.calc_batch_stats(x)
        p_moments = []
        for n in range(1, self.n_moments + 1):
            p_moments.append(getattr(self, 'm{0}'.format(n)))
        self.surprise = surprise_moments(p_moments, q_moments, self.mins, self.maxs, self.norm_range)

    def forward(self, x):
        self._check_input_dim_if_implem(x)
        if self.track_range:
            # Update mins and maxs
            self.update_range(x)

        elif self.track_stats:
            self.update_stats(x)

        if self.calc_surprise:
            self.update_surprise(x)

        return x

    def _check_input_dim_if_implem(self, x):
        _check_input_dim = getattr(self, "_check_input_dim", None)
        if callable(_check_input_dim):   # i.e. object instance has a method called _check_input_dim
            _check_input_dim(x)

    def _exp_avg_factor(self):
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
        return exponential_average_factor

    def _load_from_state_dict(self, *args):
        super(MomentStats, self)._load_from_state_dict(*args)
        if self.maxs[0] >= self.mins[0]:  # valid range has been loaded
            self.track_range = False

    def extra_repr(self):
        return '{n_features}, k={n_moments}'.format(**self.__dict__)


class MomentStats1D(MomentStats, Layer1D):
    def calc_batch_stats(self, x):
        if self.norm_range:
            x = x.transpose()
            x = self._norm_inputs(x)
            x = x.transpose()
        m1 = torch.mean(x, dim=0)
        moments = [m1]
        for n in range(2, self.n_moments + 1):
            moments.append(nmoment(x, m1, n, dim=0))
        return moments


class MomentStats2D(MomentStats, Layer2D):
    """
    Within-channel "spatial samples" can be dealt with in a number of ways.
        -Tracking: default = treat these spatial samples as normal samples, i.e. flatten to B*H*W. Other
                 options include tracking each location separately.
        -Surprise: default = mean surprise across spatial locations, using a common distribution. Other
                 options include counting the number of locations surprised (i.e. thresholding
                 based on quality/mean or quantity/counts).

    TODO: track_per_loc flag
    """
    def calc_batch_stats(self, x):
        raise NotImplementedError("Need to norm 2d inputs if want to use this")
        m1 = torch.mean(x, dim=(0, 2, 3))
        moments = [m1]
        for n in range(2, self.n_moments + 1):
            moments.append(nmoment(x, m1.view(-1, 1, 1), n, dim=(0, 2, 3)))
        return moments


class BinStats(StatsLayer):
    """
        Module to track the unit activation distributions of a layer with bins.
    """
    def __init__(self, n_features, eps=1e-5, momentum=0.9, rng=None, track_stats=False,
                 calc_surprise=False, track_range=False, n_bins=8, surprise_score="PSI", cpu=True, norm_range=True):
        super(BinStats, self).__init__(n_features, eps, momentum, rng, track_stats, calc_surprise, surprise_score, cpu)
        #  What to track : buffers (savable params) -- used for bins later
        self.register_buffer('mins', 1e6*torch.ones(self.n_features))
        self.register_buffer('maxs', -1e6*torch.ones(self.n_features))
        self.track_range = track_range
        self.n_bins = n_bins
        self.norm_range = norm_range
        self.n_bin_edges = n_bins + 1
        self.register_buffer('bin_edges', torch.zeros([self.n_features, self.n_bin_edges + 2]))  # 2 end bins
        self.register_buffer('bin_counts', torch.zeros([self.n_features, self.n_bins + 2]))   # 2 end bins
        self.register_buffer('feature_ranges', torch.ones(self.n_features, 1))

    def reset_stats(self):
        self.bin_counts.zero_()

    def reset_range(self):
        self.min.fill_(1e6)
        self.max.fill_(-1e6)
        self.bin_edges.zero_()

    def get_stats(self):
        return {"edges": self.bin_edges.detach().cpu().numpy(), "counts": self.bin_counts.detach().cpu().numpy(),
                "mins": self.mins.detach().cpu().numpy(), "maxs": self.maxs.detach().cpu().numpy()}

    def update_range(self, x):
        """
        Update activation range, if inputs fall outside current range. Flatten conv "spatial samples" by default.
        """
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        x = x.transpose(0, 1).flatten(1).detach()

        # Get min over dim=1, i.e. over samples, one per feature/channel
        mns, mxs = x.min(1)[0], x.max(1)[0]
        self.mins = torch.where(self.mins < mns, self.mins, mns).detach()
        self.maxs = torch.where(self.maxs > mxs, self.maxs, mxs).detach()

    def init_bins(self):
        """
        Initialise bins with running range [min, max].
        """
        self.maxs += EPS  # add small value to edge to incl act in rightmost bin

        #  Set bin edges
        mns, mxs = self.mins, self.maxs
        if self.norm_range:
            feature_ranges = mxs - mns
            mxs = mxs / feature_ranges
            mns = mns / feature_ranges
            self.feature_ranges = feature_ranges.unsqueeze(1)

        b_edges = [torch.linspace(st, sp, self.n_bin_edges, device=self.maxs.device) for st, sp in list(zip(mns, mxs))]
        b_edges = torch.stack(b_edges, 0)
        # b_edges = torch.stack(b_edges, 0).to(torch.device('cuda'))   # [num_features, n_bin_edges]
        is_relu = torch.allclose(mns, torch.zeros_like(mns))

        #  Set width of additional end bins (i.e. range extensions)
        exts = 0.25 * (mxs - mns)                           # end bin widths = 0.25 * range
        r_exts = (mxs + exts).reshape(-1, 1)                # right
        if is_relu:                                         # relu unit, larger left end bin width required
            l_exts = (mns - 2 * exts).reshape(-1, 1)
        else:
            l_exts = (mns - exts).reshape(-1, 1)

        self.bin_edges = torch.cat([l_exts, b_edges, r_exts], 1).detach()  # [num_features, n_bin_edges + 2]

    def _get_batch_counts(self, x):
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        with torch.no_grad():
            x = x.transpose(0, 1).flatten(1)
            x = self._norm_inputs(x)
            bin_indices = torch.searchsorted(self.bin_edges[:, 1:-1], x)
            batch_counts = bincount2D_vectorized(bin_indices, minlength=(self.n_bins + 2))
        return batch_counts.detach()

    def _norm_inputs(self, x):
        """ x.shape = [C, -1]"""
        if self.norm_range and not self.track_range:
            return x / self.feature_ranges
        return x

    def update_bins(self, x):
        """
        Update bin counts for new inputs.
        :param x: array of inputs of dim (b,c,h,w) or (b,c).
        """
        if (self.bin_counts == 0.).all():    # counts all zero
            self.init_bins()

        batch_counts = self._get_batch_counts(x)
        self.bin_counts += batch_counts
        # self.bin_counts = self.bin_counts * self.momentum + batch_counts * (1. - self.momentum)

    def update_surprise(self, x):
        p_counts = self.bin_counts.detach().clone().cpu().numpy()
        batch_counts = self._get_batch_counts(x).detach().float().cpu().numpy()   # q_counts
        scores = surprise_bins(p_counts, batch_counts, score_type=self.surprise_score, fast=True)
        self.surprise = torch.from_numpy(scores).float().to(x.device)

    def forward(self, x):
        self._check_input_dim_if_implem(x)
        tracked_x = x.clone()                       # branch off the in comp. graph

        if self.track_range:
            # Update mins and maxs
            self.update_range(tracked_x)

        elif self.track_stats:
            # Update bin counts
            self.update_bins(tracked_x)

        if self.calc_surprise:
            # Update surprise scores
            self.update_surprise(tracked_x)
        return x

    def _check_input_dim_if_implem(self, x):
        _check_input_dim = getattr(self, "_check_input_dim", None)
        if callable(_check_input_dim):   # i.e. object instance has a method called _check_input_dim
            _check_input_dim(x)

    def _load_from_state_dict(self, *args):
        super(BinStats, self)._load_from_state_dict(*args)
        if self.maxs[0] >= self.mins[0]:  # valid range has been loaded
            self.track_range = False


class BinStats1D(BinStats, Layer1D):
    pass


class BinStats2D(BinStats, Layer2D):
    """
    Within-channel "spatial samples" can be dealt with in a number of ways.
        -Tracking: default = treat these spatial samples as normal samples, i.e. flatten to B*H*W. Other
                 options include tracking each location separately.
        -Surprise: default = mean surprise across spatial locations, using a common distribution. Other
                 options include counting the number of locations surprised (i.e. thresholding
                 based on quality/mean or quantity/counts).

    TODO: track_per_loc flag
    """
    pass


class SoftBinStats(BinStats):
    """
    Module to track the unit activation distributions of a layer with a mixture of Gaussians.
    Treated as a differentiable analog of BinStats layer, where flags like track_bins refer to tracking
    mixture weights/coefficients.
    """
    def __init__(self, n_features,  eps=1e-5, momentum=0.1, rng=None, track_stats=False,
                 calc_surprise=False, track_range=False, n_bins=8, surprise_score="KL_Q_P", tau=0.01, cpu=True,
                 norm_range=True):
        super(SoftBinStats, self).__init__(n_features, eps, momentum, rng, track_stats, calc_surprise,
                                           track_range, n_bins, surprise_score, cpu, norm_range)
        self.tau = tau
        self.register_buffer('W', torch.zeros((self.n_features, 1, self.n_bin_edges + 1)))
        self.register_buffer('b', torch.zeros((self.n_features, 1, self.n_bin_edges + 1)))
        self._test_time = False

    @property
    def test_time(self):
        return self._test_time

    @test_time.setter           # clears up some memory
    def test_time(self, value):
        if not isinstance(value, bool):
            raise ValueError('test_time must be a boolean value')
        self._test_time = value
        if value:
            self.mins, self.maxs, self.bin_edges = None, None, None
            self.bin_counts = self.bin_counts.detach().clone()

    def init_bins(self):
        """
        TODO: Vectorize.
        See sec. 3.1 in https://arxiv.org/abs/1806.06988 for details on the soft binning procedure.
        """
        # Init bins using super class
        super(SoftBinStats, self).init_bins()

        # Define W + b for soft binning
        inner_edges = self.bin_edges[:, 1:-1]           # soft binning method using n "inner" edges for n+1 intervals
        n_edges = inner_edges.shape[1]
        dev = self.bin_edges.device
        W = []
        b = []
        for f in range(self.n_features):
            w_ = torch.reshape(torch.linspace(1, n_edges + 1, n_edges + 1, device=dev), [1, -1])
            b_ = torch.cumsum(torch.cat([torch.zeros([1], device=dev), -inner_edges[f]], 0), 0)
            W.append(w_)
            b.append(b_)
        self.W = torch.vstack(W).reshape((self.n_features, 1, n_edges + 1))  # reshape for matmul later
        self.b = torch.vstack(b).reshape((self.n_features, 1, n_edges + 1))  # reshape for matmul later

    def _get_batch_counts(self, x):
        """ See Neural Decision Forests paper for details on the soft binning procedure. """
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        x = x.transpose(0, 1).flatten(1)
        x = self._norm_inputs(x)
        x = x.reshape((self.n_features, -1, 1))         # reshape for make-shift batch outer prod via matmul

        # Calculate "logits" per sample via batch outer-product.
        # x:[n_features, n_samples, 1] x W:[n_features, 1, n_bins] = [n_features, n_samples, n_bins]
        z = torch.matmul(x, self.W) + self.b

        # Calculate soft allocations per sample ("soft" --> sum to 1)
        sft_cs = torch.nn.Softmax(dim=2)(z / self.tau)  # [n_features, n_samples, n_bins]

        # Sum over samples to get total soft counts ("soft" --> real number)
        total_sft_cs = sft_cs.sum(1)

        return total_sft_cs

    def _smooth_batch_counts(self):
        self.bin_counts = softly_smooth_bins(self.bin_counts, self.bin_edges, self.W, self.b, 0.1)

    def _test_tau_smoothing(self, x):
        fine_tau = 0.001
        coarse_tau = 0.1

        self.tau = fine_tau
        batch_counts_fine = self._get_batch_counts(x)

        self.tau = coarse_tau
        batch_counts_coarse = self._get_batch_counts(x)

        batch_counts_smoothed = softly_smooth_bins(batch_counts_fine, self.bin_edges, self.W, self.b, coarse_tau)

        print(batch_counts_coarse - batch_counts_smoothed)

    def update_surprise(self, x):
        p_counts = self.bin_counts.clone()
        q_counts = self._get_batch_counts(x)
        # self._test_tau_smoothing(x)
        # smooth_p = softly_smooth_bins(self.bin_counts, self.bin_edges, self.W, self.b, 1.)
        # smooth_q = softly_smooth_bins(batch_counts, self.bin_edges, self.W, self.b, 1.)

        self.surprise = surprise_soft_bins(p_counts, q_counts, score_type=self.surprise_score)
        # soft_smoother = 0.01
        # self.tau = soft_smoother
        # batch_counts = self._get_batch_counts(x)
        # smooth_p = softly_smooth_bins(self.bin_counts, self.bin_edges, self.W, self.b, soft_smoother)
        # smooth_q = softly_smooth_bins(batch_counts, self.bin_edges, self.W, self.b, soft_smoother)
        # self.surprise = surprise_soft_bins(smooth_p, smooth_q, score_type=self.surprise_score)
        # self.surprise = surprise_soft_bins(smooth_p, batch_counts, score_type=self.surprise_score)


    def get_stats(self):
        stats = super(SoftBinStats, self).get_stats()
        stats = add_prefix_to_dict_keys(stats, "soft_")         # unique key names vs. BinStats layers
        stats["tau"] = np.array([self.tau])
        return stats

    def extra_repr(self):
        return '{n_features}, tau={tau}, ss={surprise_score}'.format(**self.__dict__)


class SoftBinStats1D(SoftBinStats, Layer1D):
    pass


class SoftBinStats2D(SoftBinStats, Layer2D):
    """
    Within-channel "spatial samples" can be dealt with in a number of ways.
        -Tracking: default = treat these spatial samples as normal samples, i.e. flatten to B*H*W. Other
                 options include tracking each location separately.
        -Surprise: default = mean surprise across spatial locations, using a common distribution. Other
                 options include counting the number of locations surprised (i.e. thresholding
                 based on quality/mean or quantity/counts).
    TODO: track_per_loc flag
    """
    pass


class SoftBinStatsSoftmax(SoftBinStats1D):
    def forward(self, x):
        soft_preds = F.softmax(x, dim=1)
        return super(SoftBinStatsSoftmax, self).forward(soft_preds)

    # def update_surprise(self, x):
    #     # Gather p and q counts
    #     # p_counts = self.bin_counts.clone()
    #     # p_total_counts = 50000                                                  # hardcoded for now
    #     q_counts = self._get_batch_counts(x)
    #     # q_total_counts = x.shape[0]                                             # batch size
    #     #
    #     # # Aggregate overflow end-bin counts into the first and last bins
    #     # q_centre_counts = q_counts[:, 1:-1]
    #     # q_centre_counts[:, 0] += q_counts[:, 0]
    #     # q_centre_counts[:, -1] += q_counts[:, -1]
    #     #
    #     # p_centre_counts = p_counts[:, 1:-1]
    #     # p_centre_counts[:, 0] += p_counts[:, 0]
    #     # p_centre_counts[:, -1] += p_counts[:, -1]
    #     #
    #     # # Get centre (i.e. non-extended) bin centres
    #     # non_extended_edges = self.bin_edges[:, 1:-1]
    #     # non_extended_centres = (non_extended_edges[:, :-1] + non_extended_edges[:, 1:]) / 2.
    #     #
    #     # # Overwrite first and last centres with values 0 and 1 respectively
    #     # non_extended_centres[:, 0] = torch.zeros_like(non_extended_centres[:, 0])
    #     # non_extended_centres[:, -1] = torch.ones_like(non_extended_centres[:, 1])
    #     #
    #     # # Calculate total counts per class
    #     # q_weighted_counts = non_extended_centres * q_centre_counts
    #     # p_weighted_counts = non_extended_centres * p_centre_counts
    #     #
    #     # # Test calculations were correct
    #     # # total_q_counts = q_weighted_counts.sum()
    #     # # total_p_counts = p_weighted_counts.sum()
    #     # # print(total_q_counts)
    #     # # print(total_p_counts)
    #     #
    #     # # Calculate proportions per class, i.e. prior probabilities / class balances
    #     # q_label_prob = q_weighted_counts.sum(1) / q_total_counts
    #     # p_label_prob = p_weighted_counts.sum(1) / p_total_counts
    #     # label_shifts = (p_label_prob - q_label_prob)
    #
    #     self.surprise = surprise_soft_bins(self.bin_counts.clone(), q_counts, score_type=self.surprise_score)
    #     # self.surprise = self.surprise + label_shifts


class MoGStats(BinStats):
    """
    Module to track the unit activation distributions of a layer with a mixture of Gaussians.
    Treated as a differentiable analog of BinStats layer, where flags like track_bins refer to tracking
    mixture weights/coefficients.
    """
    def __init__(self, n_features,  eps=1e-5, momentum=0.9, rng=None, track_stats=False,
                 calc_surprise=False, track_range=False, n_bins=4, surprise_score="KL_Q_P", cpu=True,
                 n_update_steps=1):
        super(MoGStats, self).__init__(n_features, eps, momentum, rng, track_stats, calc_surprise,
                                       track_range, n_bins, surprise_score, cpu)
        self.n_cs = self.n_bins
        self.n_cs_ext = self.n_cs + 2    # 2 end bins
        self.n_update_steps = n_update_steps
        #  Init uniform weights on the mixture components (sum to 1)
        self.register_buffer('weights', torch.ones([self.n_features, self.n_cs_ext]) / float(self.n_cs_ext))
        self.register_buffer('cs', torch.zeros([self.n_features, self.n_cs_ext]))        # centres/means
        self.register_buffer('stds', torch.ones([self.n_features, self.n_cs_ext]))      # stds
        #  Init widths (hs), i.e. distances between consecutive centres (may not be constant)
        self.register_buffer('hs', torch.zeros([self.n_features, self.n_cs_ext]))

    def reset_stats(self):
        self.weights.fill_(1./float(self.n_cs_ext))

    def reset_range(self):
        self.mins.fill_(1e6)
        self.maxs.fill_(-1e6)
        self.cs.zero_()
        self.stds.fill_(1)
        self.hs.zero_()

    def get_stats(self):
        """ Underscore in mins_, maxs_ only to make attr names differ from BinStats class. """
        return {"weights": self.weights.detach().numpy(), "centres": self.cs.detach().numpy(),
                "stds": self.stds.detach().numpy(), "hs": self.hs.detach().numpy(), "mins_": self.mins.numpy(),
                "maxs_": self.maxs.numpy()}

    def init_bins(self):
        """
        Read bins as weights, as init_bins() is called by superclass BinStats when the tracking of bins
        is turned on, i.e. when we set track_bins=True.

        Ignore end bins for now -- asymmetric distributions needed, or just far-away centre?

        TODO:
            1) numpy --> torch in order to use GPU
                i) See init_bins of BinStats
                ii) np.tile --> torch.repeat_interleave(x.view(1, ...), self.n_cs) -- something like this
        """
        # get ranges
        mns, mxs = self.mins.cpu().detach().numpy(), self.maxs.cpu().detach().numpy()
        mxs += EPS  # add small value to edge to incl act in rightmost bin

        # set centres of dists ("bins")
        if float(np.__version__.strip()[:4]) >= 1.16:
            cs = np.linspace(mns, mxs, self.n_cs).T  # [num_features, n_cs]
        else:
            cs = np.array([np.linspace(st, sp, self.n_cs) for st, sp in zip(mns, mxs)])  # [num_features, n_cs]
        hs = (mxs - mns) / self.n_cs  # distance between distribution centres, i.e. "bin widths"
        hs = np.tile(hs, (self.n_cs, 1)).T  # [num_features, n_cs] -- repeat (tile) feature width for each dist/bin

        #  Set h_ends ("bin widths"): distance of the end dist (bin) from the rest.
        h_l_ends = 0.25 * (mxs - mns)  # 0.25 * range
        h_r_ends = 0.25 * (mxs - mns)  # 0.25 * range

        #  Set cs_l_end and cs_r_end ("bin centres"): centres for the left and right end distributions.
        cs_l_end = (mns - h_l_ends).reshape(-1, 1)  # left end
        cs_r_end = (mxs + h_r_ends).reshape(-1, 1)  # right end

        #  Stack distribution centres. Shape: [num_features, n_cs + 2]. Constant increases, except for ends.
        # e.g. [0.1, 0.5, 0.6, 0.7, 1.1]
        self.cs = torch.from_numpy(np.hstack([cs_l_end, cs, cs_r_end]))

        #  Stack distribution "widths". Shape: [num_features, n_cs + 2]. Constant values, except for ends.
        #  e.g. [0.4, 0.1, 0.1, 0.1, 0.4]
        self.hs = torch.from_numpy(np.hstack([h_l_ends.reshape(-1, 1), hs, h_r_ends.reshape(-1, 1)]))

        #  Set distribution stddevs
        self.stds = self.hs / 3.

    def _update_weights(self, x, weights):
        # Conv: [B, C, H, W] --> [C, BxHxW], FC: [B, C] --> [C, B]
        x = x.transpose(0, 1).flatten(1)
        new_weights = torch.ones_like(weights)
        for i, unit_samples in enumerate(x):
            new_weights[i] = update_mixture_weights(unit_samples, weights[i], self.cs[i], self.stds[i])
        return new_weights

    def update_bins(self, x):
        """
        Read bins as weights, as update_bins() is called by superclass BinStats in forward() method.
        Update mixture weights.
        """
        if (self.cs == 0.).all():  # centres all zero (yet to be initialised from tracked range)
            self.init_bins()
        self.weights = self._update_weights(x, self.weights)

    def update_surprise(self, x):
        """
        TODO:
            1) Rewrite bin_variance function in torch to allow gradients and GPU usage.
            2) Think about the number of updates that should be perform to get posterior weights q --> just 1?
        """
        # Form Q from a single batch, starting from uniform weights -- sensible? Start from p? How many steps?
        q_weights = torch.ones_like(self.weights) / float(self.weights.size()[1])
        for _ in range(self.n_update_steps):
            q_weights = self._update_weights(x, q_weights)

        self.surprise = surprise_mogs(self.weights, q_weights, self.surprise_score, fast=True)


class MoGStats1D(MoGStats, Layer1D):
    pass


class MoGStats2D(MoGStats, Layer2D):
    """
    Within-channel "spatial samples" can be dealt with in a number of ways.
        -Tracking: default = treat these spatial samples as normal samples, i.e. flatten to B*H*W. Other
                 options include tracking each location separately.
        -Surprise: default = mean surprise across spatial locations, using a common distribution. Other
                 options include counting the number of locations surprised (i.e. thresholding
                 based on quality/mean or quantity/counts).

    TODO: track_per_loc flag
    """
    pass


class SamplesStats(StatsLayer):
    """
       Module to track the unit activation distributions of a layer.
       """

    def __init__(self, n_features, eps=1e-5, momentum=0.9, rng=None, track_stats=False,
                 max_samples=200, max_features=8):
        super(SamplesStats, self).__init__(n_features, eps, momentum, rng, track_stats)
        self.max_samples = max_samples      # max samples per unit to store (save memory)
        self.max_features = max_features    # max num features/units/channels for which to store samples
        self.n_units_to_track = min(self.n_features, self.max_features)
        # indices of features for which to store samples
        inds = torch.as_tensor(select_inds(self.n_features, self.n_units_to_track, rng=self.rng))
        self.register_buffer('inds', inds)
        self.register_buffer('samples', torch.zeros([self.n_units_to_track, self.max_samples]))

    def init_samples(self):
        """
        Choose indices of units to track -- saves memory.
        """
        self._samples = [deque(maxlen=self.max_samples) for _ in range(self.n_units_to_track)]

    def update_samples(self, x):
        """
        Update stored samples (for selected units) in a sliding-window fashion (via FIFO deque).
        """
        if (self.samples == 0.).all():
            self.init_samples()

        selected_samples = x[:, self.inds].T
        for i, selected_unit_samples in enumerate(selected_samples):
            self._samples[i].extend(selected_unit_samples.detach().numpy())

        #  Ugly "double storage" (list, tensor) and per-iteration conversion to allow samples to be stored in buffer
        #  Could 1) Only convert before saving model (need to register hook) or 2) don't store samples as buffers!
        self.samples = torch.as_tensor(self._samples)

    def get_stats(self):
        return {"samples": self.samples.numpy(), "indices": self.inds.numpy()}

    def reset_stats(self):
        self.samples.zero_()
        self.init_samples()

    def forward(self, x):
        self._check_input_dim_if_implem(x)
        if self.track_stats:
            self.update_samples(x)
        return x

    def _check_input_dim_if_implem(self, x):
        _check_input_dim = getattr(self, "_check_input_dim", None)
        if callable(_check_input_dim):   # i.e. object instance has a method called _check_input_dim
            _check_input_dim(x)


class SamplesStats1D(SamplesStats, Layer1D):
    pass


class SamplesStats2D(SamplesStats, Layer2D):
    """
    Within-channel "spatial samples" can be dealt with in a number of ways.
      -Tracking: default = treat these spatial samples as normal samples, i.e. flatten to B*H*W. Other
                 options include tracking each location separately.
    TODO: track_per_loc flag
    """
    def __init__(self, n_features, eps=1e-5, momentum=0.9, rng=None, track_stats=False,
                 max_samples=200, max_features=8, track_per_loc=False, max_spatial_samples=None):
        super(SamplesStats2D, self).__init__(n_features, eps, momentum, rng, track_stats, max_samples, max_features)
        self.track_per_loc = track_per_loc
        if max_spatial_samples is None:                   # max "within-channel" or spatial samples to store per sample
            self.max_spatial_samples = max_samples // 20  # We want samples to reflect the last 20 (real) samples
        else:
            self.max_spatial_samples = max_spatial_samples

    def update_samples(self, x):
        """
        Update stored samples (for selected units) in a sliding-window fashion (via FIFO deque).
        """
        if (self.samples == 0.).all():
            self.init_samples()

        selected_feature_samples = x[:, self.inds].transpose(0, 1)                              # [B,C,H,W]-->[C,B,H,W]

        # Collapse spatial "samples" if not tracking per-loc distributions
        if not self.track_per_loc:
            selected_feature_samples = selected_feature_samples.flatten(2)                      # [C,B,H,W]-->[C,B,H*W]
            _, _, h, w = list(x.size())
            spatial_inds = select_inds(h*w, self.max_spatial_samples, rng=self.rng)
            selected_feature_samples = selected_feature_samples[:, :, spatial_inds].flatten(1)  # [C,B,H*W]-->[C,B*S]

        for i, selected_unit_samples in enumerate(selected_feature_samples):
            self._samples[i].extend(selected_unit_samples.detach().numpy())

        self.samples = torch.as_tensor(self._samples)  # ugly double storage for persistent buffers
