import os
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import random
from collections import Counter
from collections.abc import Iterable   # import directly from collections for Python < 3.3
import numpy as np
from typing import Union
import networkx as nx
MAX_MOMENT_DISTANCE = 1e5
NO_CHOICE_CONCEPT = 1e5


# List operation
def deduplicate(arr: list):
    """
    :param arr: the original list
    :return: deduplicated list
    """
    return list(set(arr))


def is_overlap(a, b):
    """
    :param a: list to compare
    :param b: list to compare
    :return: True if there are common element in a and b
    """
    a, b = deduplicate(a), deduplicate(b)
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    for e in shorter:
        if e in longer:
            return True
    return False


def subtract(tbs, ts):
    """
    :param tbs: list to be subtracted
    :param ts: list to subtract
    :return: subtracted list a
    """
    ts = deduplicate(ts)
    return [e for e in tbs if e not in ts]


def expand_window(data, window_set=1):
    """
    :param data: the original list
    :param window_set: int or list of int
    :return: a list of lists shifted by the bias which is specified by window_set
    """
    if isinstance(window_set, int):
        window_set = [window_set]
    window_list = []
    for bias in window_set:
        if bias > 0:
            tmp = [data[0]] * bias + data[:-bias]
        elif bias < 0:
            tmp = data[-bias:] + [data[-1]] * -bias
        else:
            tmp = data
        window_list.append(tmp)
    return window_list


def mean(data: list):
    """
    :param data: a list of int
    :return: the average of integers in data
    """
    res = sum(data) / len(data)
    return res


def flatten(list_of_lists):
    """
    :param list_of_lists: a list of sub-lists like "[[e_{00}, ...], ..., [e_{n0}, ...]]"
    :return: flatten all sub-lists into single list
    """
    return [item for sublist in list_of_lists for item in sublist]


def padding(seq: list, max_length: int, pad_tok=None):
    """
    :param seq: list to pad
    :param max_length: length of padded list
    :param pad_tok: token used to pad
    :return: padded list
    """
    return (seq + [pad_tok] * max_length)[:max_length]


# Numpy operation
def cosine_sim_np(fea: np.ndarray):
    """
    :param fea: feature vector [N, D]
    :return: score: cosine similarity score [N, N]
    """
    vec_norm = np.linalg.norm(fea, axis=-1, keepdims=True)
    vec_norm = np.where(vec_norm != 0, vec_norm, 1)  # avoid that divided by zero
    fea = fea / vec_norm
    score = fea.dot(fea.T)
    return score


def pairwise_equation(data: np.ndarray, tok_illegal=None):
    """
    :param data: the original data array
    :param tok_illegal: the tokens which is meaningless
    :return: an indicator matrix A where A_{i, j} == 1 denotes data[i] == data[j]
    """
    placeholder = "1&*^%!2)!"  # an impossible string
    str_mat = data.reshape(len(data), 1).repeat(len(data), axis=1)
    if tok_illegal is not None:
        data[data == tok_illegal] = placeholder
    mat = str_mat == data
    indicator_matrix = np.tril(mat, -1).astype(int)
    return indicator_matrix


def indicator_vec(indices: Union[int, list], n: int, device=None, dtype=torch.float):
    """
    :param indices: indices which is one
    :param n: number of classes
    :param device:
    :param dtype:
    :return: an indicator vector
    """
    vec = torch.zeros(n, dtype=dtype)
    vec[indices] = 1
    if device is not None:
        vec = vec.to(device)
    return vec


def clean_npint32_to_int(data):
    if isinstance(data, list):
        return [np.int32(n).item() if isinstance(n, np.int32) else n for n in data]
    elif isinstance(data, np.int32):
        return np.int32(data).item()
    return data


# Dict operation
def l2_norm(vec: torch.Tensor):
    """
    :param vec: feature vector [D] or [N, D]
    """
    vec /= torch.norm(vec, dim=-1, keepdim=True)
    return vec


def sample_dict(d: dict, n: int, seed=None):
    """
    :param d: original dict
    :param n: number of keys to sample
    :param seed: random seed of sampling
    :return: sampled dictionary
    """
    if seed is not None:
        random.seed(seed)
    keys = random.sample(d.keys(), n)
    sample_d = {k: d[k] for k in keys}
    return sample_d


def sorted_dict(d):
    """
    :param d: the original dict
    :return: the dict sorted by value
    """
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


# Tensor operation
def cosine_sim(fea: torch.Tensor):
    """
    :param fea: feature vector [N, D]
    :return: score: cosine similarity score [N, N]
    """
    fea /= torch.norm(fea, dim=-1, keepdim=True)
    score = fea.mm(fea.T)
    return score


def uniform_normalize(t: torch.Tensor):
    """
    :param t:
    :return: normalized tensor
    >>> a = torch.rand(5)
    tensor([0.3357, 0.9217, 0.0937, 0.1567, 0.9447])
    >>> uniform_normalize(a)
    tensor([0.2843, 0.9730, 0.0000, 0.0740, 1.0000])
    """
    t -= t.min(-1, keepdim=True)[0]
    t /= t.max(-1, keepdim=True)[0]
    return t


def build_sparse_adjacent_matrix(edges: list, n: int, device=None, dtype=torch.float, undirectional=True):
    """
    Return adjacency matrix
    :param edges: list of edges, for example (st, ed)
    :param n: number of vertices
    :param device:
    :param dtype:
    :param undirectional: make adjacency matrix un-directional
    :return: the sparse adjacent matrix
    """
    i = torch.tensor(list(zip(*edges)))
    v = torch.ones(i.shape[1], dtype=dtype)
    sparse = torch.sparse_coo_tensor(i, v, (n, n))
    if device is not None:
        sparse = sparse.to(device)
    a = sparse.to_dense()
    if undirectional:
        ud_a = ((a > 0) | (a.transpose(-2, -1) > 0)).to(dtype)
        a = ud_a
    return a


def undirectionalize(mat: torch.Tensor):
    dtype = mat.dtype
    return ((mat > 0) | (mat.T > 0)).to(dtype)


def remove_undirectional_edge(mat, edges):
    if not edges:
        return mat
    x, y = zip(*edges)
    indices = [x+y, y+x]  # un-directional edges
    mat[indices] = 0  # remove edges
    return mat


def cuda(data, device):
    return [i.to(device) if isinstance(i, torch.Tensor) else i for i in data]


def stack_tensor_list(tensor_list, dim=0, value=0):
    data_size = len(tensor_list)
    probe = tensor_list[0]
    assert isinstance(probe, torch.Tensor), "Error: Invalid element type %s" % str(type(probe))
    dim_num = len(probe.shape)
    max_dims = []
    data_dims = [[] for _ in range(data_size)]
    for i_dim in range(dim_num):
        _max_dim = 0
        for i_data in range(data_size):
            data = tensor_list[i_data]
            _max_dim = max(_max_dim, data.shape[i_dim])
            data_dims[i_data].append(data.shape[i_dim])
        max_dims.append(_max_dim)

    to_stack = []
    for i_data in range(data_size):
        data = tensor_list[i_data]
        pad = []
        for i_dim in range(dim_num):
            num_to_pad = max_dims[i_dim] - data.shape[i_dim]
            pad = [0, num_to_pad] + pad
        padded = F.pad(data, pad, mode="constant", value=value)
        to_stack.append(padded)

    return torch.stack(to_stack, dim=dim)


# List statistic
def percentile(data: list, p=0.5):
    """
    :param data: origin list
    :param p: frequency percentile
    :return: the element at frequency percentile p
    """
    assert 0 < p < 1
    boundary = len(data) * p
    counter = sorted(Counter(data).items(), key=lambda x: x[0])
    keys, counts = zip(*counter)
    accumulation = 0
    for i, c in enumerate(counts):
        accumulation += c
        if accumulation > boundary:
            return keys[i]
    return None


# List visualization
def save_plt(fig_name: str, mute):
    """
    :param fig_name: path of target file
    :param mute: mute the output if True
    :return: None
    """
    if "/" in fig_name:
        fig_dir = fig_name[:fig_name.rfind("/")]
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
    plt.savefig(fig_name)
    if not mute:
        print("'%s' saved." % fig_name)


def list_histogram(data: list, color="b", title="Histogram of element frequency.", x_label="", y_label="Frequency", fig_name="hist.png", mute=False):
    """
    :param data: the origin list
    :param color: color of the histogram bars
    :param title: bottom title of the histogram
    :param x_label: label of x axis
    :param y_label: label of y axis
    :param fig_name: path of target file
    :param mute: mute the output if True
    :return: None
    """
    def adaptive_bins(_data):
        """
        :param _data: the original list to visualize
        :return: the adaptive number of bins
        """
        n = len(deduplicate(_data))
        return n

    bins = adaptive_bins(data)
    plt.hist(data, color=color, bins=bins)
    plt.gca().set(xlabel=x_label, ylabel=y_label)
    plt.title("Fig. "+title, fontdict={'family': 'serif', "verticalalignment": "bottom"})
    save_plt(fig_name, mute)
    plt.clf()


def plot_line(x, y, color="blue", marker="o", x_label="x", y_label="y", y_range=None, title="Plot of values.", fig_name="plot.png", keep_plot=False):
    """ Save the line described by x and y
    :param x:
    :param y:
    :param color:
    :param marker:
    :param x_label:
    :param y_label:
    :param y_range:
    :param title:
    :param fig_name:
    :param keep_plot:
    :return:
    """
    plt.plot(x, y, color=color, marker=marker)
    plt.gca().set(xlabel=x_label, ylabel=y_label)
    if y_range is not None:
        plt.gca().set_ylim([y_range[0], y_range[1]])
    plt.title("Fig. "+title, fontdict={'family': 'serif', "verticalalignment": "bottom"})
    save_plt(fig_name, mute=False)
    if not keep_plot:
        plt.clf()


def save_mat_show(mat, fig_name):
    """ Save the visualization of score matrix
    :param mat: the matrix to be visualized
    :param fig_name:
    :return:
    """
    plt.matshow(mat)
    save_plt(fig_name, mute=False)


def show_type_tree(data, indentation=4, depth=0, no_leaf=True):
    """
    :param data: the data to show the structure
    :param indentation: number of space of indentation
    :param depth: variable used for recursive
    :param no_leaf: don't display the leaf (non-iterable) node if True
    :return: None
    """
    def _indent(content: str):
        if depth == 0:
            print()
        print(" " * (depth * indentation) + content)

    if not isinstance(data, Iterable):
        if no_leaf:
            return

        if isinstance(data, int):
            _indent("int: %d" % data)
        elif isinstance(data, float):
            _indent("float: %.2f" % data)
        else:
            _indent("emm? " + str(type(data)))

        return

    if isinstance(data, list):
        _indent("list with size %d" % len(data))
        for item in data:
            show_type_tree(item, indentation=indentation, depth=depth+1, no_leaf=no_leaf)

    elif isinstance(data, tuple):
        _indent("tuple with size %d" % len(data))
        for item in data:
            show_type_tree(item, indentation=indentation, depth=depth+1, no_leaf=no_leaf)

    elif isinstance(data, dict):
        _indent("dict with size %d" % len(data))
        for key in data:
            _indent(str(key))
            show_type_tree(data[key], indentation=indentation, depth=depth+1, no_leaf=no_leaf)

    elif isinstance(data, str):
        _indent("str: " + data)

    elif isinstance(data, torch.Tensor):
        _indent("Tensor with shape" + str(list(data.shape)))

    else:
        _indent(str(type(data)))
        for item in data:
            show_type_tree(item, indentation=indentation, depth=depth+1, no_leaf=no_leaf)


def random_color(seed=None):
    if seed:
        random.seed(seed)  # cover all 6 times of sampling
    color = "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
    return color


def calculate_similarity(unit_visual_fea):
    score = torch.tril(cosine_sim(unit_visual_fea), diagonal=-2)
    return score


def get_adj_nei(_g: nx.Graph, _center, _nei, device=None):
    edges = [_e for _e in _g.edges if not (_center in _e and _nei not in _e)]
    return build_sparse_adjacent_matrix(edges, _g.number_of_nodes(), device=device, undirectional=True)


def prob_filter_top_k(prob, k=10):
    clip_size = prob.shape[0]
    coo = (np.arange(clip_size), prob.argsort()[:, -k])
    mask = (prob >= np.expand_dims(prob[coo], 1)).astype(np.float32)
    prob = np.where(mask > 0.5, prob, mask)
    return prob
