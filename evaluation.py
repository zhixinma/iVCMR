import torch
import numpy as np


class MetricScorer:
    def __init__(self, k=0):
        self.k = k

    def score(self, sorted_labels):
        return 0.0

    def getLength(self, sorted_labels):
        length = self.k
        if length > len(sorted_labels) or length <= 0:
            length = len(sorted_labels)
        return length

    def name(self):
        if self.k > 0:
            return "%s@%d" % (self.__class__.__name__.replace("Scorer",""), self.k)
        return self.__class__.__name__.replace("Scorer","")


class APScorer(MetricScorer):
    def __init__(self, k):
        MetricScorer.__init__(self, k)

    def score(self, sorted_labels):
        nr_relevant = len([x for x in sorted_labels if x > 0])
        if nr_relevant == 0:
            return 0.0

        length = self.getLength(sorted_labels)
        ap = 0.0
        rel = 0

        for i in range(length):
            lab = sorted_labels[i]
            if lab >= 1:
                rel += 1
                ap += float(rel) / ( i +1.0)
        ap /= nr_relevant
        return ap


def get_scorer(name):
    mapping = {"AP": APScorer}
    elems = name.split("@")
    if len(elems) == 2:
        k = int(elems[1])
    else:
        k = 0
    return mapping[elems[0]](k)


def calc_rank(c2i_score, n_caption, eval_direction="c2i"):
    """ Text to Videos Retrieval
        c2i: [n_caption * N, N] matrix of caption to video errors
        vis_details: if true, return a dictionary for ROC visualization purposes
    """
    assert c2i_score.shape[0] / c2i_score.shape[1] == n_caption, "Inconsistent data shape and caption number."
    is_c2i = eval_direction == "c2i"
    score_matrix = c2i_score if is_c2i else c2i_score.transpose(1, 0)
    a_size, b_size = score_matrix.shape
    ranks = []
    for i in range(a_size):
        score_i = score_matrix[i]
        sorted_ids = score_i.argsort(descending=True)
        label = i // n_caption if is_c2i else i
        sorted_ids = sorted_ids if is_c2i else sorted_ids // n_caption
        rank = torch.where(sorted_ids == label)[0]
        ranks.append(rank[0])

    return torch.tensor(ranks).cuda()


def mean_average_precision(c2i_score, n_caption=2, eval_direction="c2i"):
    assert c2i_score.shape[0] / c2i_score.shape[1] == n_caption, "Inconsistent data shape and caption number."
    is_c2i = eval_direction == "c2i"
    score_matrix = c2i_score if is_c2i else c2i_score.transpose(1, 0)
    scorer = get_scorer('AP')
    score_list = []
    for i in range(score_matrix.shape[0]):
        d_i = score_matrix[i, :]
        labels = [0 for _ in range(len(d_i))]
        if is_c2i:
            labels[i // n_caption] = 1
        else:
            labels[i * n_caption:(i + 1) * n_caption] = [1] * n_caption
        sorted_labels = [labels[x] for x in d_i.argsort()]
        current_score = scorer.score(sorted_labels)
        score_list.append(current_score)
    return np.sum(score_list)


def evaluate_rank(ranks, direction, epoch_num):
    # Compute metrics
    data_size = ranks.shape[0]
    r1 = 1.0 * np.where(ranks < 1)[0].shape[0] / data_size
    r2 = 1.0 * np.where(ranks < 2)[0].shape[0] / data_size
    r5 = 1.0 * np.where(ranks < 5)[0].shape[0] / data_size
    r10 = 1.0 * np.where(ranks < 10)[0].shape[0] / data_size
    r_med = np.floor(np.median(ranks)).item() + 1
    r_mean = ranks.mean().item() + 1

    tag = {
        "c2i": "Text to video",
        "i2c": "Video to text"
    }[direction]

    # caption retrieval
    print(" * {} rank@[1, 2, 5, 10]: {}".format(tag, [round(r1, 3), round(r2, 3), round(r5, 3), round(r10, 3)]))
    print(" * median rank, mean rank: {}".format([round(r_med, 3), round(r_mean, 3)]))
    print(" * " + '-' * 10)

    score = (r1 + r2 + r5 + r10) / 4
    return score
