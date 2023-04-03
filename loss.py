import torch
import torch.nn as nn


def l2norm(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    norm = torch.norm(x, dim=-1, keepdim=True) + eps
    return 1.0 * x / norm


def order_sim(s, im):
    """ Order embeddings similarity measure $max(0, sim)$
    """
    y_m_x = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
             - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -y_m_x.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(s, im):
    y_m_x = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
             - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -y_m_x.pow(2).sum(2).t()
    return score


def calc_similarity(vid: torch.Tensor, cap: torch.Tensor, measure='cosine') -> torch.Tensor:
    """ similarity of vector matching
    """
    if measure != 'cosine':
        raise NotImplementedError

    vid = l2norm(vid)
    cap = l2norm(cap)
    sim = torch.mm(cap, vid.t())
    return sim


def calc_seq_similarity(vid_frames, vid_len, cap_words, cap_len, measure='cosine'):
    """ similarity of sequence matching
    """
    raise NotImplementedError


class TripletLoss(nn.Module):
    """ triplet ranking loss for vr
    """

    def __init__(self,
                 margin=0,
                 measure="cosine",
                 max_violation=False,
                 cost_style='sum',
                 direction='all',
                 pos_loss_term=False,
                 seq_match=False):

        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        self.seq_match = seq_match
        self.pos_loss_term = pos_loss_term
        self.sim = {
            "order": order_sim,
            "euclidean": euclidean_sim,
            "cosine": calc_similarity
        }[measure]
        self.seq_sim = {"cosine": calc_seq_similarity}[measure]
        self.max_violation = max_violation

    def forward(self, cap, vid, cap_len=None, vid_len=None):
        """ compute image-sentence score matrix
        if cap and vid are 2-D
            cap: caption embedding in shape of [batch_size, word_emb_size]
            vid: video embedding in shape of [batch_size, vid_embed_size]

        if cap and vid are 3-D
            cap: caption word embedding sequence in shape of [batch_size, max_word_size, word_emb_size]
            vid: video frame embedding in shape of [batch_size, max_frame_size, vid_embed_size]
            cap_len: length of word in each caption, [batch_size, 1]
            vid_len: length of frame in each video, [batch_size, 1]
        """
        scores = self.calc_sim(cap, vid, cap_len, vid_len)
        loss = self.ranking_loss(scores)
        return loss

    def ranking_loss(self, scores):
        batch_size = scores.shape[0]
        diag = scores.diag().view(batch_size, 1)  # score between i-th im and i-th sentence
        d1 = diag.expand_as(scores)
        d2 = diag.t().expand_as(scores)

        # clear diagonals
        diag_mask = torch.eye(batch_size, dtype=torch.bool).cuda()
        cost_c, cost_im = torch.zeros(1).cuda(), torch.zeros(1).cuda()

        if self.direction in ['i2t', 'all']:
            # caption retrieval
            cost_c = (self.margin + scores - d1).clamp(min=0)
            cost_c = cost_c.masked_fill_(diag_mask, 0)  # clear margin
            if self.max_violation:
                cost_c = cost_c.max(dim=1)[0]

        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(diag_mask, 0)  # clear margin
            if self.max_violation:
                cost_im = cost_im.max(dim=0)[0]

        if self.cost_style == 'sum':
            return cost_c.sum() + cost_im.sum()

        return cost_c.mean() + cost_im.mean()

    def ranking_loss_one_to_more(self, scores: torch.Tensor, label: torch.Tensor, mask: torch.Tensor):
        device = scores.device
        batch_size, neighbor_size = scores.shape
        coo_gold = (torch.arange(batch_size, device=device), label)
        d1 = scores[coo_gold].unsqueeze(dim=1).expand_as(scores)

        # clear diagonals
        diag_mask = torch.zeros_like(mask, device=device, dtype=torch.bool)
        diag_mask[coo_gold] = True

        # caption retrieval
        cost_c = (scores - d1 + self.margin).clamp(min=0)
        cost_c = cost_c.masked_fill_(diag_mask, 0)  # clear margin
        cost_c = cost_c.masked_fill_(torch.logical_not(mask), 0)  # clear illegal sample

        if self.max_violation:
            cost_c = cost_c.max(dim=1)[0]
        else:
            n_sample = mask.sum(dim=-1, keepdim=True)
            cost_c = cost_c.sum(dim=-1, keepdim=True) / (n_sample + 1e-5)
        return cost_c.mean()

    def calc_sim(self, cap: torch.Tensor, vid, cap_len=None, vid_len=None):
        if self.seq_match:
            assert cap.dim() == 3 and vid.dim() == 3, "incorrect dimensionality"
            scores = self.seq_sim(cap, cap_len, vid, vid_len)  # image-to-text
        else:
            assert cap.dim() == 2 and vid.dim() == 2, ("incorrect dimensionality", cap.shape, vid.shape)
            scores = self.sim(cap, vid)  # image-to-text
        return scores


class FavorPositiveBCELoss(nn.Module):
    """
    positive cross entropy
    Math:
    """
    def __init__(self, loss_lambda=0.1, cost_style='mean'):
        super(FavorPositiveBCELoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.cost_style = cost_style

    def forward(self, outs, labels):
        alpha = self.loss_lambda
        positive_mask = labels.float()
        negative_mask = 1-positive_mask
        bce_loss= -labels * torch.log(outs+1e-05) - (1 - labels) * torch.log(1 - outs+1e-05)
        positive_loss = bce_loss * positive_mask
        negative_loss = bce_loss * negative_mask

        positive_loss_batch = torch.sum(positive_loss, 1) / torch.sum(positive_mask, 1)
        negative_loss_batch = torch.sum(negative_loss, 1) / torch.sum(negative_mask, 1)
        combined_loss = alpha*positive_loss_batch + (1-alpha)*negative_loss_batch
        if self.cost_style == 'sum':
            loss = torch.sum(combined_loss)
        else:
            loss = torch.mean(combined_loss)
        return loss
