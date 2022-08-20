import torch
import torch.nn as nn


def ContrastiveLoss(pair_dist, P, margin, lambd, pos_mask=None):
    dist_sq = pair_dist * pair_dist
    N = P.numel()
    pos_num = P.sum()
    neg_num = N - pos_num
    if pos_mask is not None:
        pos_num = pos_mask.sum()
        N = pos_num + neg_num
    pos_weight = neg_num / N
    neg_weight = pos_num / N * lambd
    if pos_mask is not None:
        pos_weight = pos_weight * pos_mask

    loss = pos_weight * P * dist_sq + neg_weight * (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)

    loss = torch.sum(loss) * (N / pos_num / neg_num)
    return loss
