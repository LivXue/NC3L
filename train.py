import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from alignment import euclidean_dist
from loss import *
from modules.util import *
from knn_cluter import knn_pseudo_label
pseudo_precision_list, pseudo_recall_list = [], []


def train(train_loader, model, optimizer, epoch, args):
    pos_mean_dist = 0  # mean distance of pos. pairs
    neg_mean_dist = 0
    false_neg_dist = 0  # mean distance of false neg. pairs (pairs in noisy labels)
    true_neg_dist = 0
    pos_count = 0  # count of pos. pairs
    neg_count = 0
    false_neg_count = 0  # count of neg. pairs (pairs in noisy labels)
    true_neg_count = 0

    print()
    logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    loss_value = 0
    pseudo_precision, pseudo_recall = [], []
    for list_tmp in train_loader:
        # real_labels are the clean labels for these pairs
        # we additionally get index for data analysis
        list_tmp = [_.to(args.gpu) for _ in list_tmp]
        x0, x1, labels = list_tmp

        h0, h1, LOBO0, LOBO1, z0, z1 = model(x0, x1)
        distance = euclidean_dist(h0, h1)
        pos_dist = torch.diag(distance)
        real_labels = (labels[:, None] == labels[None, :]).float()

        true_neg_dist += torch.sum(distance[real_labels == 0]).item()
        false_neg_dist += torch.sum(distance[real_labels == 1]).item() - pos_dist.sum().item()
        pos_mean_dist += pos_dist.sum().item()
        neg_mean_dist += distance.sum().item() - pos_dist.sum().item()
        pos_count += len(h0)
        neg_count += len(h0) ** 2 - len(h0)
        true_neg_count += (real_labels == 0).sum().item()
        false_neg_count += (real_labels == 1).sum().item() - len(h0)

        contra_labels = torch.diag(torch.ones_like(labels, dtype=torch.float))
        pseudo_label1, pseudo_label2 = (z0 > 1 / 50) * z0, (z1 > 1 / 50) * z1
        pseudo_label1, pseudo_label2 = pseudo_label1.detach(), pseudo_label2.detach()
        pseudo_label = SimiMatrix(pseudo_label1, pseudo_label2, args.threshold)
        #pseudo_label = knn_pseudo_label(h0.detach().cpu().numpy(), h1.detach().cpu().numpy())  #knn
        #pseudo_label = (pseudo_label[:, None] == pseudo_label[None, :]).float().to(args.gpu)   #knn
        pseudo_recall.append((pseudo_label * real_labels).sum().item() / (real_labels.sum().item() - x0.shape[0]))
        pseudo_precision.append((pseudo_label * real_labels).sum().item() / pseudo_label.sum().clamp(min=1e-6).item())

        if epoch > args.start_epoch:
            contra_labels = contra_labels + pseudo_label
            contra_labels = contra_labels.clamp(min=0., max=1.0)

        loss = ContrastiveLoss(distance, contra_labels, args.margin, args.lambd)
        #loss = ContrastiveLoss(distance, real_labels, args.margin, args.lambd)
        if epoch <= args.end_epoch:
            loss += LOBO0 + LOBO1
        loss_value += loss.item()
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pos_mean_dist /= pos_count
    neg_mean_dist /= neg_count
    true_neg_dist /= true_neg_count
    false_neg_dist /= false_neg_count

    pseudo_precision = sum(pseudo_precision) / len(pseudo_precision)
    pseudo_precision_list.append(pseudo_precision)
    pseudo_recall = sum(pseudo_recall) / len(pseudo_recall)
    pseudo_recall_list.append(pseudo_recall)

    # margin = the pos. distance + neg. distance before training
    if epoch == 0 and args.margin != 1.0:
        args.margin = max(1, (pos_mean_dist + neg_mean_dist))
        logging.info("margin = {}".format(args.margin))

    noise_rate = false_neg_count / (false_neg_count + true_neg_count) * 100
    logging.info("distance: pos. = {:.2f}, neg. = {:.2f}, true neg. = {:.2f}, false neg. = {:.2f}, noise rate = {:.2f}%".
                 format(pos_mean_dist, neg_mean_dist, true_neg_dist, false_neg_dist, noise_rate))
    logging.info("precision = {:.2f}%, recall = {:.2f}%, loss = {:.2f}".
                 format(pseudo_precision*100, pseudo_recall*100, loss_value / len(train_loader)))
    return pos_mean_dist, neg_mean_dist, false_neg_dist, true_neg_dist, pseudo_precision_list, pseudo_recall_list
