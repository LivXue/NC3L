import logging

import numpy as np
import torch


def train(train_loader, model, optimizer, args):
    print()
    pseudo_precision_list, pseudo_recall_list = [], []
    for epoch in range(args.epochs + 1):
        pos_mean_dist = 0  # mean distance of pos. pairs
        neg_mean_dist = 0
        false_neg_dist = 0  # mean distance of false neg. pairs (pairs in noisy labels)
        true_neg_dist = 0
        pos_count = 0  # count of pos. pairs
        neg_count = 0
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
        model.train()
        loss_value = 0
        pseudo_precision, pseudo_recall = [], []
        false_neg_count, true_neg_count = 0, 0
        for list_tmp in train_loader:
            # real_labels & labels are for data analysis
            list_tmp = [_.cuda() for _ in list_tmp]
            x0, x1, labels, pseudo_labels, _ = list_tmp

            loss, pseudo_contra_labels, pair_dist = model.loss(x0, x1, pseudo_labels, epoch, args)

            real_labels = (labels[:, None] == labels[None, :]).float()
            pseudo_recall.append((pseudo_contra_labels * real_labels).sum().item() / real_labels.sum().item())
            pseudo_precision.append(
                (pseudo_contra_labels * real_labels).sum().item() / pseudo_contra_labels.sum().clamp(min=1e-6).item())

            pos_dist = torch.diag(pair_dist)
            true_neg_dist += torch.sum(pair_dist[real_labels == 0]).item()
            false_neg_dist += torch.sum(pair_dist[real_labels == 1]).item() - pos_dist.sum().item()
            pos_mean_dist += pos_dist.sum().item()
            neg_mean_dist += pair_dist.sum().item() - pos_dist.sum().item()
            pos_count += len(x0)
            neg_count += len(x0) ** 2 - len(x0)
            true_neg_count += (real_labels == 0).sum().item()
            false_neg_count += (real_labels == 1).sum().item() - len(x0)

            loss_value += loss.item()

            if epoch != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #if epoch == args.start_CICL_epoch:
        #    update_pseudo_labels(train_loader, model)

        # for analysis
        pseudo_precision = sum(pseudo_precision) / len(pseudo_precision) * 100
        pseudo_precision_list.append(pseudo_precision)
        pseudo_recall = sum(pseudo_recall) / len(pseudo_recall) * 100
        pseudo_recall_list.append(pseudo_recall)

        pos_mean_dist /= pos_count
        neg_mean_dist /= neg_count
        true_neg_dist /= true_neg_count
        false_neg_dist /= false_neg_count

        if epoch == 0 and args.margin != 1.0:
            args.margin = max(1, (pos_mean_dist + neg_mean_dist))
            logging.info("margin = {}".format(args.margin))

        noise_rate = false_neg_count / (false_neg_count + true_neg_count) * 100
        logging.info("loss = {:.2f}, FNP rate = {:.2f}%".format(loss_value / len(train_loader), noise_rate))
        logging.info("distance: pos. = {:.2f}, neg. = {:.2f}, true neg. = {:.2f}, false neg. = {:.2f}".
                     format(pos_mean_dist, neg_mean_dist, true_neg_dist, false_neg_dist))
        logging.info("Pseudo labels: pre = {:.2f}%, rec = {:.2f}%".format(pseudo_precision, pseudo_recall))

    return model, pseudo_precision_list, pseudo_recall_list


def update_pseudo_labels(train_loader, model):
    for list_tmp in train_loader:
        # real_labels are for data analysis
        list_tmp = [_.cuda() for _ in list_tmp]
        x0, x1, _, _, inds = list_tmp

        with torch.no_grad():
            modal0_feature = model.projector(model.modal0_encoder(x0))
            modal1_feature = model.projector(model.modal1_encoder(x1))

            z0 = model.DPGMM.cluster(modal0_feature)
            z1 = model.DPGMM.cluster(modal1_feature)
            z0, z1 = (z0 > 1 / 50) * z0, (z1 > 1 / 50) * z1

            z = (z0 + z1) / 2

        train_loader.dataset.pseudo_y[inds.cpu()] = z.cpu().numpy()
