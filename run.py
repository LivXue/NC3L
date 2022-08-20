import argparse
import random
import time
import copy
import logging
import os
import sys
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from data_loader import loader
from models import *
from alignment import tiny_infer
from Clustering import Clustering
from train import train

parser = argparse.ArgumentParser(description='NC3L in PyTorch')
parser.add_argument('--data', default=3, type=int,
                    help='choice of dataset, 0-Scene15, 1-Caltech101, 2-Reuters10, 3-NoisyMNIST')
parser.add_argument('-lr', '--learn-rate', default=1e-3, type=float, help='learning rate of adam')
parser.add_argument('-ap', '--aligned-prop', default='0.5', type=float,
                    help='originally aligned proportions in the partially view-aligned data')
parser.add_argument('-m', '--margin', default='5', type=int, help='initial margin')
parser.add_argument('--threshold', default='0.95', type=float)
parser.add_argument('--gpu', default=1, type=int, help='GPU device idx to use.')
# mean distance of four kinds of pairs, namely, pos., neg., true neg., and false neg. (noisy labels)
pos_dist_mean_list, neg_dist_mean_list, true_neg_dist_mean_list, false_neg_dist_mean_list = [], [], [], []


if __name__ == '__main__':
    args = parser.parse_args()
    data_name = ['Scene15', 'Caltech101', 'Reuters_dim10', 'NoisyMNIST-30000']
    NetSeed = 64
    random.seed(NetSeed)
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)  # set CPU random seed
    torch.cuda.manual_seed(NetSeed)  # set GPU random seed
    torch.cuda.manual_seed_all(NetSeed)

    # Dataset settings
    if args.data == 0:
        model = MvCLNfcScene().to(args.gpu)
        args.bs = 512   # batch size
        args.epochs = 100  # number of epochs to run
        args.start_epoch = 5
        args.end_epoch = 20
        args.lambd = 2
    elif args.data == 1:
        model = MvCLNfcCaltech().to(args.gpu)
        args.bs = 512
        args.epochs = 100
        args.start_epoch = 8
        args.end_epoch = 20
        args.lambd = 2
    elif args.data == 2:
        model = MvCLNfcReuters().to(args.gpu)
        args.bs = 512
        args.epochs = 100
        args.start_epoch = 5
        args.end_epoch = 15
        args.lambd = 2
    elif args.data == 3:
        model = MvCLNfcMNIST().to(args.gpu)
        args.bs = 1024
        args.epochs = 100
        args.start_epoch = 5
        args.end_epoch = 30
        args.lambd = 2

    train_pair_loader, all_loader, divide_seed = loader(args.bs, args.aligned_prop, data_name[args.data])

    best_model_wts = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    if not os.path.exists("./log/"):
        os.mkdir("./log/")
    path = os.path.join("./log/" + str(data_name[args.data]) + "_" + 'time=' +
                        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    # os.mkdir(path)

    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(path + '.txt', 'w')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(
        "******** Training begin, use gpu {}, batch_size = {}, unaligned_prop = {}, NetSeed = {}, DivSeed = {} ********".format(
            args.gpu, args.bs, (1 - args.aligned_prop), NetSeed, divide_seed)
    )
    logging.info("Using dataset: {}".format(data_name[args.data]))

    CAR_list = []
    acc_list, nmi_list, ari_list = [], [], []
    train_time = 0
    best_scores = {'sum': 0.0}
    # train
    for i in range(0, args.epochs + 1):
        time0 = time.time()

        if i == 0:
            with torch.no_grad():
                tmp = train(train_pair_loader, model, optimizer, i, args)
                pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, precision_list, recall_list = tmp
        else:
            tmp = train(train_pair_loader, model, optimizer, i, args)
            pos_dist_mean, neg_dist_mean, false_neg_dist_mean, true_neg_dist_mean, precision_list, recall_list = tmp

        epoch_time = time.time() - time0

        pos_dist_mean_list.append(pos_dist_mean)
        neg_dist_mean_list.append(neg_dist_mean)
        true_neg_dist_mean_list.append(true_neg_dist_mean)
        false_neg_dist_mean_list.append(false_neg_dist_mean)

        # test
        v0, v1, pred_label, alignment_rate = tiny_infer(model, args.gpu, all_loader, args.ap)
        CAR_list.append(alignment_rate)
        data = [v0, v1]
        aligned_data = np.concatenate(data, axis=1)
        y_pred, ret = Clustering(aligned_data, pred_label)
        
        # record best scores
        if (ret['kmeans']['accuracy'] + ret['kmeans']['NMI'] + ret['kmeans']['ARI']) > best_scores['sum']:
            best_scores['accuracy'] = ret['kmeans']['accuracy']
            best_scores['NMI'] = ret['kmeans']['NMI']
            best_scores['ARI'] = ret['kmeans']['ARI']
            best_scores['sum'] = best_scores['accuracy'] + best_scores['NMI'] + best_scores['ARI']
            best_model_wts = copy.deepcopy(model.state_dict())

        logging.info("******** testing ********")
        logging.info("CAR={:.4f}, kmeans: acc={}, nmi={}, ari={}".format(alignment_rate, ret['kmeans']['accuracy'],
                                                                         ret['kmeans']['NMI'], ret['kmeans']['ARI']))
        acc_list.append(ret['kmeans']['accuracy'])
        nmi_list.append(ret['kmeans']['NMI'])
        ari_list.append(ret['kmeans']['ARI'])

        train_time += epoch_time
        logging.info("epoch_time = {:.2f} s".format(epoch_time))

    model.load_state_dict(best_model_wts)
    # plot(acc_list, nmi_list, ari_list, CAR_list, args, data_name[args.data])
    print()
    logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))
    logging.info("******** Best Scores ********")
    logging.info("kmeans: acc={}, nmi={}, ari={}".format(best_scores['accuracy'], best_scores['NMI'], best_scores['ARI']))

    json_history = {'acc': acc_list, 'nmi': nmi_list, 'ari': ari_list, 'pseudo_precision': precision_list,
                    'pseudo_recall': recall_list, 'pos': pos_dist_mean_list, 'neg': neg_dist_mean_list,
                    'true neg': true_neg_dist_mean_list, 'false neg': false_neg_dist_mean_list}

    #grad_info = {'avg_total_grad': cum_norm_list, 'avg_norm_list': avg_norm_list}
