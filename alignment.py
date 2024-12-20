import random

import torch
import numpy as np


def single_infer(model, device, all_loader):
    model.eval()
    out0, out1 = [], []
    labels0, labels1 = [], []
    with torch.no_grad():
        for batch_idx, (x0, x1, labels, class_labels0, class_labels1) in enumerate(all_loader):
            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            z0, z1, _, _ = model(x0, x1)

            out0.append(z0)
            out1.append(z1)
            labels0.append(class_labels0)
            labels1.append(class_labels1)

    out0 = torch.concat(out0)
    out1 = torch.concat(out1)
    labels0 = torch.concat(labels0)
    labels1 = torch.concat(labels1)

    return out0.cpu().numpy(), out1.cpu().numpy(), labels0.cpu().numpy(), labels1.cpu().numpy()


def tiny_infer(model, device, all_loader, aligned_prop):
    model.eval()
    align_out0 = []
    align_out1 = []
    align_labels0 = []
    align_labels1 = []
    with torch.no_grad():
        for x0, x1, labels, class_labels0, class_labels1 in all_loader:
            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            h0, h1, _, _, _, _ = model(x0, x1)

            align_out0.append(h0)
            align_out1.append(h1)
            align_labels0.append(class_labels0)
            align_labels1.append(class_labels1)

    align_out0 = torch.concat(align_out0)
    align_out1 = torch.concat(align_out1)
    align_labels0 = torch.concat(align_labels0)
    align_labels1 = torch.concat(align_labels1)

    data_num = len(align_out0)
    train_num = int(data_num * aligned_prop)
    test_num = data_num - train_num

    # mean train features
    mean_train_data = (align_out0[:train_num] + align_out1[:train_num]) / 2
    align_out0[:train_num] = mean_train_data
    align_out1[:train_num] = mean_train_data

    align_out0, align_out1, class_labels, align_acc = random_align(align_out0, align_out1, align_labels0, align_labels1)

    return align_out0.cpu().numpy(), align_out1.cpu().numpy(), class_labels.cpu().numpy(), align_acc


def random_align(h0, h1, class_labels0, class_labels1, sampling_size=2048):
    """
    Args:
        h0: features of view 0  (n, d)
        h1: features of view 1  (n, d)
        class_labels0: class labels of view 0
        class_labels1: class labels of view 1
        sampling_size: size of random sampling
    Returns:
        align_out0: aligned features of view 0
        align_out1: aligned features of view 1
        class_labels: class labels for align_out0
        align_acc: accuracy of alignment
    """
    length = len(class_labels0)
    acc_counter = 0

    def row_exchange(A, i, j):
        if i == j:
            return
        b = A[i].clone()
        A[i], A[j] = A[j], b

    for i in range(length):
        # sampling candidates
        sampling_num = min(sampling_size, length - i)
        sampling_idx = random.sample(list(range(i, length)), sampling_num)
        #sampling_idx = list(range(i, i + sampling_num))
        candidates_view1 = h1[sampling_idx]

        # select optimal candidate
        dist = euclidean_dist(h0[i], candidates_view1)
        idx = sampling_idx[torch.argmin(dist.squeeze())]

        # move to h1[i]
        row_exchange(h1, idx, i)
        row_exchange(class_labels1, idx, i)

        if class_labels0[i] == class_labels1[i]:
            acc_counter += 1

    align_acc = acc_counter / length

    return h0, h1, class_labels0, align_acc


def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        euclidean dist: pytorch Variable, with shape [m, n]
    """

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist -= 2 * x.mm(y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def deprecated_tiny_infer(model, device, all_loader):
    model.eval()
    all_loader.shuffle = True
    align_out0 = []
    align_out1 = []
    class_labels_cluster = []
    len_alldata = len(all_loader.dataset)
    align_labels = torch.zeros(len_alldata)
    with torch.no_grad():
        for batch_idx, (x0, x1, labels, class_labels0, class_labels1) in enumerate(all_loader):
            test_num = len(labels)

            x0, x1, labels = x0.to(device), x1.to(device), labels.to(device)
            h0, h1, _, _, _, _ = model(x0, x1)

            C = euclidean_dist(h0, h1)
            for i in range(test_num):
                idx = torch.argmin(C[i, :])
                C[:, idx] = float("inf")
                align_out0.append((h0[i].cpu()).numpy())
                align_out1.append((h1[idx].cpu()).numpy())
                if class_labels0[i] == class_labels1[idx]:
                    align_labels[len(align_out0) - 1] = 1

            class_labels_cluster.extend(class_labels0.numpy())

    count = torch.sum(align_labels)
    inference_acc = count.item() / len_alldata

    return np.array(align_out0), np.array(align_out1), np.array(class_labels_cluster), inference_acc
