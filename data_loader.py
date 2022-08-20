import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from utils import TT_split, normalize, BackgroundGenerator
import torch
import random


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_data(dataset, test_prop):
    all_data = []
    label = []

    mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'NoisyMNIST-30000':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

    divide_seed = 123   # random.randint(1, 1000)
    train_idx, test_idx = TT_split(len(label), test_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]

    # Use test_prop*sizeof(all data) to train the model, and shuffle the rest data to simulate the unaligned data.
    # Note that, model establishes the correspondence of the all data rather than the unaligned portion in the testing.
    # When test_prop = 0, model is directly performed on the all data without shuffling.
    if test_prop != 0:
        shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        test_Y = test_Y[shuffle_idx]
        test_label_X, test_label_Y = test_label, test_label[shuffle_idx]
        all_data.append(np.concatenate((train_X, test_X)))
        all_data.append(np.concatenate((train_Y, test_Y)))
        all_label = np.concatenate((train_label, test_label))
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))
    elif test_prop == 0:
        all_data.append(train_X)
        all_data.append(train_Y)
        all_label, all_label_X, all_label_Y = train_label, train_label, train_label

    return train_X, train_Y, train_label, all_data, all_label, all_label_X, all_label_Y, divide_seed


class ContraDataset(Dataset):
    def __init__(self, view0_data, view1_data, labels):
        self.view0_data = torch.FloatTensor(view0_data)
        self.view1_data = torch.FloatTensor(view1_data)
        self.labels = labels

    def __getitem__(self, i):
        fea0, fea1 = self.view0_data[i], self.view1_data[i]
        label = self.labels[i]

        return fea0, fea1, label

    def __len__(self):
        return len(self.labels)


class GetAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1

    def __getitem__(self, index):
        fea0, fea1 = torch.from_numpy(self.data[0][index]).float(), torch.from_numpy(self.data[1][index]).float()
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        return fea0, fea1, label, class_labels0, class_labels1

    def __len__(self):
        return len(self.labels)


def loader(train_bs, test_prop, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param test_prop: known aligned proportions for training MvCLN
    :param dataset: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    train_X, train_Y, train_label, all_data, all_label, all_label_X, all_label_Y, divide_seed = load_data(dataset, test_prop)
    train_pair_dataset = ContraDataset(train_X, train_Y, train_label)
    all_dataset = GetAllDataset(all_data, all_label, all_label_X, all_label_Y)

    train_pair_loader = DataLoaderX(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoaderX(
        all_dataset,
        batch_size=1024,
        shuffle=False
    )
    return train_pair_loader, all_loader, divide_seed
