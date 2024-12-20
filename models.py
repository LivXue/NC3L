import torch.nn as nn
import torch.nn.functional as F

from modules.layers import *
from modules.kernel import RFF
#from gibbs.igmm import igmm_full_cov_sampler


class MvCLNfcMNIST(nn.Module):
    def __init__(self, output_dim=64):
        super(MvCLNfcMNIST, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.DPGMM = DirichletGaussianLayer(output_dim)

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)

        LOBO0, z0 = self.DPGMM(h0)
        LOBO1, z1 = self.DPGMM(h1)

        return h0, h1, LOBO0, LOBO1, z0, z1


class MvCLNfcCaltech(nn.Module):
    def __init__(self, output_dim=64):
        super(MvCLNfcCaltech, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(1984, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.DPGMM = DirichletGaussianLayer(output_dim)

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)

        LOBO0, z0 = self.DPGMM(h0)
        LOBO1, z1 = self.DPGMM(h1)

        return h0, h1, LOBO0, LOBO1, z0, z1


class MvCLNfcScene(nn.Module):  # 20, 59
    def __init__(self, output_dim=64):
        super(MvCLNfcScene, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True),
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(59, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.DPGMM = DirichletGaussianLayer(output_dim)

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)

        LOBO0, z0 = self.DPGMM(h0)
        LOBO1, z1 = self.DPGMM(h1)

        return h0, h1, LOBO0, LOBO1, z0, z1


class MvCLNfcReuters(nn.Module):
    def __init__(self, output_dim=64):
        super(MvCLNfcReuters, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(True)
        )
        self.DPGMM = DirichletGaussianLayer(output_dim)

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)

        LOBO0, z0 = self.DPGMM(h0)
        LOBO1, z1 = self.DPGMM(h1)

        return h0, h1, LOBO0, LOBO1, z0, z1
