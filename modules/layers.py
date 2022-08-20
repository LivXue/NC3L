from math import pi, pow

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from modules.util import *


class DirichletGaussianLayer(nn.Module):
    def __init__(self, dim: int, max_components_num=128, alpha=2.0):
        super(DirichletGaussianLayer, self).__init__()
        self.dim = dim
        self.max_components_num = max_components_num  # T
        self.alpha = float(alpha)

        components_mean = torch.Tensor(max_components_num, dim)  # Txd
        components_mean = torch.relu(torch.nn.init.orthogonal_(components_mean, gain=1))
        self.components_mean = nn.Parameter(components_mean)

        uncertainty = torch.Tensor(max_components_num)  # T
        uncertainty = torch.nn.init.constant_(uncertainty, val=0)
        self.uncertainty = nn.Parameter(uncertainty)

        log_components_cov = torch.Tensor(max_components_num)  # T
        log_components_cov = torch.nn.init.constant_(log_components_cov, val=0)
        self.log_components_cov = nn.Parameter(log_components_cov)

        raw_V = torch.Tensor(max_components_num)  # T
        raw_V = torch.nn.init.constant_(raw_V, val=-1)
        self.raw_V = nn.Parameter(raw_V)

    def LOBO(self, x):
        """
        Args:
            x: Nxd
        Returns:
            E_q(logP(V|alpha)) + E_q(logP(mu, cov|H)) + E_x(E_q[logP(Z|V)] + E_q[logP(x|Z)])
        """
        uncertainty = torch.exp(self.uncertainty)
        noise = torch.randn_like(self.components_mean) * uncertainty.unsqueeze(1)
        components_mean = self.components_mean + noise
        variance = torch.exp(self.log_components_cov).clamp(min=1e-6)  # T
        difference = x[:, None, :] - components_mean[None, :, :]  # NxTxd
        logit_z = -0.5 * (self.dim * self.log_components_cov.unsqueeze(0) +
                          difference.pow(2).sum(-1) / variance[None, :])  # NxT, omit -0.5 * self.dim * np.log(2 * pi) for numerical stability
        z = torch.softmax(logit_z, dim=1)  # NxT

        V = torch.sigmoid(self.raw_V)  # T
        log_Pv_alpha = LBetaPDF(V, 1.0, self.alpha, brief=True).sum()
        log_Pmusigma_H = LNormalGammaPDF(components_mean, variance, 0.5, 1.0, 1.0, 1.0, brief=True).sum()
        log_Pz_v = (z * torch.log(V).unsqueeze(0)).sum(1).mean()

        q_z_cum = 0
        for i in range(self.max_components_num - 1):
            q_z_cum = (q_z_cum + z[:, i]).clamp(max=1.0)  # N
            log_Pz_v += ((1 - q_z_cum) * torch.log(1 - V[i])).mean()

        log_Px_z = (z * (logit_z - 0.5 * self.dim * np.log(2 * pi))).sum(1).mean()
        log_Qz = (z * torch.log(z.clamp(min=1e-6))).sum(1).mean() - \
                 0.5 * (self.dim * self.uncertainty + noise.pow(2).sum(-1) / uncertainty).sum()

        LOBO = log_Pv_alpha + log_Pmusigma_H + 64 * (log_Pz_v + log_Px_z) - log_Qz

        return -LOBO, z

    def cluster(self, x):
        """
        Cluster without uncertainty
        """
        components_mean = self.components_mean
        variance = torch.exp(self.log_components_cov).clamp(min=1e-6)  # T
        difference = x[:, None, :] - components_mean[None, :, :]  # NxTxd
        logit_z = -0.5 * (self.dim * self.log_components_cov.unsqueeze(0) +
                          difference.pow(2).sum(-1) / variance[None, :])  # NxT
        z = torch.softmax(logit_z, dim=1)  # NxT

        return z

    def forward(self, x):
        if self.training:
            return self.LOBO(x.detach())
        else:
            return None, self.cluster(x)
