import torch
import numpy as np


def LBetaPDF(x, a, b, brief=False):
    """
    x ~ Be(a, b)
    Args:
        x: variable
        a: alpha
        b: beta
        brief: omit constants to compute faster for only optimizing x
    Returns:
        logarithm of Beta probability density function
    """
    if brief:
        LP = (a - 1) * torch.log(x) + (b - 1) * torch.log(1 - x)
    else:
        LP = torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b) + (a - 1) * torch.log(x) + (b - 1) * torch.log(1-x)

    return LP


def LGammaPDF(x, a, b, brief=False):
    """
    x ~ Gamma(a, b)
    Args:
        x: variable
        a: alpha
        b: beta
        brief: omit constants to compute faster for only optimizing x
    Returns:
        logarithm of Gamma probability density function
    """
    if brief:
        LP = (a - 1) * x - x / b
    else:
        LP = (a - 1) * x - x / b - a * np.log(b) - torch.lgamma(a)

    return LP


def LNormalGammaPDF(mu, sigma, m=0, c=1, a=1, b=1, brief=False):
    """
    mu, sigma^-1 ~ NormalGamma(m, c^-1, a, b)
    sigma^-1 ~ Gamma(a,b)
    mu ~ N(m, sigma * c)
    Args:
        mu: posterior mean, Txd-dimensional
        sigma: posterior covariance, T-dimensional
        m: prior mean, d-dimensional
        c: prior covariance weight, scalar
        a: alpha of gamma distribution, scalar
        b: beta of gamma distribution, scalar
        brief: omit constants to compute faster for only optimizing mu, sigma
    Returns:
        LP: logarithm of NormalGamma probability density function, T-dimensional
    """
    if mu.ndim == 1:
        d = mu.shape[0]
    else:
        d = mu.shape[1]
    pi = 3.1415926535898
    LPsab = d * LGammaPDF(1 / sigma.clamp(min=1e-6), a, b, brief=brief)

    if brief:
        LPmusc = -0.5 * (d * torch.log(c * sigma) + (mu - m).pow(2).sum(1) / (c * sigma).clamp(1e-6))
    else:
        LPmusc = -0.5 * (d * np.log(2 * pi) + d * torch.log(c * sigma) + (mu - m).pow(2).sum(1) / (c * sigma).clamp(1e-6))

    LP = LPsab + LPmusc

    return LP


def SimiMatrix(label1, label2=None, threshold=0.95):
    if label2 is None:
        label2 = label1
    simi = (label1.mm(label1.T) + label2.mm(label2.T)) / 2
    simi = simi / torch.diag(simi).clamp(1e-6).sqrt().unsqueeze(0) / torch.diag(simi).clamp(1e-6).sqrt().unsqueeze(1)
    #simi = ((simi - torch.diag(torch.ones_like(label1[:, 0]))) > 0.95) * simi
    simi -= torch.diag(torch.diag(simi))
    simi = (simi > threshold) * simi
    simi = simi.clamp(min=0.0, max=1.0)
    return simi


def pos_grad(h0, h1):
    """
    Args:
        h0: view0 features
        h1: view1 features

    Returns: negative gradients of positive pairs
    """
    grad0 = 2 * (h1[None, :, :] - h0[:, None, :])
    grad1 = -grad0.transpose(0, 1)
    return grad0, grad1
