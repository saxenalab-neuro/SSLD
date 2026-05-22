import numpy as np
import torch
import torch.nn as nn
import math

def get_loss(gamma1,delta1,prob_ini,prob_all_s,prob_all_h,prob_all_y,prob_all_beh):
    # see Equation 15 of our paper
    t1=torch.sum(torch.exp(gamma1)*(prob_ini+prob_all_h[:,0]+prob_all_y[:,0][:,None]+prob_all_beh[:,0][:,None]))

    # Vectorized: replaces for loop over time steps
    # delta1 shape: (B, T-1, N, N)
    # prob_all_s[:, 1:] shape: (B, T-1, N, N)
    # prob_all_h[:, 1:] shape: (B, T-1, N) -> need (B, T-1, 1, N) for broadcasting
    # prob_all_y[:, 1:] shape: (B, T-1) -> need (B, T-1, 1, 1) for broadcasting
    # prob_all_beh[:, 1:] shape: (B, T-1) -> need (B, T-1, 1, 1)
    terms = (prob_all_s[:, 1:]
           + prob_all_h[:, 1:, None, :]
           + prob_all_y[:, 1:, None, None]
           + prob_all_beh[:, 1:, None, None])
    t2 = torch.sum(torch.exp(delta1) * terms)

    return t1,t2

def get_cross_entropy(pos,pri):
    # Fully vectorized: replaces double nested loop over (k, i)
    return (pos * pri).sum()


def get_kernel(X, Z, ksize):
    # Expanding X at dimension 1 and subtracting Z to calculate pairwise squared distances
    G = torch.sum((X.unsqueeze(1) - Z)**2, dim=-1)  # Gram matrix
    
    # Calculating the Gaussian kernel
    G = torch.exp(-G / ksize) / (math.sqrt(2 * math.pi * ksize) * torch.ones_like(-G/ksize))
    return G

class csLoss(nn.Module):
    def __init__(self,ks):
        super(csLoss, self).__init__()
        self.ksize=ks
    def forward(self, X,Z):

        ksize=self.ksize
        Gxx = get_kernel(X, X,ksize)
        Gzz = get_kernel(Z, Z,ksize)
        Gxz = get_kernel(X, Z,ksize)
        r = torch.log(torch.sqrt(torch.mean(Gxx) * torch.mean(Gzz) + 1e-5) / (torch.mean(Gxz) + 1e-5))
        return r