import torch
from paths import RUNS


def save_model(ckpt):
    last = RUNS / 'last.pt'
    torch.save(ckpt, last)