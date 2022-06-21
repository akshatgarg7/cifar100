# standard libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor

def transform():
    return ToTensor()

def get_data(path='data/'):
    """Downloading dataset from pytorch libary"""
    dataset = CIFAR100(root=path, download=True, train=True, transform=transform())
    test_dataset = CIFAR100(root=path, download=True, train=False, transform=transform())
    return dataset,test_dataset

def main():
    dataset, test_dataset = get_data()

if __name__ == '__main__':
    main()