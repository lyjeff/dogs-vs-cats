import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data.sampler import SubsetRandomSampler


def argument_setting():

    parser = ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='VGG19')
    parser.add_argument('--holdout_p', type=float, default=0.8)
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--train_path', type=str, default='./data/train/')
    parser.add_argument('--test_path', type=str, default='./data/test1/')
    parser.add_argument('--submit_csv', type=str, default='./data/sample_submission.csv')
    parser.add_argument('--weights_path', type=str, default='./output/')

    return parser.parse_args()


def cross_validation(full_set, p=0.8):
    """
    hold out cross validation
    """
    train_len = len(full_set)

    # get shuffled indices
    indices = np.random.permutation(range(train_len))
    split_idx = int(train_len * p)

    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]
    full_set = np.array(full_set)
    train_set = full_set[list(SubsetRandomSampler(train_idx))]
    valid_set = full_set[list(SubsetRandomSampler(valid_idx))]
    train_set = torch.from_numpy(train_set)
    valid_set = torch.from_numpy(valid_set)
    return train_set, valid_set