import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from torch.utils.data.sampler import SubsetRandomSampler


def argument_setting(inhert=False):

    class Range(object):
        def __init__(self, start, end):
            self.start = start
            self.end = end
        def __eq__(self, other):
            return self.start <= other <= self.end

    parser = ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)

    # dataset argument
    parser.add_argument('--holdout-p', type=float, default=0.8)
    parser.add_argument('--num-workers', type=int, default=8)

    # training argument
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument(
        '--model', type=str,
        choices=['VGG19', 'VGG19_2', 'ResNet', 'MyCNN', 'Densenet'],
        metavar='VGG19, VGG19_2, ResNet, Densenet, MyCNN',
        default='VGG19'
    )
    parser.add_argument('--iteration', action="store_true", default=False)
    parser.add_argument('--train-all', action="store_true", default=False)

    # optimizer argument
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)

    # scheduler argument
    parser.add_argument('--scheduler', action="store_true", default=False)
    parser.add_argument('--gamma', type=float, default=0.99985)

    # post-processing argument
    parser.add_argument(
        '--threshold',
        type=float,
        choices=[Range(0.0, 1.0)],
        metavar='Float number >= 0 and <=1'
    )

    parser.add_argument('--output-path', type=str, default='./output/')
    parser.add_argument('--train-path', type=str, default='./data/train/')
    parser.add_argument('--test-path', type=str, default='./data/test1/')
    parser.add_argument('--submit-csv', type=str, default='./data/sample_submission.csv')
    parser.add_argument('--weights-path', type=str, default='./output/')

    # for the compatiable
    if inhert is True:
        return parser

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


def threshold_function(data, threshold, device='cpu'):

    data = torch.where(data >= threshold, torch.tensor(1.0, dtype=data.dtype).to(device), data)
    data = torch.where(data < (1-threshold), torch.tensor(0.0, dtype=data.dtype).to(device), data)

    return data


def adaptive_threshold(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray_img, 11, 75, 75)
    blur = cv2.GaussianBlur(bilateral, (5, 5), 1)
    adaptive_threshold = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return adaptive_threshold