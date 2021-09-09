import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import vgg19
from matplotlib import pyplot as plt

from dataset import DogDataset
from models.model import VGG19_1, VGG19_2, MyCNN, Densenet, ResNet
from utils import argument_setting


def test(args):

    submit_csv = pd.read_csv(args.submit_csv, header=0)
    outputs_list = []

    # dataset
    full_set = DogDataset(args.test_path, submit_csv)

    # choose training device
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.model == "VGG19_1":
        model = VGG19_1()
    elif args.model == "VGG19_2":
        model = VGG19_2()
    elif args.model == "MyCNN":
        model = MyCNN()
    elif args.model == "ResNet":
        model = ResNet()
    elif args.model == "Densenet":
        model = Densenet()
    else:
        model = vgg19(pretrained=True)

    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.weights_path, 'model_weights.pth')))

    # set dataloader
    dataloader = DataLoader(
        full_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    # start to evaluate
    model.eval()
    with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Evaluation') as t, torch.set_grad_enabled(False):
        for _, inputs in t:
            inputs = inputs.to(device)

            # forward
            outputs = model(inputs)
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = outputs.data.cpu().numpy()[:, 1].tolist()
            outputs_list = outputs_list + outputs

    print("\nFinished Evaluating")

    submit_csv['label'] = outputs_list
    submit_csv.to_csv(os.path.join(args.weights_path, 'answer.csv'), index=False)

if __name__ == '__main__':

    args = argument_setting()

    test(args)
