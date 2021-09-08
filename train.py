import os
import time
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import vgg19
from torchsummary import summary
from matplotlib import pyplot as plt

from dataset import CatDataset, cross_validation

def train(args):

    # get state name
    t = time.localtime()
    run_time = time.strftime("%Y_%m_%d_%H_%M_%S", t)
    state_name = f"{args.model}_{args.optim}_{args.epochs}_{args.lr}_{args.batch_size}_{run_time}"
    save_path = os.path.join(args.output_path, state_name)

    # dataset
    full_set = CatDataset(args.train_path)

    # use torch random_split to create the validation dataset
    lengths = [int(round(len(full_set) * args.holdout_p)), int(round(len(full_set) * (1 - args.holdout_p)))]
    train_set, valid_set = random_split(full_set, lengths)

    # build hold out CV
    # train_set, valid_set = cross_validation(full_set, args.holdout_p)

    # load model
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    if args.model == "VGG":
        model = vgg19(pretrained=True)
    else:
        model = vgg19(pretrained=True)

    # 把參數凍結
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    if args.model == "VGG19":
        model.classifier[3] = nn.Linear(4096, 4096)
        model.classifier[6] = nn.Linear(4096, 2)

    params_to_update = []
    for _,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    model = model.to(device)

    # set optimizer
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(params_to_update, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum)

    # set loss function
    criterion = nn.BCELoss()

    # train
    loss_list = {'train':[], 'valid':[]}
    accuracy_list = {'train':[], 'valid':[]}
    dataloader={'train':None, 'valid':None}
    best = 100

    # set dataloader
    dataloader['train'] = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                            #   sampler=train_sampler,
                              pin_memory=False,)

    dataloader['valid'] = DataLoader(valid_set,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                            #   sampler=valid_sampler,
                              pin_memory=False,)

    # check output path exist
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # start to train
    for epoch in range(args.epochs):
        for phase in ['train', 'valid']:

            epoch_loss = 0.0
            correct = 0

            with tqdm(enumerate(dataloader[phase]), desc=f'{epoch}/{args.epochs}, {phase}') as t, torch.set_grad_enabled(phase=='train'):
                for _, data in t:
                    inputs, targets = data[0].to(device), data[1].to(device)

                    # forward
                    outputs = model(inputs)
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs.data, 1)
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs[:,1], targets)

                    # backward
                    if phase == 'train':
                        optimizer.zero_grad() # 清空上一輪算的 gradient
                        loss.backward()       # 計算 gradient
                        optimizer.step()      # 更新參數

                    epoch_loss += loss.item() * inputs.data.size(0)
                    correct += torch.sum(preds == targets.data)

            epoch_loss /= len(dataloader[phase].dataset)
            loss_list[phase].append(epoch_loss)
            accuracy = float(correct) / len(dataloader[phase].dataset)
            accuracy_list[phase].append(accuracy)

            if phase == 'valid' and epoch_loss < best:
                best = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pth'))

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {loss_list['train'][-1]:.4f}, Validation Loss: {loss_list['valid'][-1]:.4f}")
        print(f"Train Accuracy: {accuracy_list['train'][-1]:.4f}, Validation Accuracy: {accuracy_list['valid'][-1]:.4f}\n")

    print("\nFinished Training")

     # plot the loss curve for training and validation
    pd.DataFrame({
        "train-loss": loss_list['train'],
        "valid-loss": loss_list['valid']
    }).plot()
    plt.xlabel("Epoch"),plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "Loss_curve.jpg"))

    # plot the accuracy curve for training and validation
    pd.DataFrame({
        "train-accuracy": accuracy_list['train'],
        "valid-accuracy": accuracy_list['valid']
    }).plot()
    plt.xlabel("Epoch"),plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_path, "Training_accuracy.jpg"))

def argument_setting():

    parser = ArgumentParser()

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='VGG19')
    parser.add_argument('--holdout_p', type=float, default=0.8)
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--train_path', type=str, default='./data/train/')
    parser.add_argument('--test_path', type=str, default='./data/test1/')

    return parser.parse_args()

if __name__ == '__main__':

    args = argument_setting()

    train(args)
