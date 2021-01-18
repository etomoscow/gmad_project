import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from utils.prepare_dataset import load_data
from utils.train_utils import train, evaluate, epoch_time

sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', type=float, default=0.0007,
                    help='learning rate (default: 0.0007)')
parser.add_argument('--save_model', default=False,
                    help='specify in order to save finetuned model')
parser.add_argument('--model_type', type=str, default='resnet18',
                    help='specify model type to finetune (default: ResNet18)')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='number of epochs of finetuning (default: 100)')

args = parser.parse_args()

if __name__ == '__main__':

    trainloader, validloader = load_data('../images/', phase='train', batch_size=args.batch_size)
    testloader = load_data('../images/', phase='test', batch_size=args.batch_size)

    if args.model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    if args.model_type == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    if args.model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)

    # we finetune only the last layer
    for param in model.parameters():
        param.requires_grad = False

    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, 2)

    optimizer = optim.Adam(model.parameters(), lr=0.0007)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    os.makedirs('../trained_models', exist_ok=True)

    best_valid_loss = float('inf')
    best_valid_acc = 0.80
    losses = []
    for epoch in range(args.n_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, validloader, criterion, device)
        losses.append(train_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        if args.save_model == True:
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                print('model copied')
                best_model = copy.deepcopy(model)
                model_name = args.model_type + '_best.pt'
                torch.save(best_model.state_dict(), '../trained_models/' + model_name)

        if epoch >= 10 and train_loss >= losses[-1]:
            print('Early stopping')
            break
