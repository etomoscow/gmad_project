import argparse
import os
import sys

sys.path.append('../')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score,
                             balanced_accuracy_score,
                             precision_recall_fscore_support)
from torchvision import models

from utils.prepare_dataset import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='resnet18',
                    help='specify model to check performance (available: resnet18, resnet34, resnet50, all)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='specify batch_size (better to be the same as used for training (e.g. 32))')
parser.add_argument('--save_results', type=bool, default=True,
                    help='if true creates dataframe containing results for the model (default False)')
parser.add_argument('--show_metrics', type=bool, default=False,
                    help='if true shows metrics (default False)')

mapping = {'resnet18': models.resnet18(pretrained=False),
           'resnet34': models.resnet34(pretrained=False),
           'resnet50': models.resnet50(pretrained=False),
           'all': [
               models.resnet18(pretrained=False),
               models.resnet34(pretrained=False),
               models.resnet50(pretrained=False)
           ]}

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    testloader = load_data('../images/', phase='test', batch_size=args.batch_size)
    if isinstance(mapping[args.model_type], models.resnet.ResNet):
        model = mapping[args.model_type]
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(
            torch.load('../trained_models/' + args.model_type + '_best.pt', map_location=torch.device('cpu')))
        model.to(device)
        trues, predictions = [], []
        with torch.no_grad():
            for batch in testloader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                preds = model(inputs)
                _, out = torch.max(preds, 1)
                trues.append(labels.cpu().numpy())
                predictions.append(out.cpu().numpy())
        trues = [y for x in trues for y in x]
        predictions = [y for x in predictions for y in x]
        if args.save_results:
            os.mkdir('../results', exist_ok=True)
            results = pd.DataFrame(
                {
                    'accuracy': balanced_accuracy_score(trues, predictions),
                    'ROC-AUC': roc_auc_score(trues, predictions, average='weighted'),
                    'precision': np.mean(precision_recall_fscore_support(trues, predictions, average='weighted')[0]),
                    'recall': np.mean(precision_recall_fscore_support(trues, predictions, average='weighted')[1]),
                    'f1-score': np.mean(precision_recall_fscore_support(trues, predictions, average='weighted')[2])
                },
                index=args.model_type
            )
            results.to_csv('../results/results.csv')
        if args.show_results:
            print(results)


    else:
        accs, rocs, precs, recs, fs = [], [], [], [], []
        model_1 = mapping[args.model_type][0]
        model_2 = mapping[args.model_type][1]
        model_3 = mapping[args.model_type][2]
        for model, path in zip(
                [
                    model_1,
                    model_2,
                    model_3
                ],
                [
                    '../trained_models/resnet18_best.pt',
                    '../trained_models/resnet34_best.pt',
                    '../trained_models/resnet50_best.pt'
                ]):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            model.to(device)
            model.eval()
            trues, predictions = [], []
            with torch.no_grad():
                for batch in testloader:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    preds = model(inputs)
                    _, out = torch.max(preds, 1)
                    trues.append(labels.cpu().numpy())
                    predictions.append(out.cpu().numpy())
            trues = [y for x in trues for y in x]
            predictions = [y for x in predictions for y in x]
            accs.append(balanced_accuracy_score(trues, predictions))
            rocs.append(roc_auc_score(trues, predictions, average='weighted'))
            precs.append(np.mean(precision_recall_fscore_support(trues, predictions, average='weighted')[0]))
            recs.append(np.mean(precision_recall_fscore_support(trues, predictions, average='weighted')[1]))
            fs.append(np.mean(precision_recall_fscore_support(trues, predictions, average='weighted')[2]))
        if args.save_results:
            os.mkdir('../results', exist_ok=True)
            results = pd.DataFrame(
                {
                    'accuracy': accs,
                    'ROC-AUC': rocs,
                    'precision': precs,
                    'recall': recs,
                    'f1-score': fs
                },
                index=['resnet18', 'resnet34', 'resnet50']
            )
            results.to_csv('../results.csv')
        if args.show_results:
            print(results)
