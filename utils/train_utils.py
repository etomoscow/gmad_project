import numpy as np
import torch


def train(model: torch.nn.Module, iterator: torch.utils.data.DataLoader, criterion, optimizer: torch.optim.Optimizer,
          device: str):
    epoch_loss, epoch_acc = 0, []
    model.train()

    for batch in iterator:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, labels.reshape((32,)).long())
        correct, incorrect = 0, 0
        for p in range(inputs.size(0)):
            a = []
            for j in preds[p]:
                a.append(float(j.detach()))
            pred = a.index(max(a))
            if pred == int(labels[p]):
                correct = correct + 1
            else:
                incorrect = incorrect + 1
        accuracy = correct / (correct + incorrect)
        epoch_acc.append(accuracy)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), np.mean(epoch_acc)


def evaluate(model: torch.nn.Module, iterator: torch.utils.data.DataLoader, criterion, device: str):
    epoch_loss, epoch_acc = 0, []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            preds = model(inputs)
            loss = criterion(preds, labels)
            correct, incorrect = 0, 0
            for p in range(inputs.size(0)):
                a = []
                for j in preds[p]:
                    a.append(float(j.detach()))
                pred = a.index(max(a))
                if pred == int(labels[p]):
                    correct = correct + 1
                else:
                    incorrect = incorrect + 1
            epoch_loss += loss.item()
            accuracy = correct / (correct + incorrect)
            epoch_acc.append(accuracy)
    return epoch_loss / len(iterator), np.mean(epoch_acc)


def epoch_time(start_time, end_time):
    '''
    compute elapsed time for training
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
