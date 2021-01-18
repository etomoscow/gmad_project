import sys

import torch
from PIL import ImageFile
from torchvision import datasets, transforms

sys.path.append('../')

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(data_folder, batch_size, phase='train', train_val_split=True):
    transform_dict = {
        'train': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                ),
            ]
        ),
        'test': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
            ])}
    data_train = datasets.ImageFolder(
        root=data_folder + '/train',
        transform=transform_dict['train']
    )
    data_val = datasets.ImageFolder(
        root=data_folder + '/test',
        transform=transform_dict['test'])
    data_test = datasets.ImageFolder(
        root=data_folder + '/test',
        transform=transform_dict['test'])

    if phase == 'train':
        if train_val_split:
            train_loader = torch.utils.data.DataLoader(
                data_train,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0
            )
            val_loader = torch.utils.data.DataLoader(
                data_val,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                num_workers=0
            )
            return train_loader, val_loader
        else:
            train_loader = torch.utils.data.DataLoader(
                data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0
            )
            return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            data_test,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        return test_loader
