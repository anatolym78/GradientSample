import torch
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.data import dataset as ds
from AngleSelectorDataset import AngleSelectorDataset
from AngleSelectorModel import AngleSelectorModule
from torch.utils.data import DataLoader

class AngleSelectorTrainer:
    def __init__(self):
        dataset = AngleSelectorDataset()

        k = 0.8
        train_len = int(dataset.__len__()*0.8)
        val_len = dataset.__len__() - train_len
        train_subset, val_subset = ds.random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_subset, 16, shuffle=True)
        val_loader = DataLoader(val_subset, 16)

        model = AngleSelectorModule()
        n_epoch = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = torch.nn.BCELoss()

        for epoch in range(0, n_epoch):
            break

        pass