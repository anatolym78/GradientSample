import torch
import torch.utils.data
import torch.nn
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler, SequentialSampler
import pandas
import sklearn.preprocessing
import sklearn
import sklearn.linear_model
import sklearn.model_selection
from torch.utils.data import dataset as ds
from GradientDataset import GradientDataset
from GradientModule import GradientModule
from SimpleModule import SimpleModule
import torchvision
from torchvision.transforms import  v2
from torch.optim import optimizer
import torch.nn as nn


class GradientTrainer:
    def __init__(self, dataset_file_path):
        num_model = 1
        records = pandas.read_csv(dataset_file_path).values.tolist()
        for is_horizontal in [True]:
            print(f"Train model {num_model}")
            num_model += 1

            # create dataset
            transform = v2.Compose([v2.ToDtype(torch.float32, scale=False), v2.Normalize([128], [128])])
            dataset = GradientDataset(records, is_horizontal, input_transforms=transform)

            # split dataset
            ratio = 0.8
            train_data_len = int(ratio*dataset.__len__())
            val_data_len = dataset.__len__() - train_data_len
            train_data, val_data = ds.random_split(dataset, [train_data_len, val_data_len])

            # create data loaders
            batch_size = 32
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size)

            model = SimpleModule()

            optimiser = torch.optim.SGD(model.parameters(), lr=0.1)
            loss_fn = torch.nn.MSELoss()

            device = torch.device("cuda")
            model.to(device)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.995)

            n_epoch = 20
            epoch_losses = []
            val_losses = []
            for n in range(0, n_epoch):
                model.train()

                batch_train_losses = []
                count_batches = 0
                for image, label in train_loader:
                    try:
                        image = image.to(device)
                        label = label.to(device)
                        label_predict = model(image)
                        loss = loss_fn(label_predict, label)
                        loss.backward()
                        optimiser.step()
                        optimiser.zero_grad()
                        batch_train_losses.append(loss.item())
                        count_batches += 1
                    except Exception as e:
                        print(e)

                torch_batch_train_losses = torch.FloatTensor(batch_train_losses)
                train_mean_tensor = torch.mean(torch_batch_train_losses)
                epoch_losses.append(train_mean_tensor)

                model.eval()
                batch_val_losses = []
                with torch.no_grad():
                    for image, label in val_loader:
                        image = image.to(device)
                        label = label.to(device)
                        label_predict = model(image)
                        loss = loss_fn(label_predict, label)
                        batch_val_losses.append(loss.item())

                torch_batch_val_losses = torch.FloatTensor(batch_val_losses)
                val_mean_tensor = torch.mean(torch_batch_val_losses)
                val_losses.append(val_mean_tensor)

                print(f"{n}) Mean train/val: {train_mean_tensor}/{val_mean_tensor}")

                scheduler.step()
                print(optimiser.param_groups[0]["lr"])

            torch.save({"model_state": model.state_dict()}, f"model_state_{num_model}.pth")

            pass

    @staticmethod
    def is_horizontal(self, angle_string, thresh = 75):
        angle = float(angle_string)
        return thresh > angle > -thresh

