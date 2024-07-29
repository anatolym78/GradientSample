import torch
from torchvision.transforms import v2
from torch.nn import Module


class AngleSelectorModule(Module):
    def __init__(self):
        super(AngleSelectorModule, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=2, out_features=2)
        self.relu = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(in_features=2, out_features=1)

    def forward(self, angles_predict):
        x = self.linear1(angles_predict)
        x = self.relu(x)
        x = self.linear2(x)
        return torch.tanh(x)