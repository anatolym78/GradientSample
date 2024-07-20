import torch
import torch.nn


class GradientModule(torch.nn.Module):
    def __init__(self):
        super(GradientModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5) #28x28
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2) # 14x14
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5) # 10x10
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2) # 5x5
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=180, kernel_size=5) #1x2
        self.relu3 = torch.nn.ReLU()
        #self.pool3 = torch.nn.MaxPool2d(kernel_size=3) #3x3

        self.flatten = torch.nn.Flatten() #9*12

        self.linear1 = torch.nn.Linear(in_features=180, out_features=180)
        self.relu4 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=180, out_features=1)

    def forward(self, tensor_image):
        x = self.conv1(tensor_image)
        #x = torch.tanh(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        #x = torch.tanh(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        #x = torch.tanh(x)
        x = self.relu3(x)
        # x = self.relu3(x)
        # x = self.pool3(x)

        f = self.flatten(x)
        f = torch.transpose(f, 0, 1)
        l1 = self.linear1(f)
        l1 = self.relu4(l1)
        r1 = self.linear2(l1)

        result = torch.tanh(r1)
        result = r1

        return result

