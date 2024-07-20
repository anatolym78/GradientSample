import torch
import torch.nn
import torchvision.io
from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas
from PIL import Image


class GradientDataset(Dataset):
    def __init__(self, records, input_transforms = None):
        self.records = records
        self.input_transforms = input_transforms
        pass

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, string_label = self.records[index]
        angle_normalized = float(string_label)/90.0
        output_label = torch.tensor([angle_normalized])

        with Image.open(path) as image:
            image.load()
        to_tensor = v2.PILToTensor()
        tensor_image = to_tensor(image)
        tensor_image = self.input_transforms(tensor_image)
        return tensor_image, output_label
