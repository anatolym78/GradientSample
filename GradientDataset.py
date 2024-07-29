import torch
import torch.nn
import torchvision.io
from torch.utils.data import Dataset
from torchvision.transforms import v2
import pandas
from PIL import Image


class GradientDataset(Dataset):
    def __init__(self, records, horizontal_flag, input_transforms=None):
        self.horizontal_flag = horizontal_flag
        if horizontal_flag:
            self.records = [r for r in records if GradientDataset.is_horizontal(r[1])]
        else:
            self.records = [r for r in records if GradientDataset.is_vertical(r[1])]

        self.input_transforms = input_transforms

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, string_label = self.records[index]

        angle = float(string_label)
        if not self.horizontal_flag:
            if angle < 0:
                angle += 180
            angle *= -1
            angle -= 90

        angle = angle/90

        output_label = torch.tensor([angle])

        with Image.open(path) as image:
            image.load()
        to_tensor = v2.PILToTensor()
        tensor_image = to_tensor(image)
        tensor_image = self.input_transforms(tensor_image)
        return tensor_image, output_label

    @staticmethod
    def is_horizontal(angle_string, thresh=75):
        angle = float(angle_string)
        return thresh > angle > -thresh

    @staticmethod
    def is_vertical(angle_string, thresh=15):
        angle = float(angle_string)
        if angle < 0:
            angle += 180

        return thresh < angle < 180 - thresh

