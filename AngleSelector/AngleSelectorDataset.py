import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

from SimpleModule import SimpleModule
from GradientImagesCreator import GradientImagesCreator

from Models.ModelsLoader import ModelsLoader


class AnglesPair:
    def __init__(self):
        self.predicts = []
        self.real_angles = []
        pass

    def add_pair(self, real_angle, predicts):
        self.real_angles.append(real_angle)
        self.predicts.append(predicts)


class AngleSelectorDataset(Dataset):
    def __init__(self):
        models = ModelsLoader()
        image_creator = GradientImagesCreator()
        pairs = AnglesPair()
        for angle in range(-90, 90):
            image, angle = image_creator.create_sample(32, torch.tensor([0.5, 0.5]), 0.6, angle)
            angles = []
            for model in models:
                angles.append(angle/90.0, model(image))

            pairs.add_pair(angles)

        pass

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return 0

