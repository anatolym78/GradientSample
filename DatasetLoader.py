import torch
import torchvision
import pandas
from torchvision.transforms import v2

from PIL import Image
class DatasetLoader:
    def __init__(self, database_path):
        self.source_data = pandas.read_csv(database_path).values.tolist()
        self.data = []
        to_tensor = v2.PILToTensor()
        for pair in self.source_data:
            image_path = pair[0]
            with Image.open(image_path) as image:
                image.load()
            #tensor_image = torchvision.io.read_image(image_path)
            tensor_image = to_tensor(image)

            string_label = pair[1]
            items = str.split(string_label, ',')
            sin_value = torch.tensor([float(items[1])])
            sigmoid_value = (sin_value + 1)*0.5

            self.data.append((tensor_image, sigmoid_value))

    def get_data(self):
        return self.data
