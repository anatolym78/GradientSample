import torch
from PIL import Image
import math
from torchvision.transforms import v2

import csv
import random


class GradientImagesCreator:
    def __init__(self):
        pass

    #@staticmethod
    def make_database(self, directory, image_size):
        random.seed(42)
        image_margin = 0.25
        step = 4
        start = image_margin*image_size
        end = (1 - image_margin)*image_size
        values = torch.linspace(start, end, step)/image_size
        count_coordinates = step*step
        x, y = torch.meshgrid(values, values)
        x_coordinates = torch.reshape(x.flatten(), (count_coordinates, 1))
        y_coordinates = torch.reshape(y.flatten(), (count_coordinates, 1))
        to_pil = v2.ToPILImage()
        positions = torch.cat((x_coordinates, y_coordinates), dim=1)
        data = [['path', 'label']]
        i = 0
        for angle in range(-90, 90):
            print(f"angle: {angle}")
            for ipos in range(0, 10):
                pos_x = random.uniform(0.40, 0.60)
                pos_y = random.uniform(0.40, 0.60)
                pos = torch.tensor([pos_x, pos_y])
                for n in range(0, 10):
                    stretch = random.uniform(0.25, 0.75)
                    tensor, label = self.create_sample(image_size, pos, stretch, angle)
                    image = to_pil(tensor)
                    path = f"{directory}/{i}.png"
                    image.save(path)
                    data.append([path, f"{label.item()}"])
                    i += 1

        with open(f"{directory}/data.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)

    # image_size - ширина и высота изображения (квадрат)
    # gradient_center - положение точки через которую проходит центр градиента (в нормализованных координатах - (0, 1))
    # stretch - ширина градиента в процентах от размера изображения
    # angle - угол в градусах
    @staticmethod
    def create_sample(image_size: object, gradient_center: object, stretch: object, angle: object) -> object:
        max_gradient_length = 0.5 * image_size * stretch
        gradient_image = Image.new("L", [image_size, image_size], 96)
        center = image_size*gradient_center
        ex = math.cos(math.radians(angle))
        ey = math.sin(math.radians(angle))
        direction = torch.tensor([ex, ey])
        pixels = gradient_image.load()
        for y in range(0, image_size):
            for x in range(0, image_size):
                pixel_pos = torch.tensor([x, y])
                dp = pixel_pos - center
                dl = torch.dot(dp, direction).item()
                k = math.fabs(dl/max_gradient_length)

                if k > 1:
                    k = 1

                value = int(255*(1 - k))

                byte_value = value.to_bytes(1, "big")
                pixels[x, y] = byte_value[0]

        to_tensor = v2.PILToTensor()

        return to_tensor(gradient_image), torch.tensor([angle])