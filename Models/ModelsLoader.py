import torch

from torchvision.transforms import v2

from SimpleModule import SimpleModule


class ModelsLoader:
    def __init__(self):
        self.models = []
        for i_model in range(0, 2):
            check_point = torch.load(f"model_state_{i_model + 1}.pth")
            model = SimpleModule()
            model.load_state_dict(check_point["model_state"])
            model.eval()
            self.models.append(model)

    def model(self, index: int):
        return self.models[index]
        pass
