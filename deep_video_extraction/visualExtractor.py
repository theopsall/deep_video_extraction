import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets


class VisualExtractor(nn.Module):
    def __init__(self):
        super(VisualExtractor, self).__init__()
        self._rn18 = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self._rn18.children())[:-1])

    def forward(self, x):
        return self.model(x)


def main():
    transform = transforms.Compose(
        [transforms.ToTensor()])

    test = datasets.CIFAR10(root='./data/cifar_test',
                            download=True,
                            transform=transform,
                            train=False)
    testLoader = DataLoader(test, batch_size=2, shuffle=False,  num_workers=2)
    vs = VisualExtractor()

    for i, data in enumerate(testLoader):
        images, labels = data
        predicted = vs(images)
        print(predicted[0])
        print(predicted[0].flatten().shape)

        print(predicted[0].shape)
        break


if __name__ == "__main__":
    main()
