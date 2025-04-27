import torch.nn as nn
from torch.nn import Conv2d, Linear
from torch.nn.functional import relu, max_pool2d


class CookingNet(nn.Module):
    """
    Convolutional Neural Network for image classification.

    This network consists of two convolutional layers followed by batch normalization,
    max pooling, and two fully connected layers with dropout for regularization.
    """
    def __init__(self):
        super().__init__()
        self.num_labels = 2
        self.conv1 = Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)

        fc1_input_size = 64
        self.fc1 = Linear(fc1_input_size, 512)
        self.fc2 = Linear(512, self.num_labels)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(max_pool2d(x, kernel_size=2, stride=2, padding=0))

        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(max_pool2d(x, kernel_size=2, stride=2, padding=0))

        x = x.view(x.size(0), -1)

        x = relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x
