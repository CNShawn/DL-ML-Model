import torch
import torch.nn as nn


class MNIST_Network(nn.Module):
    def __init__(self):
        super(MNIST_Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.fc3 = nn.Linear(7*7*64, 1024)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(1024, 10)
        self.softmax4 = nn.Softmax(dim=1)

    # 前向传播
    def forward(self, input1):
        x = self.conv1(input1)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size()[0], -1)
        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.softmax4(x)
        return x