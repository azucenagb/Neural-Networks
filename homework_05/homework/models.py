import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class ConvNetModel(nn.Module):


    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 2, 1)
        self.fc = nn.Linear(6272, 6)
        self.relu = nn.ReLU(True)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(-1, 6272)
        x = self.fc(x)
        return x

        return x
