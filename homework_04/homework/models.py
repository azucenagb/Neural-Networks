import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class ConvNetModel(nn.Module):
    '''
        Your code for the model that computes classification from the inputs to the scalar value of the label.
        Classification Problem (1) in the assignment
    '''

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 18, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(18, 12, kernel_size = 3, stride = 1, padding = 1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride = 2, padding = 0)
        self.fc1 = nn.Linear(12*16*16, 64)
        self.fc2 = nn.Linear(64, 6)


    def forward(self, x):

        relu = nn.ReLU()
        x = relu(self.conv1(x))
        x = self.pool(x)
        x = relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 12*16*16)
        x = relu(self.fc1(x))
        x = self.fc2(x)

        return x
