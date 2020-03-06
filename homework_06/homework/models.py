import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class Block(nn.Module):
    '''
    Your code for resnet blocks
    '''

    def __init__(self, in_channel, bottle_channel, out_channel, stride):
        super(Block, self).__init__()
        '''
        Your code here
        '''
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.conv3 = nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.conv4 = nn.Conv2d(in_channel, out_channel, kernel_size=5, bias = False)

    def forward(self, x):
        '''
        Your code here
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        return out




class ConvNetModel(nn.Module):
    '''
    Your code for the model that computes classification from the inputs to the scalar value of the label.
    Classification Problem (1) in the assignment
    '''

    def __init__(self):
        super(ConvNetModel, self).__init__()
        '''
        Your code here
        '''
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer2 = Block(32,32,64,1)
        self.layer3 = Block(64,64,128,1)
        self.layer4 = Block(128,128,256,1)
        self.layer5 = Block(256,256,512,1)
        self.fc = nn.Linear(115200, 6)

        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        Your code here
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
