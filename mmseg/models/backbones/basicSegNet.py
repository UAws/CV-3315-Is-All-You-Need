from torch import nn

from ..builder import BACKBONES


@BACKBONES.register_module
class BasicSegNet(nn.Module):
    def __init__(self, n_class):
        super(BasicSegNet, self).__init__()
        #stage 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  #1/2
        #stage 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  #1/4
        #stage 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  #1/8
        #stage 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  #1/16
        #stage 5
        self.conv5_1 = nn.Conv2d(512, 2048, 3)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.drop5_1 = nn.Dropout2d()
        self.conv5_2 = nn.Conv2d(2048, 2048, 1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.drop5_2 = nn.Dropout2d()
        self.conv5_3 = nn.Conv2d(2048, n_class, 1)

    def forward(self, x):
        inp_shape = x.shape[2:]
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.pool3(x)

        x = self.relu4_1(self.conv4_1(x))
        x = self.pool4(x)

        x = self.relu5_1(self.conv5_1(x))
        x = self.drop5_1(x)
        x = self.relu5_2(self.conv5_2(x))
        x = self.drop5_2(x)
        x = self.conv5_3(x)

        return x

