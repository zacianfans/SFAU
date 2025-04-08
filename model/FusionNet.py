import torch
from torch import nn
from torchsummary import summary
from model.PGCU import PGCU
from model.SFAU import SFAU

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class FusionNet(nn.Module):
    def __init__(self, spectral_num, criterion, channel=32):
        super(FusionNet, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num
        self.criterion = criterion

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        # self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output  # lms + outs


class FusionNet_PGCU(nn.Module):
    def __init__(self, spectral_num, criterion=None, channel=32):
        super(FusionNet_PGCU, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num
        self.criterion = criterion

        self.PGCU = PGCU(4, 128)  # pan, ms
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        # self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan
        up_x = self.PGCU(y, x) # pan, ms
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, up_x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output, up_x  # lms + outs


class FusionNet_SANM(nn.Module):
    def __init__(self, spectral_num, criterion=None, channel=32):
        super(FusionNet_SANM, self).__init__()
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.spectral_num = spectral_num
        self.criterion = criterion

        self.SANM = SFAU(1, 4)  # ms, pan
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!
        # self.apply(init_weights)

    def forward(self, x, y):  # x= lms; y = pan
        up_x, _, _ = self.SANM(x, y) # ms, pan
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, up_x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64

        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64

        return output, up_x  # lms + outs


if __name__ == '__main__':
    pan = torch.ones((32, 1, 128, 128))
    ms = torch.ones((32, 4, 32, 32))
    Model = FusionNet_SANM(4)
    output1, output2 = Model.forward(ms, pan)
    summary(Model, input_size=[(4, 32, 32), (1, 128, 128)], device='cpu')
    print(output1.shape, output2.shape)