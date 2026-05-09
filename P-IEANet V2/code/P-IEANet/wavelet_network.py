import torch
from FFA import FFA
import torch.nn as nn
from structure_model import Encoder_block

from torchvision.transforms import Resize
torch_resize = Resize([256,256])

class Low(nn.Module):
    def __init__(self):
        super(Low, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.conv.weight.data.fill_(0.25)

    def forward(self, x):
        x = self.conv(x)
        return x

class High_1(nn.Module):
    def __init__(self):
        super(High_1, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.conv.weight.data.fill_(0.25)
        self.conv.weight.data[0, 0, 1, :] = -0.25

    def forward(self, x):
        x = self.conv(x)
        return x

class High_2(nn.Module):
    def __init__(self):
        super(High_2, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.conv.weight.data.fill_(0.25)
        self.conv.weight.data[0, 0, :, 1] = -0.25

    def forward(self, x):
        x = self.conv(x)
        return x

class High_3(nn.Module):
    def __init__(self):
        super(High_3, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=2, stride=2)
        self.conv.weight.data.fill_(0.25)
        self.conv.weight.data[0, 0, 1, 0] = -0.25
        self.conv.weight.data[0, 0, 0, 1] = -0.25

    def forward(self, x):
        x = self.conv(x)
        return x

class Fusion(nn.Module):
    def __init__(self, in_channels=2, out_channels=9):
        super(Fusion, self).__init__()
        self.blur_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.tail_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x, blur):
        x = torch_resize(x).to(blur.device)
        x = self.tail_conv(x)
        blur = torch.unsqueeze(blur, dim=1)
        blur = blur + x

        return blur

class Tail(nn.Module):
    def __init__(self):
        super(Tail, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

class Wavelet_Net(nn.Module):
    def __init__(self, ):
        super(Wavelet_Net, self).__init__()

        self.ffa_group = FFA(gps=3, blocks=2)
        self.encoder = Encoder_block(in_channel=9, out_channel=3)

        self.Low = Low()
        self.High_1 = High_1()
        self.High_2 = High_2()
        self.High_3 = High_3()

        self.fusion = Fusion()



    def forward(self, x):
        low = self.Low(x)
        high_1 = self.High_1(x)
        high_2 = self.High_2(x)
        high_3 = self.High_3(x)

        blur = self.ffa_group(x)

        high = torch.cat([high_1, high_2, high_3], dim=1)
        high = self.encoder(high)

        result = self.fusion(high,blur)


        return result

