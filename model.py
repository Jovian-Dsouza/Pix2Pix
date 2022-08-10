import torch 
import torch.nn as nn
import pytorch_lightning as pl

class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                bias=False, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels*2, 
                features[0],
                4, 
                stride=2,
                padding=1,
                padding_mode="reflect"
            ),
            nn.LeakyReLU(0.2)
        )

        self.layers = nn.ModuleList()
        in_channels = features[0]
        for feature in features[1:]:
            self.layers.append(
                DisBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        self.layers.append(
            nn.Conv2d(in_channels, 1, 4, stride=1, padding=1, padding_mode="reflect")
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        return x

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"), #64
            nn.LeakyReLU(0.2),
        )
        self.down1 = GenBlock(features, features * 2, down=True, act="leaky", use_dropout=False) # 128
        self.down2 = GenBlock(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False #256
        )
        self.down3 = GenBlock(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False #512
        )
        self.down4 = GenBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False # 1024
        )
        self.down5 = GenBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False # 1024
        )
        self.down6 = GenBlock(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False #1024
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU() #1024
        )

        self.up1 = GenBlock(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = GenBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = GenBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = GenBlock(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = GenBlock(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = GenBlock(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = GenBlock(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))

def test_dis():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print(preds.shape)

def test_gen():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, features=64)
    preds = model(x)
    print(preds.shape)
if __name__ == '__main__':
    # test_dis()
    test_gen()