import torch 
import torch.nn as nn
import torchvision.models as models


class DecoderBlock(nn.Module):
    def __int__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        def foward(self, x, skip=None):
            x = self.up(x)

            if skip is not None:
                x = torch.cat([x, skip], dim=1)

            return self.conv(x)