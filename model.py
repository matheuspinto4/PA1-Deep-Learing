from typing import Any

import torch 
import torch.nn as nn
import torchvision.models as models


# Criando a classe do Decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # inplace=True otimiza memória 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        return self.conv(x)

# Criando a classe para a Rede U-NET, iremos utilizar como base.
class ModularUNet(nn.Module):
    def __init__(self, num_classes=1) -> None:
        super().__init__()

        # Encoder
        # Iremos utilizar a ResNet pré-treinada e pegar os vetores
        # das skip connections em cada escala
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Extraindo as skip connections:
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        # Decoder
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Head
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
        

    def forward(self, x):
        # Informação sendo passada pela rede

        # Primeiro ela desce 😏
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.enc4(s3)

        # depois ela sobe 😏
        d4 = self.dec4(bottleneck, s3)
        d3 = self.dec3(d4, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        out_features = self.final_up(d1)

        return self.head(out_features)















