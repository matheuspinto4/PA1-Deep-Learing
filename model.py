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
    def __init__(self, num_classes=1, decoder_type="skip") -> None:
        super().__init__()
        self.decoder_type = decoder_type

        if decoder_type == "skip":
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif decoder_type == "atrous_aspp":
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            _make_dilated(resnet.layer3, dilation=2)
            _make_dilated(resnet.layer4, dilation=4)
        else:
            raise ValueError(f"decoder_type desconhecido: {decoder_type}")

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4

        if decoder_type == "skip":
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
            self.head = nn.Conv2d(32, num_classes, kernel_size=1)

        elif decoder_type == "atrous_aspp":
            self.aspp = ASPP(in_channels=512, out_channels=256)
            self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.enc4(s3)

        if self.decoder_type == "skip":
            d4 = self.dec4(bottleneck, s3)
            d3 = self.dec3(d4, s2)
            d2 = self.dec2(d3, s1)
            d1 = self.dec1(d2, s0)
            out_features = self.final_up(d1)
            return self.head(out_features)

        elif self.decoder_type == "atrous_aspp":
            features = self.aspp(bottleneck)
            logits = self.head(features)
            return nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def _make_dilated(layer, dilation):
    """
    Converte um layer do ResNet (layer3 ou layer4) para nao reduzir mais
    a resolucao espacial: troca o stride do primeiro bloco por dilatacao,
    e dilata as convolucoes 3x3 de todos os blocos do estagio (slide 40).
    Mexemos nos atributos depois de construir os blocos porque o
    BasicBlock do torchvision (usado no resnet34) nao aceita
    "dilation > 1" no construtor.
    """
    first_block = layer[0]

    first_block.conv1.stride = (1, 1)
    if first_block.downsample is not None:
        first_block.downsample[0].stride = (1, 1)

    for block in layer:
        for conv in (block.conv1, block.conv2):
            if conv.kernel_size[0] > 1:
                conv.dilation = (dilation, dilation)
                conv.padding = (dilation, dilation)

    return layer


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (DeepLabv3, slide 57): convolucoes
    atrous em paralelo com taxas diferentes, mais um ramo de image
    pooling (contexto global), concatenados e fundidos por uma conv 1x1.
    """
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()

        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.atrous_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for rate in rates
        ])

        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        num_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * num_branches, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]

        branches = [self.branch_1x1(x)]
        branches += [branch(x) for branch in self.atrous_branches]

        pooled = self.image_pooling(x)
        pooled = nn.functional.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        branches.append(pooled)

        return self.project(torch.cat(branches, dim=1))


if __name__ == "__main__":
    dummy_input = torch.randn(2, 3, 128, 128)

    model_skip = ModularUNet(num_classes=3, decoder_type="skip")
    out_skip = model_skip(dummy_input)
    print(f"decoder_type='skip'        -> saida: {tuple(out_skip.shape)}")

    model_aspp = ModularUNet(num_classes=3, decoder_type="atrous_aspp")
    out_aspp = model_aspp(dummy_input)
    print(f"decoder_type='atrous_aspp' -> saida: {tuple(out_aspp.shape)}")

    assert out_skip.shape == out_aspp.shape == (2, 3, 128, 128)
    print("OK: as duas variantes produzem saida do mesmo tamanho da entrada.")







