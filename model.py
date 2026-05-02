import torch
import torch.nn as nn
from torchvision import models
from config import num_classes

class CustomASLNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super().__init__()
        def conv_block(in_ch, out_ch, kernel=3, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2, 2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(3,   32),   # 224 -> 112
            conv_block(32,  64),   # 112 ->  56
            conv_block(64, 128),   #  56 ->  28
            conv_block(128, 256),  #  28 ->  14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),   # -> 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_mobilenet_model(num_classes=num_classes):
    model_mnv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model_mnv2.features.parameters():
        param.requires_grad = False
    in_features = model_mnv2.classifier[1].in_features
    model_mnv2.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model_mnv2
