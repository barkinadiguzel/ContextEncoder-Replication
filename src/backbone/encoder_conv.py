import torch
import torch.nn as nn

class EncoderConv(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4000):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, latent_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)
