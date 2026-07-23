
import torch
import torch.nn as nn

class ROIPolarimetricRefinerV2(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(channels,32,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(32,1)

    def forward(self,x):
        x=self.feature(x)
        return self.head(x.flatten(1))
