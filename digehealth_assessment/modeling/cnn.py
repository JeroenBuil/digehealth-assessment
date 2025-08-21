import torch.nn as nn


class BowelSoundCNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super().__init__()
        c, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(30, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
