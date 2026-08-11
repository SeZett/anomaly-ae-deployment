import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 3), stride=(2, 3)),

            nn.Conv2d(8, 12, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.encoder_fc = nn.Sequential(
            nn.Linear(12 * 25 * 25, 16),
            nn.ReLU()
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(16, 12 * 25 * 25),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 3), mode="nearest"),

            nn.Conv2d(12, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 2), mode="nearest"),

            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(4, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)

        latent = self.encoder_fc(x)

        x = self.decoder_fc(latent)
        x = x.view(-1, 12, 25, 25)

        x = self.decoder_conv(x)

        return x
