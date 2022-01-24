# The proposed architecture
import torch
import torch.nn as nn


class TerGat(nn.Module):
    def __init__(self, latent_dim=100):

        super(TerGat, self).__init__()
        self.lower_conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.lower_fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, latent_dim)
        )
        self.upper_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.upper_fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(16384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, latent_dim)
        )

        self.final_fc = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(2*latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 7)
        )

    def forward(self, lower_block, upper_block):
        """Perform forward."""

        # conv layers
        lower_x = self.lower_conv_layer(lower_block)

        # flatten
        lower_x = lower_x.view(lower_x.size(0), -1)

        # fc layer
        lower_x = self.lower_fc_layer(lower_x)
        # conv layers
        upper_x = self.upper_conv_layer(upper_block)
        # flatten
        upper_x = upper_x.view(upper_x.size(0), -1)

        # fc layer
        upper_x = self.upper_fc_layer(upper_x)

        final = self.final_fc(torch.cat((lower_x, upper_x), axis=1))
        return final
