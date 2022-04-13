import torch.nn as nn


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Omninet(nn.Module):

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, 2*hid_dim),
            conv_block(2*hid_dim, 2*hid_dim),
            conv_block(2*hid_dim, z_dim),
        )        

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class Omni_Classifier(nn.Module):
    """Feature classifier class for MNIST -> MNIST-M experiment in ATDA."""

    def __init__(self, num_classes=80):
        """Init classifier."""
        super(Omni_Classifier, self).__init__()
        self.restored = False

        self.classifier = nn.Sequential(
                nn.Linear(576, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        """Forward classifier."""
        out = self.classifier(x)
        return out