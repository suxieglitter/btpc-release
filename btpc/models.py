"""Neural network architectures: P-wave encoder, Barlow Twins wrapper and the
pseudo-label classification head used in Stage 2."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import device


class Residual1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            if use_1x1conv
            else None
        )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)


class AttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class PWaveEncoder(nn.Module):
    """1-D residual encoder with squeeze-and-excitation style attention."""

    def __init__(self, input_channels=1, base_channels=16, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.layer0 = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_channels, base_channels, 1, first_block=True)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 1)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 1)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 1)
        if use_attention:
            self.attention = AttentionModule(base_channels * 8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = base_channels * 8

    def _make_layer(self, in_channels, out_channels, blocks, first_block=False):
        layers = []
        for i in range(blocks):
            if i == 0 and not first_block:
                layers.append(Residual1D(in_channels, out_channels, stride=2, use_1x1conv=True))
            else:
                layers.append(Residual1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


class BarlowTwins(nn.Module):
    """Encoder + projector with the Barlow Twins cross-correlation loss."""

    def __init__(self, encoder, projector_dims=(512, 512, 512)):
        super().__init__()
        self.encoder = encoder
        feature_dim = encoder.feature_dim
        projector_layers = []
        prev_dim = feature_dim
        for dim in projector_dims[:-1]:
            projector_layers.extend([nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.ReLU()])
            prev_dim = dim
        projector_layers.append(nn.Linear(prev_dim, projector_dims[-1]))
        self.projector = nn.Sequential(*projector_layers)
        self.bn = nn.BatchNorm1d(projector_dims[-1], affine=False)

    def forward(self, y):
        z = self.encoder(y)
        z = self.projector(z)
        z = self.bn(z)
        return z

    def compute_loss(self, z1, z2):
        batch_size_local = z1.size(0)
        c = torch.mm(z1.T, z2) / batch_size_local
        on_diag_loss = (torch.diagonal(c) - 1).pow(2).sum()
        off_diag_loss = self.off_diagonal(c).pow(2).sum()
        return on_diag_loss, off_diag_loss

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class ClusterClassificationHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 0, num_classes: int = 2):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)


class PseudoLabelClassifier(nn.Module):
    """Frozen-encoder classifier trained on Stage 2 pseudo labels."""

    def __init__(self, encoder: PWaveEncoder, head_hidden_dim: int = 0, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head_hidden_dim = int(head_hidden_dim)
        self.head = ClusterClassificationHead(
            input_dim=encoder.feature_dim,
            hidden_dim=self.head_hidden_dim,
            num_classes=num_classes,
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)


def build_barlow_model(base_channels=16, projector_dims=512):
    """Build the Stage 1 Barlow Twins model on the default torch device."""
    encoder = PWaveEncoder(input_channels=1, base_channels=base_channels, use_attention=True)
    dims = list(projector_dims) if isinstance(projector_dims, (list, tuple)) else [projector_dims] * 3
    return BarlowTwins(encoder=encoder, projector_dims=dims).to(device)


def build_stage2_classifier(base_channels=16, head_hidden_dim=0):
    """Build the Stage 2 pseudo-label classifier on the default torch device."""
    encoder = PWaveEncoder(input_channels=1, base_channels=base_channels, use_attention=True)
    return PseudoLabelClassifier(encoder=encoder, head_hidden_dim=head_hidden_dim).to(device)
