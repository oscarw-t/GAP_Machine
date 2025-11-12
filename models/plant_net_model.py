import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        pooled = torch.cat([max_pool, avg_pool], dim=1)

        spatial_weights = self.sigmoid(self.conv(pooled))

        return spatial_weights * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        hidden_dim = in_channel // reduction

        self.mlp = nn.Sequential(
            nn.Linear(in_channel, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)
        max_pool = max_pool.view(max_pool.size(0), -1)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        mlp_max = self.mlp(max_pool)
        mlp_avg = self.mlp(avg_pool)
        channel_weights = self.sigmoid(mlp_max + mlp_avg)
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)

        return channel_weights * x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, padding=1):
        super().__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channel),
        )

        # Shortcut
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )

        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        main = self.main_path(x)
        skip = self.shortcut(x)
        out = main + skip
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ClassificationHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, out_features),
        )

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PlantNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, dropout=0.3):
        super().__init__()

        self.block_1 = ConvBlock(in_channels, 32)
        self.block_2 = nn.Sequential(ResidualBlock(32, 64, stride=2), CBAM(64))
        self.block_3 = nn.Sequential(ResidualBlock(64, 128, stride=2), CBAM(128))
        self.block_4 = nn.Sequential(ResidualBlock(128, 256, stride=2), CBAM(256))
        self.block_5 = nn.Sequential(ResidualBlock(256, 512, stride=2), CBAM(512))

        self.classification_head = ClassificationHead(512, num_classes, dropout=dropout)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        logits = self.classification_head(x)

        return logits
