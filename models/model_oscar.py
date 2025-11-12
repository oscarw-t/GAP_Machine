import torch
import torch.nn as nn

# residual block with skip connection for better gradient flow
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        # identity shortcut or projection shortcut when shapes differ
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.activation(self.main_path(x) + self.shortcut(x))


# channel attention only (lighter than full cbam)
class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction=16):
        super().__init__()
        hidden_dim = in_channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, hidden_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # squeeze feature maps into vectors via pooling
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)
        # shared network for both
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return self.sigmoid(attn).unsqueeze(-1).unsqueeze(-1) * x


class leaf_dr(nn.Module):
    def __init__(self, num_classes=39):
        super(leaf_dr, self).__init__()

        # initial conv block with light pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 4 hierarchical residual blocks with channel attention and downsampling
        self.block2 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            ChannelAttention(64)
        )
        self.block3 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ChannelAttention(128)
        )
        self.block4 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ChannelAttention(256)
        )
        self.block5 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ChannelAttention(512)
        )

        # global avg pool maps 512 feature channels to 1×1 each
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # classifier with lighter dropout to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # pass through hierarchical blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # convert features to vector
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        # final prediction
        return self.classifier(x)
