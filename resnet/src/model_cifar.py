import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''
    Basic block for CIFAR-10
    Conv 3x3 -> 3x3 
    When the dimensions increase, option (A) used
    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int,
    ) -> None:
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # 3x3
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=hidden_channels,
        )
        self.relu = nn.ReLU(inplace=True)

        # 3x3
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=hidden_channels,
        )

        self._init_params()

    def _init_params(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.stride != 1:
            identity = identity[:, :, ::self.stride, ::self.stride]

        if self.in_channels != self.hidden_channels:
            pad = (0, 0, 0, 0, 0, self.hidden_channels - self.in_channels)
            identity = F.pad(identity, pad, 'constant', 0)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(identity + x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels_per_layer: list = [16, 32, 64],
        n: int = 3,
        num_classes: int = 10,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels_per_layer[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=channels_per_layer[0],
        )
        self.relu = nn.ReLU(inplace=True)

        self.conv2_x = self._make_layer(
            in_channels=channels_per_layer[0],
            hidden_channels=channels_per_layer[0],
            stride=1,
            num_block=n,
        )

        self.conv3_x = self._make_layer(
            in_channels=channels_per_layer[0],
            hidden_channels=channels_per_layer[1],
            stride=2,
            num_block=n,
        )

        self.conv4_x = self._make_layer(
            in_channels=channels_per_layer[1],
            hidden_channels=channels_per_layer[2],
            stride=2,
            num_block=n,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels_per_layer[2], num_classes)

        self._init_params()

    def _make_layer(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int,
        num_block: int,
    ) -> nn.Sequential:
        layers = []

        layers.append(
            BasicBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                stride=stride,
            )
        )
        
        for _ in range(1, num_block):
            layers.append(
                BasicBlock(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    resnet20 = ResNet(
        in_channels=3,
        channels_per_layer=[16, 32, 64],
        n=3,
        num_classes=10,
    )

    x = torch.zeros((1, 3, 32, 32))
    out = resnet20(x)