import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    '''
    Bottleneck block for ImageNet-1k
    Conv 1x1 -> 3x3 -> 1x1 
    When the dimensions increase, option (B) used
    '''
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        expansion: int,
        stride: int,
    ) -> None:
        super().__init__()

        # 1x1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
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
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(
            num_features=hidden_channels,
        )

        # 1x1
        self.conv3 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels*expansion,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(
            num_features=hidden_channels*expansion,
        )

        # shortcut
        self.shortcut = None
        if stride != 1 or in_channels != hidden_channels*expansion:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels*expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.bn_shortcut = nn.BatchNorm2d(
                num_features=hidden_channels*expansion,
            )

        self._init_params()

    def _init_params(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        if self.shortcut is not None:
            nn.init.kaiming_normal_(self.shortcut.weight) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.bn_shortcut(self.shortcut(x)) if self.shortcut is not None else x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.relu(identity + x)
        return x

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels_per_layer: list = [64, 128, 256, 512],
        expansion: int = 4,
        num_blocks: list = [3, 4, 6, 3],
        num_classes: int = 1000,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels_per_layer[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(
            num_features=channels_per_layer[0],
        )
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv2_x = self._make_layer(
            in_channels=channels_per_layer[0],
            hidden_channels=channels_per_layer[0],
            expansion=expansion,
            stride=1,
            num_block=num_blocks[0]
        )

        self.conv3_x = self._make_layer(
            in_channels=channels_per_layer[0]*expansion,
            hidden_channels=channels_per_layer[1],
            expansion=expansion,
            stride=2,
            num_block=num_blocks[1]
        )

        self.conv4_x = self._make_layer(
            in_channels=channels_per_layer[1]*expansion,
            hidden_channels=channels_per_layer[2],
            expansion=expansion,
            stride=2,
            num_block=num_blocks[2]
        )

        self.conv5_x = self._make_layer(
            in_channels=channels_per_layer[2]*expansion,
            hidden_channels=channels_per_layer[3],
            expansion=expansion,
            stride=2,
            num_block=num_blocks[3]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels_per_layer[3]*expansion, num_classes)

        self._init_params()

    def _make_layer(
        self,
        in_channels: int,
        hidden_channels: int,
        expansion: int,
        stride: int,
        num_block: int,
    ) -> nn.Sequential:
        layers = []

        layers.append(
            BottleneckBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                expansion=expansion,
                stride=stride,
            )
        )

        for _ in range(1, num_block):
            layers.append(
                BottleneckBlock(
                    in_channels=hidden_channels*expansion,
                    hidden_channels=hidden_channels,
                    expansion=expansion,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    resnet50 = ResNet(
        in_channels=3,
        channels_per_layer=[64, 128, 256, 512],
        expansion=4,
        num_blocks=[3, 4, 6, 3],
        num_classes=1000,
    )

    x = torch.zeros((1, 3, 224, 224))
    out = resnet50(x)