import torch
import torch.nn as nn
import torchvision.models as models
from .se_module import SELayer

class SE_ResNet34(nn.Module):
    """SE-ResNet34模型"""

    def __init__(self, num_classes=2, pretrained=True, reduction=16):
        super(SE_ResNet34, self).__init__()

        # 加载预训练的ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)

        # 替换所有的BasicBlock为SE_BasicBlock
        self._replace_with_se_blocks(reduction)

        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def _replace_with_se_blocks(self, reduction=16):
        """将ResNet的BasicBlock替换为SE_BasicBlock"""

        # 替换layer1
        self.backbone.layer1 = self._make_se_layer(64, 64, 3, stride=1, reduction=reduction)

        # 替换layer2
        self.backbone.layer2 = self._make_se_layer(64, 128, 4, stride=2, reduction=reduction)

        # 替换layer3
        self.backbone.layer3 = self._make_se_layer(128, 256, 6, stride=2, reduction=reduction)

        # 替换layer4
        self.backbone.layer4 = self._make_se_layer(256, 512, 3, stride=2, reduction=reduction)

    def _make_se_layer(self, in_channels, out_channels, num_blocks, stride=1, reduction=16):
        """创建带有SE模块的残差层"""

        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        # 第一个块可能有下采样
        layers.append(SE_BasicBlock(in_channels, out_channels, stride, downsample, reduction))

        # 后续块
        for _ in range(1, num_blocks):
            layers.append(SE_BasicBlock(out_channels, out_channels, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        return self.backbone(x)


class SE_BasicBlock(nn.Module):
    """带有SE模块的BasicBlock"""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SE_BasicBlock, self).__init__()

        # 基本的残差块
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE注意力模块
        self.se = SELayer(out_channels, reduction)

        # 下采样
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用SE注意力
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SE_BasicBlock(nn.Module):
    """带有SE模块的BasicBlock"""

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(SE_BasicBlock, self).__init__()

        # 基本的残差块
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE注意力模块
        self.se = SELayer(out_channels, reduction)

        # 下采样
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用SE注意力
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def create_se_resnet34(num_classes=2, pretrained=True, reduction=16):
    """创建SE-ResNet34模型的工厂函数"""
    return SE_ResNet34(num_classes=num_classes, pretrained=pretrained, reduction=reduction)


if __name__ == "__main__":
    # 测试模型
    model = SE_ResNet34(num_classes=2)
    print(model)

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")