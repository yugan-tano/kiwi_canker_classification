import torch
import torch.nn as nn


# 基础残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# ResNet34主模型
class ResNet34(nn.Module):
    def __init__(self, num_classes=2, in_channels=3):
        super(ResNet34, self).__init__()

        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4个残差层
        self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)  # 3个残差块
        self.layer2 = self._make_layer(64, 128, blocks=4, stride=2)  # 4个残差块
        self.layer3 = self._make_layer(128, 256, blocks=6, stride=2)  # 6个残差块
        self.layer4 = self._make_layer(256, 512, blocks=3, stride=2)  # 3个残差块

        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None

        # 如果需要下采样（改变维度）
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        layers = []
        # 第一个块可能有下采样
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion

        # 后续块
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_feature_maps(self, x):
        """获取特征图（用于可视化）"""
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x.detach())  # conv1输出

        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x.detach())  # layer1输出
        x = self.layer2(x)
        features.append(x.detach())  # layer2输出
        x = self.layer3(x)
        features.append(x.detach())  # layer3输出
        x = self.layer4(x)
        features.append(x.detach())  # layer4输出

        return features


# 快捷函数，方便创建模型
def resnet34(num_classes=2, pretrained=False, in_channels=3):
    """创建ResNet34模型

    Args:
        num_classes: 分类数，猕猴桃溃疡病二分类就是2
        pretrained: 是否加载预训练权重（如果你有ImageNet预训练权重）
        in_channels: 输入通道数，RGB图像为3
    """
    model = ResNet34(num_classes=num_classes, in_channels=in_channels)

    if pretrained:
        # 这里可以添加加载预训练权重的代码
        print("注意：需要提供预训练权重文件路径")
        # 示例：model.load_state_dict(torch.load('path/to/weights.pth'))

    return model


# 测试代码（可以直接运行这个文件来验证）
if __name__ == "__main__":
    # 创建模型
    model = resnet34(num_classes=2)
    print("模型创建成功！")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)  # 2张224x224的RGB图像
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应该是 torch.Size([2, 2])

    # 测试特征提取
    features = model.get_feature_maps(x)
    for i, feat in enumerate(features):
        print(f"特征图 {i + 1} 形状: {feat.shape}")