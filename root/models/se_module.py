import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) 注意力模块

    参数:
        channels (int): 输入特征的通道数
        reduction (int): 缩减比例，默认为16
    """

    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()

        # 全局平均池化层 (Squeeze操作)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层 (Excitation操作)
        # 第一个全连接层: channels -> channels // reduction
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        # 第二个全连接层: channels // reduction -> channels
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        参数:
            x (torch.Tensor): 输入特征图，形状为 [batch_size, channels, height, width]

        返回:
            torch.Tensor: 经过SE注意力加权的特征图，形状与输入相同
        """
        batch_size, channels, height, width = x.size()

        # Squeeze: 全局平均池化，将空间维度压缩为1x1
        squeeze = self.global_avg_pool(x)  # [batch_size, channels, 1, 1]
        squeeze = squeeze.view(batch_size, channels)  # [batch_size, channels]

        # Excitation: 通过两个全连接层学习通道权重
        excitation = self.fc1(squeeze)  # [batch_size, channels // reduction]
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)  # [batch_size, channels]
        excitation = self.sigmoid(excitation)  # [batch_size, channels]

        # 重塑为原始特征图的形状
        excitation = excitation.view(batch_size, channels, 1, 1)  # [batch_size, channels, 1, 1]

        # 通道权重与原始特征图相乘
        weighted_x = x * excitation.expand_as(x)

        return weighted_x


"""# 测试代码
if __name__ == "__main__":
    # 测试SELayer的功能
    batch_size, channels, height, width = 4, 64, 32, 32
    reduction = 16

    # 创建SE模块实例
    se_layer = SELayer(channels, reduction)

    # 创建随机输入
    x = torch.randn(batch_size, channels, height, width)

    # 前向传播
    output = se_layer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"SE模块参数量: {sum(p.numel() for p in se_layer.parameters())}")

    # 验证输出形状与输入相同
    assert x.shape == output.shape, "输出形状应与输入形状相同"
    print("测试通过！")"""