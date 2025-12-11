# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from data.dataset import create_datasets
from models.resnet import resnet34  # 先用基础ResNet34


def main():
    print("=" * 50)
    print("开始训练 ResNet34 模型")
    print("=" * 50)

    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 准备数据
    print("\n1. 准备数据...")
    train_dataset, val_dataset, test_dataset = create_datasets()

    # 创建数据加载器
    batch_size = 8  # 先用小批量，内存不足可以改为4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练批次: {len(train_loader)} (批量大小: {batch_size})")
    print(f"验证批次: {len(val_loader)}")

    # 3. 创建模型
    print("\n2. 创建模型...")
    model = resnet34(num_classes=2, pretrained=False)  # 先不用预训练
    model = model.to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} (约 {total_params / 1e6:.2f}M)")

    # 4. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 训练循环
    print("\n3. 开始训练...")
    num_epochs = 3  # 先训练3轮看看效果

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 每10个批次打印一次
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 计算训练准确率
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # 打印本轮结果
        print(f"\nEpoch [{epoch + 1}/{num_epochs}] 完成:")
        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.2f}%")
        print(f"  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_accuracy:.2f}%")
        print("-" * 40)

    # 6. 保存模型
    print("\n4. 保存模型...")
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "outputs/checkpoints/resnet34_baseline.pth")
    print("模型已保存到: outputs/checkpoints/resnet34_baseline.pth")

    # 7. 测试模型
    print("\n5. 测试模型...")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"测试准确率: {test_accuracy:.2f}% ({test_correct}/{test_total})")

    print("\n" + "=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
