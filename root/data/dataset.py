import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import torchvision.transforms as transforms

import yaml

# 读取配置文件
with open('../configs/default.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 获取配置参数
SAMPLES_PER_CLASS = config['data']['num_samples_per_class']  # 每类图片数量
TRAIN_RATIO = config['data']['train_ratio']  # 训练集比例
VAL_RATIO = config['data']['val_ratio']  # 验证集比例


class KiwiCankerDataset(Dataset):
    def __init__(self, root_dir, transform=None, samples_per_class=SAMPLES_PER_CLASS):
        """
        参数:
            samples_per_class: 每类图片数量
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # 读取患病图片
        disease_dir = os.path.join(root_dir, 'disease')
        disease_images = os.listdir(disease_dir)[:samples_per_class]
        for img_name in disease_images:
            self.image_paths.append(os.path.join(disease_dir, img_name))
            self.labels.append(0)

        # 读取健康图片
        healthy_dir = os.path.join(root_dir, 'healthy')
        healthy_images = os.listdir(healthy_dir)[:samples_per_class]
        for img_name in healthy_images:
            self.image_paths.append(os.path.join(healthy_dir, img_name))
            self.labels.append(1)

        print(f"每类加载 {samples_per_class} 张，总共 {len(self.image_paths)} 张图片")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"无法加载图片: {img_path}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_datasets():
    """创建训练集、验证集、测试集（使用配置文件参数）"""
    # 计算总数：每类数量 × 2
    total_samples = SAMPLES_PER_CLASS * 2

    # 计算具体数量
    train_size = int(total_samples * TRAIN_RATIO)
    val_size = int(total_samples * VAL_RATIO)
    test_size = total_samples - train_size - val_size

    print(f"配置参数：每类{SAMPLES_PER_CLASS}张，训练比例{TRAIN_RATIO}，验证比例{VAL_RATIO}")
    print(f"计划分割：训练集{train_size}张，验证集{val_size}张，测试集{test_size}张")

    # 基本的数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建完整数据集
    full_dataset = KiwiCankerDataset(os.path.join(current_dir,"raw"), transform)

    # 先分割训练+验证集 和 测试集
    train_val_size = train_size + val_size

    train_val_dataset, test_dataset = random_split(
        full_dataset,
        [train_val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 再从训练验证集中分割训练集和验证集
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"实际分割结果：")
    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"测试集: {len(test_dataset)} 张")

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # 使用配置参数创建数据集
    train_set, val_set, test_set = create_datasets()
    print("\n数据集创建完成！")

    # 查看一个样本
    img, label = train_set[0]
    print(f"图片张量形状: {img.shape}")
    print(f"标签: {label}")
