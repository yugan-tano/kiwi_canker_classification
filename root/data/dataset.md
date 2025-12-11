# dataset.py 模块详细解释文档

## 概述

`dataset.py` 是猕猴桃溃疡病图像分类项目的数据加载和预处理模块，负责：
- 读取图像数据并构建PyTorch数据集
- 进行数据预处理和标准化
- 按比例分割训练集、验证集和测试集
- 集成配置文件管理

## 一、模块导入部分

```python
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import torchvision.transforms as transforms
import yaml
```

**关键点**：
1. **torch**：PyTorch深度学习框架
2. **Dataset**：PyTorch数据集基类，所有自定义数据集必须继承
3. **random_split**：用于随机分割数据集
4. **PIL.Image**：Python图像处理库，用于加载图像
5. **os**：操作系统接口，用于文件路径操作
6. **torchvision.transforms**：图像预处理和增强工具
7. **yaml**：读取YAML配置文件

## 二、配置文件读取

```python
with open('../configs/default.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

SAMPLES_PER_CLASS = config['data']['num_samples_per_class']
TRAIN_RATIO = config['data']['train_ratio']
VAL_RATIO = config['data']['val_ratio']
```

**作用**：从YAML配置文件读取关键参数，实现配置与代码分离
- `num_samples_per_class`：每类图片加载数量（控制数据集大小）
- `train_ratio`：训练集比例（如0.7表示70%）
- `val_ratio`：验证集比例（如0.15表示15%）

## 三、核心类：KiwiCankerDataset

### 3.1 类定义与初始化

```python
class KiwiCankerDataset(Dataset):
    def __init__(self, root_dir, transform=None, samples_per_class=SAMPLES_PER_CLASS):
        self.image_paths = []  # 存储所有图片路径
        self.labels = []       # 存储对应标签
        self.transform = transform  # 图像预处理变换
```

**关键参数**：
- `root_dir`：数据根目录，包含'disease'和'healthy'两个子文件夹
- `transform`：图像预处理操作（如resize、normalize等）
- `samples_per_class`：控制每类图片加载数量

### 3.2 数据加载逻辑

```python
# 读取患病图片（标签0）
disease_dir = os.path.join(root_dir, 'disease')
disease_images = os.listdir(disease_dir)[:samples_per_class]

# 读取健康图片（标签1）
healthy_dir = os.path.join(root_dir, 'healthy')
healthy_images = os.listdir(healthy_dir)[:samples_per_class]
```

**目录结构要求**：
```
raw/
├── disease/    # 患病猕猴桃图片（自动分配标签0）
└── healthy/    # 健康猕猴桃图片（自动分配标签1）
```

### 3.3 关键方法

#### `__len__(self)`
```python
def __len__(self):
    return len(self.image_paths)
```
**作用**：返回数据集总大小，供DataLoader使用

#### `__getitem__(self, idx)`
```python
def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    image = Image.open(img_path).convert('RGB')  # 确保RGB三通道
    label = self.labels[idx]
    
    if self.transform:
        image = self.transform(image)  # 应用预处理
        
    return image, label
```

**异常处理**：如果图片加载失败，创建黑色占位图像
```python
try:
    image = Image.open(img_path).convert('RGB')
except:
    print(f"无法加载图片: {img_path}")
    image = Image.new('RGB', (224, 224), (0, 0, 0))  # 224x224黑色图像
```

## 四、数据集创建函数：create_datasets()

### 4.1 数据集分割计算

```python
total_samples = SAMPLES_PER_CLASS * 2  # 每类数量 × 2个类别
train_size = int(total_samples * TRAIN_RATIO)
val_size = int(total_samples * VAL_RATIO)
test_size = total_samples - train_size - val_size
```

**示例计算**（配置为每类100张，训练70%，验证15%，测试15%）：
- 总样本：100 × 2 = 200张
- 训练集：200 × 0.7 = 140张
- 验证集：200 × 0.15 = 30张
- 测试集：200 - 140 - 30 = 30张

### 4.2 图像预处理管道

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一调整为224×224
    transforms.ToTensor(),          # 转为PyTorch张量（0-1范围）
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # ImageNet均值
        std=[0.229, 0.224, 0.225]   # ImageNet标准差
    )
])
```

**标准化说明**：
- 使用ImageNet的均值和标准差（适用于预训练模型）
- 公式：`normalized = (image - mean) / std`

### 4.3 数据集分割实现

```python
# 1. 创建完整数据集
full_dataset = KiwiCankerDataset(os.path.join(current_dir,"raw"), transform)

# 2. 先分割训练+验证集 和 测试集
train_val_size = train_size + val_size
train_val_dataset, test_dataset = random_split(
    full_dataset,
    [train_val_size, test_size],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
)

# 3. 再从训练验证集中分割训练集和验证集
train_dataset, val_dataset = random_split(
    train_val_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

**分割策略**：两级分割确保比例准确
**随机种子**：`manual_seed(42)`保证每次运行分割结果一致

## 五、使用示例

### 5.1 直接运行测试
```bash
python dataset.py
```
输出将显示数据集分割结果和样本信息

### 5.2 在其他模块中导入使用
```python
from dataset import create_datasets

train_set, val_set, test_set = create_datasets()

# 创建DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
```

## 六、配置文件对应关系

### default.yaml 相关配置项：
```yaml
data:
  train_ratio: 0.7        # 对应 TRAIN_RATIO
  val_ratio: 0.15         # 对应 VAL_RATIO
  num_samples_per_class: 100  # 对应 SAMPLES_PER_CLASS
  transform:
    resize: [224, 224]    # 对应 transforms.Resize((224, 224))
    mean: [0.485, 0.456, 0.406]  # 标准化均值
    std: [0.229, 0.224, 0.225]   # 标准化标准差
```

## 七、设计优点

1. **模块化设计**：数据集创建、预处理、分割逻辑清晰分离
2. **配置驱动**：关键参数从配置文件读取，便于调整实验
3. **异常处理**：图片加载失败时有降级方案
4. **可复现性**：固定随机种子确保每次分割一致
5. **兼容性**：使用ImageNet标准化，兼容预训练模型

## 八、注意事项

1. **数据目录**：确保`raw`文件夹与`dataset.py`在同一目录下
2. **图片格式**：支持常见图片格式（JPG、PNG等）
3. **内存管理**：使用路径列表而非直接加载所有图像，节省内存
4. **类别平衡**：自动确保每类样本数量相同
5. **标签映射**：disease→0，healthy→1（可在配置文件中调整）

## 九、扩展建议

如需扩展功能，可考虑：
1. 添加数据增强选项
2. 支持多分类任务
3. 添加数据可视化方法
4. 实现交叉验证支持
5. 添加样本权重平衡

这个模块是项目的数据基础，理解后可以快速进行模型训练和评估。
