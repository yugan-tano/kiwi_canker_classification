## **项目核心目标**
用深度学习模型（SE-ResNet34）对猕猴桃图像进行二分类：
- 类别0：患病（disease）
- 类别1：健康（healthy）

##  **各个文件的作用**

### **1. 配置文件（configs/）**
```
resnet34.yaml - 定义模型参数（类别数、输入通道等）
default.yaml - 定义训练参数（学习率、批量大小等）
```
**作用**：把超参数集中管理，方便修改实验设置。

### **2. 数据模块（data/）**
```python
# dataset.py（需要你创建）应该实现：
class KiwiDataset:
    def __init__(self):
        # 读取图片
        # 数据增强（旋转、裁剪等）
        # 标签编码（患病=0，健康=1）
    
    def __getitem__(self, idx):
        # 返回：图像张量, 标签
```
**作用**：把原始图片转换为PyTorch能处理的格式。

### **3. 模型代码（models/）**

#### **resnet.py**
```python
# 实现了基础的ResNet34网络
class ResNet34:
    # 包含：
    # 1. 卷积层（提取特征）
    # 2. 残差块（解决梯度消失）
    # 3. 全连接层（输出分类结果）
```
**作用**：基线模型，用于对比效果。

#### **se_module.py**
```python
# SE注意力机制
class SELayer:
    # 作用：让模型关注重要通道
    # 流程：
    # 输入特征 → 全局平均池化 → 两个全连接层 → Sigmoid → 通道权重
```
**作用**：增强模型对重要特征的关注度。

#### **se_resnet.py**
```python
# SE-ResNet34 = ResNet34 + SE注意力
class SE_ResNet34:
    # 把ResNet34中的每个BasicBlock替换为SE_BasicBlock
    # SE_BasicBlock = BasicBlock + SELayer
```
**原意**：改进模型，让它在每个残差块后都应用注意力机制。

### **4. 训练脚本（scripts/train.py）**
**应该实现的核心流程**：

```python
# 伪代码流程：
def train():
    # 1. 读取配置
    config = load_yaml('configs/default.yaml')
    
    # 2. 准备数据
    dataset = KiwiDataset()
    dataloader = DataLoader(dataset)  # 分批次加载
    
    # 3. 创建模型
    model = SE_ResNet34()
    
    # 4. 定义优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()  # 用于分类
    
    # 5. 训练循环
    for epoch in range(50):  # 训练50轮
        for batch_images, batch_labels in dataloader:
            # 前向传播
            predictions = model(batch_images)
            
            # 计算损失
            loss = criterion(predictions, batch_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 保存模型
        torch.save(model.state_dict(), f'checkpoint_{epoch}.pth')
```

## 🔄 **完整的执行流程**

### **启动训练**
```bash
python scripts/train.py
```

### **训练流程分解**

#### **阶段1：数据加载**
```
原始图片 → dataset.py → 标准化张量
```

#### **阶段2：模型初始化**
```python
# 在train.py中
model = SE_ResNet34(num_classes=2)
# 这会调用se_resnet.py中的__init__
```

#### **阶段3：前向传播（推理）**
```
输入图片(3,224,224) 
    ↓
ResNet34的conv1层 
    ↓
layer1（包含SE模块的残差块）
    ↓
layer2（包含SE模块的残差块）
    ↓
layer3（包含SE模块的残差块）
    ↓
layer4（包含SE模块的残差块）
    ↓
全局平均池化
    ↓
全连接层
    ↓
输出预测(2,)  # [患病概率, 健康概率]
```

#### **阶段4：损失计算**
```python
# 假设真实标签：患病=0
predictions = [0.3, 0.7]  # 模型预测
loss = -log(0.3)  # 因为真实类别是0，惩罚第一个概率不够高
```

#### **阶段5：反向传播更新权重**
```
计算梯度 → 通过链式法则反向传播 → 更新模型参数
```

## 🧩 **当前代码状态**

### **已完成的部分**
1. ✅ ResNet34基础模型
2. ✅ SE注意力模块
3. ✅ SE-ResNet34框架
4. ✅ 配置文件

### **缺失的关键部分**
1. ❌ **dataset.py** - 数据加载（最重要！）
2. ❌ **train.py完整实现** - 训练循环
3. ❌ **验证和测试代码**

## 📝 **你需要做的工作**

### **优先级1：创建dataset.py**
```python
# data/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class KiwiCankerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: ./data/raw/
            ├── disease/   # 患病图片
            └── healthy/   # 健康图片
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # 读取患病图片
        disease_dir = os.path.join(root_dir, 'disease')
        for img_name in os.listdir(disease_dir):
            self.image_paths.append(os.path.join(disease_dir, img_name))
            self.labels.append(0)  # 患病标签=0
        
        # 读取健康图片
        healthy_dir = os.path.join(root_dir, 'healthy')
        for img_name in os.listdir(healthy_dir):
            self.image_paths.append(os.path.join(healthy_dir, img_name))
            self.labels.append(1)  # 健康标签=1
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### **优先级2：完成train.py**
```python
# scripts/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import KiwiCankerDataset
from models.se_resnet import SE_ResNet34

def main():
    # 1. 数据
    transform = ...  # 数据增强
    dataset = KiwiCankerDataset('./data/raw', transform)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # 2. 模型
    model = SE_ResNet34(num_classes=2)
    
    # 3. 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

**一句话总结**：你现在有模型定义代码，但缺少数据加载和训练循环代码，所以整个流程无法运行。
