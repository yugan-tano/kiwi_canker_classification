## 每周进度更新
每周六晚上确定项目，汇报各自进度和问题，确保周日能完成每周进度

## 任务分工和进度规划
### 第一周：
1. 完成数据集的收集（患病和健康各300张）、
2. 项目环境和数据准备：
   - 按照下方项目结构新建项目 （之后会把项目文件上传到群里）
   - 完成基础配置文件`configs/default.yaml`和`data/dataset.py`的撰写（组长）
3. 技术预研与框架搭建
   - 编写`scripts/train.py`的训练流程骨架 （组长）
   - 编写`models/se_resnet.py`的框架，留出接口 （组长）
   - 实现`models/se_module.py`的SELayer类 （组员）

### 第二周：实现完整代码

### 第三周：模型调优，期末报告基本完成大体部分

### 第四周：准备答辩和PPT，细化报告

## 项目结构
kiwi_canker_classification/<br>
├── data/                   # 数据相关<br>
│   ├── raw/               # 原始图片数据<br>
│       ├── disease        # 患病的图片数据<br>
│       └── healthy        # 健康的图片数据<br>
│   ├── processed/         # 处理后的数据<br>
│       ├── train          # 训练集图片数据<br>
│       ├── dev            # 验证集图片数据<br>
│       └── test           # 测试集图片数据<br>
│   └── dataset.py         # 数据集定义（核心）<br>
├── models/                # 模型定义<br>
│   ├── __init__.py<br>
│   ├── resnet.py          # 基线ResNet34<br>
│   ├── se_resnet.py       # SE-ResNet34（核心）<br>
│   └── se_module.py       # SE注意力模块（分配给组员B）<br>
├── utils/                 # 工具函数<br>
│   ├── __init__.py<br>
│   ├── visualization.py   # 可视化工具（分配给组员A）<br>
│   └── metrics.py         # 评估指标（分配给组员B）<br>
├── configs/               # 配置文件<br>
│   └── default.yaml       # 所有超参数配置<br>
├── outputs/               # 输出结果<br>
│   ├── checkpoints/       # 模型保存<br>
│   ├── logs/              # 训练日志<br>
│   └── results/           # 实验结果<br>
├── scripts/               # 执行脚本<br>
│   ├── train.py           # 主训练脚本（核心）<br>
│   ├── evaluate.py        # 评估脚本<br>
│   └── visualize.py       # 可视化脚本（分配给组员A）<br>
├── requirements.txt       # 依赖包列表<br>
└── README.md             # 项目说明

# 项目报告大纲
## 基于SE-ResNet的猕猴桃溃疡病图像分类研究

### 摘要
### 1. 引言
### 2. 相关工作
### 3. 方法设计
   #### 3.1 ResNet基础模型
   #### 3.2 SE注意力机制
   #### 3.3 SE-ResNet模型设计
### 4. 实验与结果
   #### 4.1 数据集
   #### 4.2 实验设置
   #### 4.3 结果分析
### 5. 结论
### 参考文献

# 项目PPT参考结构
1. 封面页（题目、组成员及相关信息、班级、指导老师）
2. 项目背景与意义（1页）
3. 技术方案介绍（2页）
   - ResNet原理简述
   - SE注意力机制
   - 模型改进思路
4. 实验设计与实现（2页）
   - 数据集介绍
   - 实验设置
5. 结果分析与展示（2页）
   - 准确率对比
   - 可视化效果
6. 总结与展望（1页）