# models/__init__.py
from .resnet import ResNet34, resnet34
from .se_module import SELayer
from .se_resnet import SE_ResNet34, SE_BasicBlock, create_se_resnet34  # 修改这里

__all__ = [
    'ResNet34', 'resnet34',
    'SELayer',
    'SE_ResNet34', 'SE_BasicBlock', 'create_se_resnet34'  # 修改这里
]
