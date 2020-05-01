# AliProducts 薯片🍟分类

Potato Chips Classification.

## To do List

- 网络结构
  - [x] ResNet(101)
  - [x] ResNe**X**t(101)
  - [x] ResNe**S**t(101, 200)
  - [x] **i**ResNet(101, 152, 200)
  - [x] EffiCientNet(B-5, B-7)
  - [ ] DenseNet(201)

- 改结构
  - [ ] 自注意力机制
  - [ ] 解决Long-Tailed Problem

- 损失函数
  - [x] 交叉熵
  - [ ] Focal loss

- 优化器
  - [x] Adam
  - [ ] RAdam
  - [ ] NAdam
  - [ ] Look Ahead

- Data Argumentation
  - [ ] 随机旋转(-10, 10)度 (有黑边)
  - [ ] 随机左右翻转(字会变反)
  - [x] 随机放大(1, 1.3)倍
  - [x] 随机色相(-0.1, 0.1)
  - [x] 随机饱和度(-1/1.5, 1/1.5)
  - [x] 随机亮度(-1/1.5, 1/1.5)
- TTA
  - [ ] 放大、色相、饱和度、亮度
  - [ ] ttach库

- 其他Tricks
  - [ ] mix up
  - [ ] 使用fp_16训练，提高训练速度
  - [ ] One_Cycle 学习率

## Prerequisites

```yaml
python >= 3.6
torch >= 0.4
tensorboardX >= 1.6
utils-misc >= 0.0.5
torch-template >= 0.0.4
```

## Code Usage

```python
python help.py
```

## 如何添加新的模型：

```
如何添加新的模型：

① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

② 在network/__init__.py中import你的Model并且在models = {}中添加它。
    from MyNet.Model import Model as MyNet
    models = {
        'default': Default,
        'MyNet': MyNet,
    }

③ 尝试 python train.py --model MyNet 看能否成功运行
```