# CVPR 2020 AliProducts Challenge

一个通用的图像分类模板，天池/CVPR AliProducts Challenge 3/688🍟

队伍：薯片分类器！

## Features

- Backbone
  - [x] ResNet(101)
  - [x] ResNe**X**t(101) 
  - [x] ResNe**S**t(101, 200)
  - [x] Res2Net(101)
  - [x] **i**ResNet(101, 152, 200)
  - [x] EffiCientNet(B-5, B-7)
  
- 优化器
  - [x] Adam
  - [x] SGD
  - [x] Ranger(RAdam+Look Ahead)
- Scheduler
  - [x] Cos
  - [x] 自定义scheduler
  
- Input Pipeline
  
  - [x] 裁剪和切割
  - [x] 随机翻折和旋转
  - [x] 随机放大
  - [x] 随机色相
  - [x] 随机饱和度
  - [x] 随机亮度
  - [x] Norm_input

- 其他tricks
  - [x] label smooth
  - [x] model ensemble
  - [x] TTA
## Prerequisites

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
torch-template >= 0.0.4
mscv >= 0.0.3
```

都是很好装的库，不需要编译。

## Code Usage

```bash
Code Usage:
Training:
    python train.py --tag your_tag --model ResNeSt101 --epochs 20 -b 24 --lr 0.0001 --gpu 0

Finding Best Hyper Params:  # 需先设置好sweep.yml
    python grid_search.py --run

Resume Training (or fine-tune):
    python train.py --tag your_tag --model ResNeSt101 --epochs 20 -b 24 --load checkpoints/your_tag/9_ResNeSt101.pt --resume --gpu 0

Eval:
    python eval.py --model ResNeSt101 -b 96 --load checkpoints/your_tag/9_ResNeSt101.pt --gpu 1

Generate Submission:
    python submit.py --model ResNeSt101 --load checkpoints/your_tag/9_ResNeSt101.pt -b 96 --gpu 0

Check Running Log:
    cat logs/your_tag/log.txt

Clear(delete all files with the tag, BE CAREFUL to use):
    python clear.py --tag your_tag

See ALL Running Commands:
    cat run_log.txt
```

参数用法：

`--tag`参数是一次操作(`train`或`eval`)的标签，日志会保存在`logs/标签`目录下，保存的模型会保存在`checkpoints/标签`目录下。  

`--model`是使用的模型，所有可用的模型定义在`network/__init__.py`中。  

`--epochs`是训练的代数。  

`-b`参数是`batch_size`，可以根据显存的大小调整。  

`--lr`是初始学习率。

`--load`是加载预训练模型。  

`--resume`配合`--load`使用，会恢复上次训练的`epoch`和优化器。  

`--gpu`指定`gpu id`，目前只支持单卡训练。  

`--debug`以debug模式运行，debug模式下每个`epoch`只会训练前几个batch。

另外还可以通过参数调整优化器、学习率衰减、验证和保存模型的频率等，详细请查看`python train.py --help`。  

## 如何添加自定义的模型：

```
如何添加新的模型：

① 复制network目录下的Default文件夹，改成另外一个名字(比如MyNet)。

② 在network/__init__.py中import你的Model并且在models = {}中添加它。
    from MyNet.Model import Model as MyNet
    models = {
        'default': Default,
        'MyNet': MyNet,
    }

③ 尝试 python train.py --model MyNet --debug 看能否成功运行
```