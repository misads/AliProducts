# CVPR 2020 AliProducts Challenge

一个通用的图像分类模板，天池/CVPR AliProducts Challenge 3/688🍟

队伍：薯片分类器！

<img alt='preview' src='http://www.xyu.ink/wp-content/uploads/2020/06/preview.png' width=600 height=400>

## 支持的功能

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

## 环境需求

```yaml
python >= 3.6
torch >= 1.0
tensorboardX >= 1.6
utils-misc >= 0.0.5
mscv >= 0.0.3
opencv-python==4.2.0.34  # opencv>=4.4需要编译，建议安装4.2版本
opencv-python-headless==4.2.0.34
albumentations>=0.5.1 
```

都是很好装的库，不需要编译。

## 使用方法

### 训练和验证模型

① 生成输入图片和标签对应的train.txt和val.txt

　　新建一个datasets文件夹，制作文件列表train.txt和val.txt并把它们放在datasets目录下，train.txt和val.txt需要满足这样的格式：每行是一个样本的图像绝对路径和标签，用空格隔开。如下所示：
  
```yml
# datasets/train.txt
/home/xhy/datasets/aliproducts/train/img_11739.jpg 24
/home/xhy/datasets/aliproducts/train/img_15551.jpg 31
/home/xhy/datasets/aliproducts/train/img_19451.jpg 39
/home/xhy/datasets/aliproducts/train/img_16965.jpg 34
/home/xhy/datasets/aliproducts/train/img_1271.jpg 3
/home/xhy/datasets/aliproducts/train/img_6502.jpg 13
/home/xhy/datasets/aliproducts/train/img_3148.jpg 7
```

　　生成好train.txt和val.txt后目录结构是这样的：
  
```yml
AliProducts
    └── datasets
          ├── train.txt      
          └── val.txt    
```

② 训练模型

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --tag resnest --model ResNeSt101 --optimizer sgd --scheduler 2x -b 24 --lr 0.0001  # --tag用于区分每次实验，可以是任意字符串
```

　　`scheduler 2x`一共训练24个epochs，具体可参考`scheduler/__init__.py`。训练的中途可以在验证集上验证，添加`--val_freq 10`参数可以指定10个epoch验证一次，添加`--save_freq 10`参数可以指定10个epoch保存一次checkpoint。

③ 验证训练的模型

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py --model ResNeSt101 -b 24 --load checkpoints/resnest/20_ResNeSt101.pt
```

　　验证的结果会保存在`results/<tag>`目录下，如果不指定`--tag`，默认的`tag`为`cache`。

④ 恢复中断的训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --tag resnest_resume --model ResNeSt101 --epochs 20 -b 24 --lr 0.0001 --load checkpoints/resnest/20_ResNeSt101.pt --resume
```

　　`--load`的作用是载入网络权重；`--resume`参数会同时加载优化器参数和epoch信息(继续之前的训练)，可以根据需要添加。

⑤ 在测试集上测试

```bash
CUDA_VISIBLE_DEVICES=0 python submit.py --model ResNeSt101 --load checkpoints/resnest/20_ResNeSt101.pt
```

### 记录和查看日志

　　所有运行的命令和运行命令的时间戳会自动记录在`run_log.txt`中。

　　不同实验的详细日志和Tensorboard日志文件会记录在`logs/<tag>`文件夹中，checkpoint文件会保存在`checkpoints/<tag>`文件夹中。如下所示：

```yml
AliProducts
    ├── run_log.txt    # 运行的历史命令
    ├── logs
    │     └── <tag>
    │           ├── log.txt  
    │           └── [Tensorboard files]
    └── checkpoints
          └── <tag>
                ├── 1_Model.pt
                └── 2_Model.pt
          
```

### 参数说明

`--tag`参数是一次操作(`train`或`eval`)的标签，日志会保存在`logs/标签`目录下，保存的模型会保存在`checkpoints/标签`目录下。  

`--model`是使用的模型，所有可用的模型定义在`network/__init__.py`中。  

`--epochs`是训练的代数。  

`-b`参数是`batch_size`，可以根据显存的大小调整。  

`--lr`是初始学习率。

`--load`是加载预训练模型。  

`--resume`配合`--load`使用，会恢复上次训练的`epoch`和优化器。  

`--gpu`指定`gpu id`，目前只支持单卡训练。  

`--debug`以debug模式运行，debug模式下每个`epoch`只会训练前几个batch。

另外还可以通过参数调整优化器、学习率衰减、验证和保存模型的频率等，详细请查看`options/options.py`。  


### 清除不需要的实验记录

　　运行 `python clear.py --tag <your_tag>` 可以清除不需要的实验记录，注意这是不可恢复的，如果你不确定你在做什么，请不要使用这条命令。

## 如何添加自定义的模型：

```
如何添加新的模型：

① 复制`network`目录下的`ResNeSt`文件夹，改成另外一个名字(比如MyNet)。

② 仿照`ResNeSt`的model.py，修改自己的网络结构、损失函数和优化过程。

③ 在network/__init__.py中import你的Model并且在models = {}中添加它。
    from MyNet.Model import Model as MyNet
    models = {
        'default': Default,
        'MyNet': MyNet,
    }

④ 运行 python train.py --model MyNet 看能否正常训练
```