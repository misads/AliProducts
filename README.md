# AliProducts

Potato Chips Classification.

### Requirements

```yaml
python >= 3.5
torch >= 0.4
tensorboardX >= 1.6
utils-misc >= 0.0.3
torch-template >= 0.0.4
```

### Training

```bash
python3 train.py --tag train_1 --model res101 --b 16 --epochs 100 --gpu 0
```

### Load Checkpoint

```bash
python3 train.py --load checkpoints/train_1 --which-epoch 500
```

### Testing

```shell script
python3 eval.py --tag eval_1 --model res101 checkpoints/train_1 --which-epoch 500 
# test results will be saved in 'results/eval_1' directory
```

### Visulization on TensorBoard

```shell script
tensorboard --logdir logs/train_1
```

### Documentation

Detailed file structure and code usage can be found in train.py.

### 如何添加新的模型：

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