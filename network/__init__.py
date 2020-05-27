from .Default.Model import Model as Default
from .ResNeXt.Model import Model as ResNeXt
from .ResNeSt.Model import Model as ResNeSt
from .Res2Net.Model import Model as Res2Net
from .iResNet.Model import Model as iResNet
from .EfficientNet.Model import Model as Efficient
from .MetaEmbedding.Model import Model as MetaEmbedding

models = {
    'default': Default,  # if --model is not specified
    'res101': Default,
    'ResNeXt101': ResNeXt,
    'ResNeSt101': ResNeSt,
    'ResNeSt200': ResNeSt,
    'Res2Net101': Res2Net,  # Res2Net-v1b-101
    'iResNet101': iResNet,
    'iResNet152': iResNet,
    'iResNet200': iResNet,
    'iResNet1001': iResNet,
    'EfficientNet-B5': Efficient,
    'EfficientNet-B7': Efficient,
    'MetaEmbedding': MetaEmbedding,  # ResNeST101 + MetaEmbedding

}


def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

