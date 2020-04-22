from .Default.Model import Model as Default
from .ResNeSt.Model import Model as ResNeSt
from .iResNet.Model import Model as iResNet

models = {
    'default': Default,  # if --model is not specified
    'res101': Default,
    'ResNeSt101': ResNeSt,
    'ResNeSt200': ResNeSt,
    'iResNet101': iResNet,
    'iResNet152': iResNet,
    'iResNet200': iResNet,
    'iResNet1001': iResNet


}


def get_model(model: str):
    if model in models:
        return models[model]
    else:
        raise Exception('No such model: "%s", available: {%s}.' % (model, '|'.join(models.keys())))

