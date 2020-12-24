from .label_smooth import LabelSmoothing
from torch import nn
from options import opt

criterionCE = nn.CrossEntropyLoss()

if opt.smooth != 0:
    label_smooth_loss = LabelSmoothing(smoothing=opt.smooth)
else:
    label_smooth_loss = 0.

def get_loss(predicted, label, avg_meters, *args):
    ce_loss = criterionCE(predicted, label)

    if opt.smooth == 0:  # 不用label smooth
        self.avg_meters.update({'CE loss': ce_loss.item()})
        return ce_loss
    else:
        smooth_loss = label_smooth_loss(predicted, label) 
        self.avg_meters.update({f'Smooth loss(smooth={opt.smooth})': smooth_loss.item()})
        return smooth_loss