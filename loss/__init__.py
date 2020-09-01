from .rangeloss import range_loss as criterionRange
from .label_smooth import label_smooth_loss
from torch import nn

criterionCE = nn.CrossEntropyLoss()
