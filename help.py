from network import models
import misc_utils as utils
import os


def get_dirs(dir):
    res = []
    for file in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, file)):
            res.append(file)

    return '|'.join(res)


model_choices = '|'.join(models.keys())
log_choices = get_dirs('logs')
checkpoints_choices = get_dirs('checkpoints')


help = """Usage:
Training:
    python train.py --tag your_tag --model {%s} -b 24 --gpu 0""" % model_choices + """

Finding Best Hyper Params:
    python runx.py --run

Debug:
    python train.py --model {%s} --debug""" % model_choices + """
    
Resume Training:
    python train.py --tag your_tag --model ResNeSt101 --epochs 20 -b 24 --load checkpoints/{%s}/9_Model.pt """ % checkpoints_choices + """--resume

Eval:
    python eval.py --tag your_tag2 --model ResNeSt101 -b 48 --load checkpoints/{%s}/9_Model.pt """ % checkpoints_choices + """

Generate Submission:
    python submit.py --model ResNeSt101 --load checkpoints/my_ckpt.pt -b 48 --gpu 0

See Running Log:
    cat logs/{%s}/log.txt""" % log_choices + """

Clear(delete all files with the tag, BE CAREFUL to use):
    python clear.py --tag {%s} """ % log_choices + """

See ALL Running Commands:
    cat run_log.txt

"""

print(help)