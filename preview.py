# encoding: utf-8
import torch
import ipdb
import cv2
import numpy as np
from options import opt
# from dataloader import paired_dataset
from mscv.summary import create_summary_writer, write_image
from mscv.image import tensor2im

from dataloader.dataloaders import train_dataloader, val_dataloader
import cv2

import misc_utils as utils

import random

"""
source domain 是clear的
"""
writer = create_summary_writer('logs/preview')

"""
这个改成需要预览的数据集
"""
previewed = train_dataloader  # train_dataloader, val_dataloader

from PIL import Image, ImageDraw, ImageFont
 

names = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物",
    "40": "可回收物/毛巾",
    "41": "可回收物/饮料盒",
    "42": "可回收物/纸袋"
}

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  
        img = Image.fromarray(img)  # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
    draw = ImageDraw.Draw(img)
    # 字体
    fontStyle = ImageFont.truetype(
        "MSYHBD.TTC", textSize, encoding="utf-8")
 
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
 
    return np.asarray(img)


def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    # denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    # img -= mean
    # img *= denominator
    img *= std
    img += mean
    return img

CanChinese = False

for i, sample in enumerate(previewed):
    # if i > 30:
    #     break
    utils.progress_bar(i, len(previewed), 'Handling...')
    if opt.debug:
        ipdb.set_trace()

    image = sample['input'][0].detach().cpu().numpy().transpose([1,2,0])
    # image = (image.copy()
    image = (denormalize(image, max_pixel_value=1.0)*255).astype(np.uint8).copy()

    label = sample['label'][0].item()

    if CanChinese:
        name = names[str(label)]
        image = cv2ImgAddText(image, name, 7, 3, (255, 0, 0), textSize=24)
    else:
        cv2.putText(image, 'label: ' + str(label), (10, 30), 0, 1, (255, 0, 0), 2)

    write_image(writer, f'preview_{opt.dataset}/{i}', '0_input', image, 0, 'HWC')


writer.flush()