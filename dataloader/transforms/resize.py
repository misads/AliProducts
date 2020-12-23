import albumentations as A
from dataloader.transforms import custom_transform as C
from albumentations.pytorch.transforms import ToTensorV2
from options import opt


class Resize(object):
    width = height = opt.scale if opt.scale else 256

    train_transform = A.Compose(
        [
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), p=1.0),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.4, sat_shift_limit=0.4,
                                        val_shift_limit=0.4, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.3,
                                            contrast_limit=0.3, p=0.9),
            ],p=0.9),  # 色相饱和度对比度增强
            A.GaussianBlur(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),  # TTA×8
            A.Normalize(max_pixel_value=1.0, p=1.0),
            A.CoarseDropout(p=0.5, max_width=32, max_height=32),
            # A.Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    divisor = 8  # padding成8的倍数
    val_transform = A.Compose(
        [
            # A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=divisor, pad_width_divisor=divisor, p=1.0),
            A.Resize(height=height, width=width, p=1.0),
            A.Normalize(max_pixel_value=1.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
