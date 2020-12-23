from .no_transform import No_Transform
from .resize import Resize


transforms = {
    'resize': Resize,
    'none': No_Transform,
}


def get_transform(transform: str):
    if transform in transforms:
        return transforms[transform]
    else:
        raise Exception('No such transform: "%s", available: {%s}.' % (transform, '|'.join(transforms.keys())))

