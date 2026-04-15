import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from VisualAD_lib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD


class PadToSquare:
    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        if hasattr(img, "size") and not callable(img.size):
            width, height = img.size
        else:
            height, width = img.shape[-2:]

        if width == height:
            return img

        max_side = max(width, height)
        pad_w = max_side - width
        pad_h = max_side - height
        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2,
        )
        return TF.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)


def _convert_to_rgb(image):
    return image.convert("RGB")


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def get_transform(args):
    target_transform = transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    preprocess = transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize(
            size=(args.image_size, args.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True
        ),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
    ])
    return preprocess, target_transform
