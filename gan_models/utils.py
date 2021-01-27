import csv
import math
import numpy as np
from argparse import Namespace, ArgumentTypeError
from pathlib2 import Path


def save_args(args: Namespace, save_dir: Path) -> None:
    with open(save_dir / 'hparams.csv', 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(['hparam', 'value'])
        for k, v in args.__dict__.items():
            csvw.writerow([k, v])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def filtered_kwargs(func, **kwargs):
    return {key: value for key, value in kwargs.items()
            if key in func.__code__.co_varnames}


def is_power_of_2(x):
    assert isinstance(x, int), f"The value passed must be of \
        type int but got {type(x)}"
    return math.ceil(math.log2(x)) == math.floor(math.log2(x))


def calc_factors_prod(n):
    """
    Calculates the product of factors not divisible by two.

    Args:
        n (int): Integer to calculate factor product

    Returns:
        int: Returns the minimal product of factors not divisible by 2.
    """
    assert isinstance(n, int), \
        f"The value passed must be of type int but got type {type(n)}"
    nums = []
    for i in range(2, n + 1):
        while n % i == 0:
            nums.append(i)
            n = n / i
        if n == 1:
            break
    nums = list(filter((2).__ne__, nums))
    return np.prod(nums)


def get_min_size(img_shape):
    """
    Returns the minimum size not divisible by 2. 
    If the size is fully divisible by 2, it returns False.

    Args:
    img_shape (tuple | list | torch.Size): Image of the shape (C, H, W).

    Returns:
        bool| tuple : Returns minimum size (H, W) \
            non-divisible by 2. 
            Returns False if the image size (H, W) is a power of 2.
    """
    # Channels are not required!
    img_shape = list(img_shape)
    img_shape.pop(0)
    assert img_shape[0] == img_shape[1],\
        f"Image height and width must be the same!! \
            Got height {img_shape[0]} and width {img_shape[1]}."

    if is_power_of_2(img_shape[0]):
        return False

    return calc_factors_prod(img_shape[0]), calc_factors_prod(img_shape[0])


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
