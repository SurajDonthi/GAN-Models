import csv
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
