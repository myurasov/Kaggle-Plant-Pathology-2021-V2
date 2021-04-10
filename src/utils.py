import hashlib
import os
import random
import shutil
from collections import namedtuple

import numpy as np


def create_dir(dir, remove=True):
    if remove:
        shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir, exist_ok=True)


def rename_dir(dir1, dir2, remove_dir2=True):
    if remove_dir2:
        shutil.rmtree(dir2, ignore_errors=True)
    os.rename(dir1, dir2)


def dict_to_struct(d):
    return namedtuple("Struct", d.keys())(*d.values())


def fix_random_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)


def list_indexes(list, cols=None):
    """
    Creates a dictionary mapping values to indexes
    """
    if cols is None:
        cols = list
    return dict([(x, list.index(x)) for x in cols])


def create_tensorboard_run_dir(run):
    """
    Creates a directory to log tensorboard data into
    """
    tb_log_dir = f"/app/.tensorboard/{run}"
    shutil.rmtree(tb_log_dir, ignore_errors=True)
    return tb_log_dir


def md5_file(path):
    """
    Calculate file MD5 hash
    """

    with open(path, "rb") as f:
        hash = hashlib.md5()

        while chunk := f.read(2 << 20):
            hash.update(chunk)

    return hash.hexdigest()
