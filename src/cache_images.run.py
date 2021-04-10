#!/usr/bin/python

import argparse
from multiprocessing import Pool, cpu_count
from pprint import pformat

import pandas as pd
from tqdm import tqdm

from src.config import c as c
from src.generator import Generator

# region: read arguments
parser = argparse.ArgumentParser(
    description="Precache decoded and resized images",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--input_csv",
    type=str,
    default=c["WORK_DIR"] + "/work.csv",
    help="Input CSV file",
)

parser.add_argument(
    "--size",
    type=int,
    default=c["IMAGE_SIZE"],
    nargs="+",
    help="Target image size (WxH)",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

g = Generator(
    df=pd.read_csv(f"{c['WORK_DIR']}/work.csv"),
    image_output_size=tuple(args.size),
    augmentation_options=None,
    shuffle=False
)


def _mapping(i):
    x, y = g.get_one(i)


with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _mapping,
                range(g.n_samples),
            ),
            total=g.n_samples,
        )
    )
