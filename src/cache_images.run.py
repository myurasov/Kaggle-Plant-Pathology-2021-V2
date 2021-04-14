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
    default=[600, 600],
    nargs="+",
    help="Target image size (WxH)",
)

parser.add_argument(
    "--zoom",
    type=float,
    default=1,
    help="Zoom factor",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

df = pd.read_csv(args.input_csv)

g = Generator(
    df=df,
    batch_size=1,
    shuffle=False,
    zoom=args.zoom,
    augmentation_options=None,
    image_output_size=tuple(args.size),
)


def _mapping(i):
    g.__getitem__(i)


with Pool(cpu_count()) as pool:
    list(
        tqdm(
            pool.imap(
                _mapping,
                range(df.shape[0]),
            ),
            total=df.shape[0],
            smoothing=0
        )
    )
