#!/usr/bin/python

# type: ignore

import argparse
import os
from multiprocessing import Pool, cpu_count
from pprint import pformat

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import c

# region: read arguments
parser = argparse.ArgumentParser(
    description="Drop dupliucvate images from dataset CSV file",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--image_dirs",
    type=str,
    default=c["SRC_IMAGE_DIRS"],
    nargs="+",
    help="Directories to read images from",
)

parser.add_argument(
    "--input_csv",
    type=str,
    default=c["WORK_DIR"] + "/work.csv",
    help="Input CSV file",
)

parser.add_argument(
    "--out_csv",
    type=str,
    default=c["WORK_DIR"] + "/work.csv",
    help="Output CSV file",
)

parser.add_argument(
    "--threshold",
    type=int,
    default=0,
    help="ImageHash comparision threshold",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion


# map image file name to a series of hashes
def _calculate_image_hashes(image_filename):
    # find image file
    for image_dir in args.image_dirs:
        image_path = f"{image_dir}/{image_filename}"
        if os.path.exists(image_path):
            break

    img = Image.open(image_path).resize((256, 256), Image.BICUBIC)
    return [imagehash.average_hash(img)]


os.chdir(c["WORK_DIR"])
df = pd.read_csv(args.input_csv)
print("* Calculating hashes...")

# # calculate hashes
with Pool(cpu_count()) as pool:

    hashes = list(
        tqdm(
            pool.imap(
                _calculate_image_hashes,
                list(df.image),
            ),
            total=len(df),
            smoothing=0,
        )
    )

hashes = np.array(hashes)

# ###
# df.insert(df.shape[0], 'hash', hashes)
# df.to_csv('hashes.csv', index=False)
# ###

# drop duplicates

print("* Searching for duplicates...")

ixs_to_drop = set()

for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
    if ix not in ixs_to_drop:

        diffs = np.min(np.abs(hashes - hashes[ix]), axis=1)
        close_ixs = np.where(diffs <= args.threshold)[0]

        if len(close_ixs) > 1:
            labels = list(df.labels[close_ixs])
            print("* Duplicates detected:", list(df.image[close_ixs]), labels)

            if len(set(labels)) == len(labels):
                # different labels - remove all
                ixs_to_drop.update(close_ixs)
            else:
                # same labels - remove others
                ixs_to_drop.update(close_ixs[1:])


# write csv file with rows dropped

print(f"* Dropping {len(ixs_to_drop)} rows...")
df = df.drop(list(ixs_to_drop))

if "/" in args.out_csv:
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

df.to_csv(args.out_csv, index=False)

print(
    f"* Written new CSV to {args.out_csv} with {len(df)} rows out of {len(hashes)} originally"
)
