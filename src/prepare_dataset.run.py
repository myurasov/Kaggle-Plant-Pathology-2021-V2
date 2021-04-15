#!/usr/bin/python

import argparse
from glob import glob
import json
import math
import os
from pprint import pformat

import numpy as np
import pandas as pd

from src.config import c
from utils import fix_random_seed, list_indexes

# region: read arguments
parser = argparse.ArgumentParser(
    description="Prepare dataset: - join multiple CSVs, - add labels, - add folds",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--input_csvs",
    type=str,
    default=c["SRC_CSVS"],
    nargs="+",
    help="Input CSV files. Wilcards can be used.",
)

parser.add_argument(
    "--out_csv",
    type=str,
    default=c["WORK_DIR"] + "/work.csv",
    help="Output CSV file",
)

parser.add_argument(
    "--folds",
    type=int,
    default=5,
    help="Number of folds",
)

parser.add_argument(
    "--labels_mode",
    type=str,
    default="multilabel",
    choices=["multilabel", "multiclass"],
    help="How to treat labels",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion


def read_csvs(wildcards):
    dfs = []

    for wildcard in wildcards:
        for csv_file in glob(wildcard):
            df = pd.read_csv(csv_file)

            if "original_file" not in df.columns:
                df["original_file"] = ["_none_"] * df.shape[0]  # type: ignore

            print(f'* Found CSV file "{csv_file}" with {df.shape[0]} lines')
            dfs.append(df)

    df = pd.concat(dfs)
    print(f"* Total {df.shape[0]} lines")

    return df


def add_Y(df, mode):
    """
    Convert labels to one-hot representation
    Returns dict mapping to 1-hot label
    """

    assert mode in ["multilabel", "multiclass"]

    if "Y" in df.columns:
        del df["Y"]

    Y = []
    classes = []
    class_1hs = []

    if mode == "multilabel":
        classes = " ".join(set(df.labels)).split(" ")
        classes = sorted(list(set(classes)))
    elif mode == "multiclass":
        classes = map(lambda x: " ".join(sorted(x.lower().split(" "))), list(df.labels))
        classes = sorted(list(set(list(classes))))

    class_ixs = list_indexes(classes)

    for label in df.labels:
        y = np.zeros((len(classes)), dtype=np.uint8)
        y[list(map(lambda x: class_ixs[x], label.split(" ")))] = 1
        Y.append(",".join(y.astype(str)))
        class_1hs.append(y)

    df.insert(df.shape[1], "Y", Y)

    # insert 1-hot class columns
    class_1hs = np.array(class_1hs)
    for i, cls in enumerate(classes):
        df[f"_{cls}_"] = class_1hs[:, i]

    print(f"* Added {len(classes)} labels: {classes}")

    return df, classes


def add_folds(df, n_folds) -> pd.DataFrame:
    """
    Add "fold" column to CSV file.
    Data is equally sampled from bins with same label combination.
    """

    if "fold" in df.columns:
        del df["fold"]

    print(f"* Adding {n_folds} folds...")

    # normalize labels to find uniques combinations
    labels_normalized = np.array(
        list(map(lambda x: " ".join(sorted(x.lower().split(" "))), list(df.labels)))
    )

    # split items into bins with the same label
    # bins is a list of indexes
    bins = []
    for label in set(labels_normalized):
        bins.append(np.where(labels_normalized == label)[0])
        np.random.shuffle(bins[-1])

    print(f"* Total {len(bins)} bins")

    folds = np.zeros((df.shape[0]), dtype=np.int32)

    for fold in range(1, 1 + n_folds):
        for bin in bins:
            bin_fold_len = int(math.ceil(len(bin) / n_folds))
            bin_fold_indexes = bin[bin_fold_len * (fold - 1) : bin_fold_len * fold]
            folds[bin_fold_indexes] = fold

    fold_lens = list(
        map(lambda x: np.where(folds == x + 1)[0].shape[0], range(n_folds))
    )
    print(f"* Fold sizes: {fold_lens}")

    df.insert(df.shape[1], "fold", folds)
    return df


fix_random_seed()
os.chdir(c["WORK_DIR"])

# read multiple csvs into single dataframe
df = read_csvs(args.input_csvs)

# add folds
df = add_folds(df, args.folds)

# add Y column
df, classes = add_Y(df, args.labels_mode)

# create output directory
if '/' in args.out_csv:
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

# save csv
df.to_csv(args.out_csv, index=False)
print(f"* Written new CSV to {args.out_csv}")

# save metadata
meta_file = args.out_csv + ".json"
print(
    json.dumps(
        {
            "classes": classes,
            "args": vars(args),
        },
        indent=4,
    ),
    file=open(meta_file, "w"),
)
print(f"* Written metadata to {meta_file}")
