#!/usr/bin/python

import argparse
import os
import re
import shutil
from glob import glob
from pprint import pformat

import pandas as pd
from tqdm import tqdm

from src.config import c
from utils import create_dir, md5_file

# region: read arguments
parser = argparse.ArgumentParser(
    description="Add external data sources",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--pp_2020",
    action="store_true",
)

parser.add_argument(
    "--aux",
    action="store_true",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")

# endregion


def _copy_file_to_external_images(path):
    """copy file with unique name into extrenal images dir"""

    md5 = md5_file(path)
    ext = os.path.splitext(path)[1].lower()
    ext = ".jpg" if ".jpeg" == ext else ext

    new_file_name = f"{md5}{ext}"
    new_file_path = f"{images_out_dir}/{new_file_name}"

    if not os.path.isfile(new_file_path):
        shutil.copyfile(path, new_file_path)
        return new_file_name  # copied

    return None  # not copied


# create extra data dir
extra_data_dir = f"{c['WORK_DIR']}/extra_data"
images_out_dir = f"{extra_data_dir}/images"
create_dir(extra_data_dir, remove=True)
create_dir(images_out_dir, remove=False)
print(f"* Extra data dir: {extra_data_dir}")

# region: pp 2020 dataset


def _add_pp_2020():

    pp20_dir = f"{c['DATA_DIR']}/pp_2020"

    in_df = pd.read_csv(f"{pp20_dir}/train.csv")

    # originally: "healthy", "multiple_diseases", "rust", "scab"
    pp20_classes = ["healthy", "complex", "rust", "scab"]

    new_labels = []
    new_files = []
    orig_paths = []

    print("* Adding PP 2020 competition data...")

    for ix, row in tqdm(in_df.iterrows(), total=in_df.shape[0]):
        image_path = f"{pp20_dir}/images/{row.image_id}.jpg"

        if new_image_file := _copy_file_to_external_images(image_path):
            new_files.append(new_image_file)

            # append image labels
            label = []
            for i, has_class in enumerate(list(row[1:])):
                if has_class:
                    label.append(pp20_classes[i])
            new_labels.append(" ".join(label))
            orig_paths.append(image_path)

    # normalize labels to match current competition
    new_labels = list(map(lambda x: " ".join(sorted(x.lower().split(" "))), new_labels))

    out_df = pd.DataFrame()
    out_df["image"] = new_files
    out_df["labels"] = new_labels
    out_df["original_file"] = orig_paths
    out_df.to_csv(f"{extra_data_dir}/pp_2020.csv", index=False)


if args.pp_2020:
    _add_pp_2020()

# endregion

# region: google images


def _add_aux_data():

    labels = []
    new_files = []
    orig_paths = []

    aux_data_dir = f"{c['DATA_DIR']}/aux_data"
    dirs = glob(f"{aux_data_dir}/*/*")

    for dir in dirs:
        label = re.findall(".*/([a-z_]+)(\\.\\d+)?", dir, re.I)[0][0]
        print(f'* Found aux data in "{dir}" with label "{label}"')

        for file in tqdm(glob(f"{dir}/*")):
            if new_file_name := _copy_file_to_external_images(file):
                orig_paths.append(file)
                new_files.append(new_file_name)
                labels.append(label)

    # save csv
    df = pd.DataFrame()
    df["image"] = new_files
    df["labels"] = labels
    df["original_file"] = orig_paths
    df.to_csv(f"{extra_data_dir}/aux.csv", index=False)


if args.aux:
    _add_aux_data()

# endregion
