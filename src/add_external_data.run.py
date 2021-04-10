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

# create extra data dir
extra_data_dir = f"{c['WORK_DIR']}/extra_data"
images_out_dir = f"{extra_data_dir}/images"
create_dir(images_out_dir, remove=True)
print(f"* Extra data dir: {extra_data_dir}")

# region: pp 2020 dataset


def _add_pp_2020():

    pp20_dir = f"{c['DATA_DIR']}/pp_2020"
    pp20_new_images_dir = f"{pp20_dir}/new_images"
    create_dir(pp20_new_images_dir, remove=True)

    # originally: "healthy", "multiple_diseases", "rust", "scab"
    pp20_classes = ["healthy", "complex", "rust", "scab"]

    df = pd.read_csv(f"{pp20_dir}/train.csv")
    new_labels = []
    new_images_files = []
    new_df = pd.DataFrame()

    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):

        # rename image files so they are unique across all external sources
        image_path = f"{pp20_dir}/images/{row.image_id}.jpg"
        new_image_file = "__pp_2020__." + row.image_id + ".jpg"
        new_image_path = f"{pp20_new_images_dir}/{new_image_file}"
        shutil.copyfile(image_path, new_image_path)
        new_images_files.append(new_image_file)

        # read labels
        label = []
        for i, has_class in enumerate(list(row[1:])):
            if has_class:
                label.append(pp20_classes[i])
        new_labels.append(" ".join(label))

    # normalize labels to match current competition
    new_labels = list(map(lambda x: " ".join(sorted(x.lower().split(" "))), new_labels))

    new_df["image"] = new_images_files
    new_df["labels"] = new_labels
    new_df.to_csv(f"{c['WORK_DIR']}/external.pp_2020.csv")


if args.pp_2020:
    _add_pp_2020()

# endregion

# region: google images


# copy file with unique name into extrenal images dir
def _copy_file_to_external_images(path):

    md5 = md5_file(path)
    ext = os.path.splitext(path)[1].lower()
    ext = ".jpg" if ".jpeg" == ext else ext

    new_file_name = f"{md5}{ext}"
    new_file_path = f"{images_out_dir}/{new_file_name}"

    if not os.path.isfile(new_file_path):
        shutil.copyfile(path, new_file_path)
        return new_file_name  # copied

    return None  # not copied


def _add_aux_data():
    labels = []
    new_file_names = []
    aux_data_dir = f"{c['DATA_DIR']}/aux_data"
    dirs = glob(f"{aux_data_dir}/*/*")

    for dir in dirs:
        label = re.findall(".*/([a-z_]+)(\\.\\d+)?", dir, re.I)[0][0]
        print(f'* Found aux data in "{dir}" with label "{label}"')
        print(f'* Copying files to "{images_out_dir}"...')

        for file in tqdm(glob(f"{dir}/*")):
            if new_file_name := _copy_file_to_external_images(file):
                new_file_names.append(new_file_name)
                labels.append(label)

    # save csv
    df = pd.DataFrame()
    df["image"] = new_file_names
    df["labels"] = labels
    df.to_csv(f"{extra_data_dir}/aux.csv", index=False)


if args.aux:
    _add_aux_data()

# endregion
