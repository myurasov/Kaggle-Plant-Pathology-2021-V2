#!/usr/bin/python

import argparse
import shutil
from pprint import pformat

import pandas as pd
from tqdm import tqdm

from src.config import c as c

from utils import create_dir

# region: read arguments
parser = argparse.ArgumentParser(
    description="Add external data sources",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--pp_2020",
    action="store_true",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")

# endregion

# region: pp 2020 dataset

if args.pp_2020:

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

# endregion
