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
    "--pp20_test_csv",
    type=str,
    default=None,
    help="Merge PP20 labeled test data from CSV at this location",
)

parser.add_argument(
    "--pp20",
    action="store_true",
)

parser.add_argument(
    "--aux_dirs",
    default=[],
    type=str,
    nargs="+",
    help=f"Aux data directories (inside {c['AUX_DATA_DIR']})",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")

# args.pp20_test_csv = "/app/res/test_20_labeled.csv"
# args.pp20 = True

# endregion

PP20_DIR = f"{c['DATA_DIR']}/pp_2020"
PP20_TRAIN_PLUS_TEST_CSV = f"{PP20_DIR}/_train_plus_test.csv"

# create extra data dir
extra_data_dir = f"{c['WORK_DIR']}/extra_data"
images_out_dir = f"{extra_data_dir}/images"
create_dir(extra_data_dir, remove=False)
create_dir(images_out_dir, remove=False)
print(f"* Extra data dir: {extra_data_dir}")

# cleanup: remove pp20 train+test dataset
if os.path.exists(PP20_TRAIN_PLUS_TEST_CSV):
    os.remove(PP20_TRAIN_PLUS_TEST_CSV)


def _copy_file_to_external_images(path):
    """copy file with unique name into extrenal images dir"""

    md5 = md5_file(path)
    ext = os.path.splitext(path)[1].lower()
    ext = ".jpg" if ".jpeg" == ext else ext

    new_file_name = f"{md5}{ext}"
    new_file_path = f"{images_out_dir}/{new_file_name}"

    if not os.path.isfile(new_file_path):
        shutil.copyfile(path, new_file_path)

    return new_file_name


def _add_pp_2020_test_csv():

    df_train = pd.read_csv(f"{PP20_DIR}/train.csv")
    df_test = pd.read_csv(args.pp20_test_csv)

    # reformat natasha's csv to be closer to train.csv
    df_test = df_test.rename(
        columns={"image": "image_id", "complex": "multiple_diseases"}
    )
    df_test = df_test.drop(["Unnamed: 0", "labels"], axis=1)
    df_test["image_id"] = list(
        map(lambda x: x.replace(".jpg", ""), df_test["image_id"])
    )

    # merge train and test csvs
    df = pd.concat([df_train, df_test]).fillna(0)

    # save combined one
    df.to_csv(PP20_TRAIN_PLUS_TEST_CSV, index=False)

    print(
        f'* Saved train+test CSV for PP20 data to "{PP20_TRAIN_PLUS_TEST_CSV}"'
        + f" with extra {df_test.shape[0]} entries"
    )


if args.pp20_test_csv is not None:
    _add_pp_2020_test_csv()


# region: pp 2020 dataset


def _add_pp_2020():

    in_csv_file = None

    # if train+test file exists, use it
    try:
        in_csv_file = PP20_TRAIN_PLUS_TEST_CSV
        in_df = pd.read_csv(in_csv_file, index_col="image_id")
    except Exception:
        in_csv_file = f"{PP20_DIR}/train.csv"
        in_df = pd.read_csv(in_csv_file, index_col="image_id")
    finally:
        print(f'* Using CSV "{in_csv_file}"')

    in_df = in_df.rename(columns={"multiple_diseases": "complex"})
    pp20_classes = list(in_df.columns)

    new_labels = []
    new_files = []
    orig_paths = []

    print("* Adding PP 2020 competition data...")

    for ix, row in tqdm(in_df.iterrows(), total=in_df.shape[0]):
        image_path = f"{PP20_DIR}/images/{ix}.jpg"

        if new_image_file := _copy_file_to_external_images(image_path):
            new_files.append(new_image_file)

            # append image labels
            label = []
            for i, has_class in enumerate(list(row)):
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


if args.pp20:
    _add_pp_2020()

# endregion

# region: google images


def _add_aux_data(aux_dir):

    labels = []
    new_files = []
    orig_paths = []

    label_dirs = glob(f"{aux_dir}/*")

    for label_dir in label_dirs:
        label = re.findall("([a-z_]+)(\\.[\\w_-]+)?$", label_dir, re.I)[0][0]
        print(f'* Found aux data in "{label_dir}" with label "{label}"')

        for file in tqdm(glob(f"{label_dir}/*")):
            if new_file_name := _copy_file_to_external_images(file):
                orig_paths.append(file)
                new_files.append(new_file_name)
                labels.append(label)

    # save csv

    csv_file = f"{extra_data_dir}/aux.csv"

    df_existing = pd.DataFrame()
    if os.path.isfile(csv_file):
        df_existing = pd.read_csv(csv_file)

    df = pd.DataFrame()
    df["image"] = new_files
    df["labels"] = labels
    df["original_file"] = orig_paths

    df = pd.concat([df_existing, df])
    df.to_csv(csv_file, index=False)


if len(args.aux_dirs) > 0:
    for aux_dir_pattern in args.aux_dirs:
        aux_dir_pattern = c["AUX_DATA_DIR"] + "/" + aux_dir_pattern
        aux_dirs = glob(aux_dir_pattern)
        for aux_dir in aux_dirs:
            _add_aux_data(aux_dir)

# endregion
