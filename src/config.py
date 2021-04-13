c = {}

c["DATA_DIR"] = "/app/_data"

# where to search for csvs
c["SRC_CSVS"] = [
    f"{c['DATA_DIR']}/competition_data/train.csv",
    f"{c['DATA_DIR']}/work/extra_data/*.csv",
]

# where to search for images
c["SRC_IMAGE_DIRS"] = [
    f"{c['DATA_DIR']}/competition_data/train_images",
    f"{c['DATA_DIR']}/work/extra_data/images",
]

# where to output preprocessed data
c["WORK_DIR"] = "/app/_data/work"

# extra data dir
# should have sub-structure: <set_name>/<label.comment>
c["AUX_DATA_DIR"] = "/app/_data/aux_data"
