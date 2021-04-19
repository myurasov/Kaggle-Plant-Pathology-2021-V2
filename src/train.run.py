#!/usr/bin/python

# type: ignore

# region: imports

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
from lib.keras_tb_logger import TensorBoard_Logger, lr_logger, gpu_temp_logger
from tensorflow import keras

from src.config import c
from src.generator import Generator, X2_Generator
from src.models import Model_ENBL2, Model_ENBX, Model_ENBX_NS, Model_ENBX_X2
from src.utils import create_dir, fix_random_seed

# endregion

# region: read arguments
parser = argparse.ArgumentParser(
    description="Train to get stronger",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--run",
    type=str,
    default="tst",
    help="Run name",
)

parser.add_argument(
    "--val_fold",
    type=float,
    default=1,
    help="Validation fold. Use float value <1 for a val split rather than fold.",
)

parser.add_argument(
    "--in_csv",
    type=str,
    default=c["WORK_DIR"] + "/work.csv",
    help="Input CSV file",
)

parser.add_argument(
    "--batch",
    type=int,
    default=2,
    help="Batch size",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
)

parser.add_argument(
    "--early_stop_patience",
    type=int,
    default=10,
    help="Stop after N epochs val accuracy is not improving",
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Inital LR",
)

parser.add_argument(
    "--multiprocessing",
    type=int,
    default=1,
    help="Number of generator threads",
)

parser.add_argument(
    "--model",
    type=str,
    default="enb0",
    choices=[f"enb{x}" for x in range(8)]
    + [f"enb{x}_ns" for x in range(8)]
    + [f"enb{x}_x2" for x in range(8)]
    + ["enl2"],
    help="Model",
)

parser.add_argument(
    "--amp",
    type=int,
    default=0,
    help="Enable AMP?",
)

parser.add_argument(
    "--aug",
    type=int,
    default=0,
    help="Augmentation level",
)

parser.add_argument(
    "--frozen_base",
    type=int,
    default=None,
    nargs="+",
    help="Epochs (starting with 1) in which base should be frozen. "
    + 'Eg "1 2" means two first epochs. "0" to disable freezing.',
)

parser.add_argument(
    "--lr_factor",
    type=float,
    default=0.2,
    help="Factor by which LR is multiplide after unfreezing base modela nd on plateau.",
)

parser.add_argument(
    "--lr_patience",
    type=int,
    default=3,
    help="LR reduction patience",
)

parser.add_argument(
    "--zooms",
    type=float,
    default=[1],
    nargs="+",
    help="Zoom levels for each input",
)

args = parser.parse_args()
print(f"* Arguments:\n{pformat(vars(args))}")
# endregion

# region: misc

loss = ""
final_activation = ""
n_image_inputs = min(2, len(args.zooms))

os.chdir(c["WORK_DIR"])
fix_random_seed(c["SEED"])

# turn amp on
if args.amp:
    print("* Using AMP")
    keras.mixed_precision.set_global_policy("mixed_float16")

# load dataset metadata
ds_meta = json.loads(Path(args.in_csv + ".json").read_text())

# endregion

# region: create train/val dataframes

df = pd.read_csv(args.in_csv)

if args.val_fold < 1:
    # split into sets based on fraction for val
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    val_df = df[: int(args.val_fold * df.shape[0])]
    train_df = df[val_df.shape[0] :]
else:
    # split into sets based on folds
    val_df = df[df.fold == int(args.val_fold)]
    train_df = df[df.fold != int(args.val_fold)]
    args.run = f"{args.run}.fold_{args.val_fold:.0f}"

assert val_df.shape[0] + train_df.shape[0] == df.shape[0]
print(f"* Training set size: {train_df.shape[0]}")
print(f"* Validation set size: {val_df.shape[0]}")

# endregion

# region: prepare paths

td_dir = f"/app/.tensorboard/{args.run}"
create_dir(td_dir, remove=True)

checkpoint_path = f"{c['WORK_DIR']}/models/{args.run}"
create_dir(f"{c['WORK_DIR']}/models", remove=False)

# endregion

# region: problem type-dependent params

assert ds_meta["args"]["labels_mode"] in ["multilabel", "multiclass"]

if ds_meta["args"]["labels_mode"] == "multiclass":
    final_activation = "softmax"
    loss = "categorical_crossentropy"
elif ds_meta["args"]["labels_mode"] == "multilabel":
    final_activation = "sigmoid"
    loss = "binary_crossentropy"

print(f"* Problem type: {ds_meta['args']['labels_mode'] }")
print(f"* Loss: {loss}")
print(f"* Final activation: {final_activation}")

# endregion

# region: create model

model_options = {
    "n_classes": len(ds_meta["classes"]),
    "final_activation": final_activation,
    "augmentation": args.aug,
}

if args.model == "enl2":

    # L2 variant
    model_buider = Model_ENBL2(**model_options)

else:

    # Bx variant
    model_en_variant, model_suffix = re.search(
        "enb(\\d)(_ns|_x2)?", args.model
    ).groups()

    if model_suffix is None:
        model_buider = Model_ENBX(variant=int(model_en_variant), **model_options)
    elif model_suffix == "_ns":
        model_buider = Model_ENBX_NS(variant=int(model_en_variant), **model_options)
    elif model_suffix == "_x2":
        model_buider = Model_ENBX_X2(variant=int(model_en_variant), **model_options)

model_buider.create()

print("* Augmentation level:", args.aug)
print(f"* Input size: {model_buider.input_shape}")
print(f"* Output size: {np.array(ds_meta['classes']).shape}")

# endregion

# region: save train run metadata

train_meta_file = f"{checkpoint_path}.meta.json"
print(
    json.dumps(
        {
            "loss": loss,
            "run": args.run,
            "args": vars(args),
            "augmentation": args.aug,
            "cmd": " ".join(sys.argv),
            "val_fold": args.val_fold,
            "classes": ds_meta["classes"],
            "val_samples": val_df.shape[0],
            "train_samples": train_df.shape[0],
            "final_activation": final_activation,
            "image_size": model_buider.input_shape,
            "problem_type": ds_meta["args"]["labels_mode"],
        },
        indent=4,
    ),
    file=open(train_meta_file, "w"),
)
print(f"* Written train run metadata to {train_meta_file}")

# endregion

# region: create generators
train_g, val_g = None, None

if n_image_inputs == 1:

    train_g = Generator(
        df=train_df,
        shuffle=True,
        zoom=args.zooms[0],
        batch_size=args.batch,
        augmentation_options=None,
        image_output_size=model_buider.input_shape,
    )

    val_g = Generator(
        df=val_df,
        shuffle=True,
        zoom=args.zooms[0],
        batch_size=args.batch,
        augmentation_options=None,
        image_output_size=model_buider.input_shape,
    )

elif n_image_inputs == 2:

    train_g = X2_Generator(
        df=train_df,
        shuffle=True,
        zooms=args.zooms,
        batch_size=args.batch,
        augmentation_options=None,
        image_output_size=model_buider.input_shape,
    )

    val_g = X2_Generator(
        df=val_df,
        shuffle=True,
        zooms=args.zooms,
        batch_size=args.batch,
        augmentation_options=None,
        image_output_size=model_buider.input_shape,
    )


# endregion

# region: callbacks

print(f"* Run name: {args.run}")

callbacks = []

callbacks.append(
    keras.callbacks.EarlyStopping(
        patience=args.early_stop_patience,
        restore_best_weights=True,
        verbose=1,
    )
)

callbacks.append(
    TensorBoard_Logger(
        log_dir=td_dir,
        histogram_freq=0,
        loggers=[
            lr_logger,
            gpu_temp_logger,
        ],
    )
)

callbacks.append(
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-9,
        verbose=1,
    )
)

callbacks.append(
    keras.callbacks.ModelCheckpoint(
        checkpoint_path + ".checkpoint/",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
    )
)

# endregion

# region: finally do something useful


def _train(start_epoch_1, epochs):
    if epochs > 0:
        model_buider.model.fit(
            x=train_g,
            validation_data=val_g,
            epochs=start_epoch_1 + epochs - 1,
            initial_epoch=start_epoch_1 - 1,
            callbacks=callbacks,
            verbose=1,
            workers=args.multiprocessing,
            max_queue_size=1,
            use_multiprocessing=args.multiprocessing > 1,
        )
    return start_epoch_1 + epochs - 1


end_epoch_1 = 0
model_buider.model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss=loss
)

# freeze/unfreeze base
if args.frozen_base is not None and len(args.frozen_base) > 1:
    # pre-freeze epochs
    end_epoch_1 = _train(1, args.frozen_base[0] - 1)

    model_buider.freeze_base()
    model_buider.model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(model_buider.model.optimizer.learning_rate)
        ),
        loss=loss,
    )
    print("* Base model frozen")

    # frozen epochs
    end_epoch_1 = _train(
        end_epoch_1 + 1, (1 + args.frozen_base[1] - args.frozen_base[0])
    )

    model_buider.unfreeze_base()
    model_buider.model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=float(model_buider.model.optimizer.learning_rate)
        ),
        loss=loss,
    )
    print("* Base model unfrozen")

    # reduce LR after unfreezing
    lr = float(model_buider.model.optimizer.learning_rate)
    print(f"* Reducing LR from {lr:e} to {lr*args.lr_factor:e}")
    keras.backend.set_value(
        model_buider.model.optimizer.learning_rate, lr * args.lr_factor
    )

# unfrozen epochs
_train(end_epoch_1 + 1, args.epochs - end_epoch_1)

# load best weights
model_buider.model.load_weights(checkpoint_path + ".checkpoint/")

# save in a single file for kaggle
model_buider.model.save(checkpoint_path + ".h5", save_format="h5")

# cleanup checkpoints
shutil.rmtree(checkpoint_path + ".checkpoint/")

# endregion
