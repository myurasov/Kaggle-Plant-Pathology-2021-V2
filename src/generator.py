import os
from hashlib import md5

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (
    random_brightness,
    random_rotation,
    random_shear,
    random_shift,
    random_zoom,
)

from src.config import c as c

default_image_augmenation_options = {
    "rotation_max_degrees": 45,
    "zoom_range": (0.75, 1.25),
    "shift_max_fraction": {"w": 0.25, "h": 0.25},
    "shear_max_degrees": 45,
    "brightness_range": (0.5, 1.5),
}


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        df,
        shuffle=True,
        batch_size=32,
        image_output_size=c["IMAGE_SIZE"],
        image_dirs=c["SRC_IMAGE_DIRS"],
        cache_dir=c["WORK_DIR"] + "/images_cache",
        augmentation_options=default_image_augmenation_options,
    ):
        self.df = df
        self.shuffle = shuffle
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.image_dirs = image_dirs
        self.image_output_size = image_output_size
        self.augmentation_options = augmentation_options

        # create cache dir for images
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

        # shuffle data, also repeated after each epoch if needed
        if self.shuffle:
            self._shuffle()

        # calc number of samples and batches
        self.n_samples = self.df.shape[0]
        self.n_batches = self.n_samples // self.batch_size

        # store output parameters
        x0, y0 = self.get_one(0)
        self.x_dtype = x0.dtype
        self.x_shape = x0.shape
        self.y_dtype = y0.dtype
        self.y_shape = y0.shape

    def __len__(self):
        """
        Length in batches
        """
        return self.n_batches

    def __getitem__(self, b_ix):
        """
        Produce batch, by batch index
        """

        assert b_ix < self.n_batches

        b_X = np.zeros(
            (self.batch_size,) + self.x_shape,
            dtype=self.x_dtype,
        )

        b_Y = np.zeros(
            (self.batch_size,) + self.y_shape,
            dtype=self.y_dtype,
        )

        for i in range(self.batch_size):
            b_X[i], b_Y[i] = self.get_one(i + self.batch_size * b_ix)

        return (b_X, b_Y)

    def get_one(self, ix):
        """
        Get single item by absolute index
        """

        image = self.df.image[ix]

        # find image file
        src_file = ""
        for image_dir in self.image_dirs:
            src_file = f"{image_dir}/{image}.jpg"
            if os.path.exists(src_file):
                break
            src_file = f"{image_dir}/{image}"
            if os.path.exists(src_file):
                break

        x = None

        # unique id for a cache file
        # eg /app/_data/src/train_images/a583cadede382c70.jpg_(224,224)
        file_id_string = src_file + "_" + str(tuple(self.image_output_size[:2]))
        cache_id = md5(file_id_string.encode()).hexdigest()

        # cache file location
        cache_file = f"{self.cache_dir}/{cache_id[0]}/{cache_id[1]}/{cache_id}.npy"

        # read from cache
        if self.cache_dir is not None:
            if os.path.exists(cache_file):
                x = np.load(cache_file)

        # x wasn't read from cache
        if x is None:
            x = Image.open(src_file)
            x = x.resize(self.image_output_size[:2], resample=Image.BICUBIC)
            x = np.array(x).astype(np.uint8)

            # save to cache
            if self.cache_dir is not None:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.save(cache_file[:-4], x)

        # verify that cached data has the corect dimensions
        # np array has HxWXC layout, unlike PIL Image's WxHxC
        assert x.shape == (self.image_output_size[1], self.image_output_size[0], 3)

        # augment
        if self.augmentation_options is not None:
            x = self._augment_image(x)

        y = np.array(self.df.Y[ix].split(",")).astype(np.float16)

        return x, y

    def _augment_image(self, x):
        """
        Randomply augment image
        """

        assert x.dtype == np.uint8

        # common options
        co = {
            "row_axis": 0,
            "col_axis": 1,
            "channel_axis": 2,
            # can be 'constant', 'nearest', 'reflect', 'wrap'
            "fill_mode": "nearest",
            "cval": 0.0,
        }

        o = self.augmentation_options

        x = random_rotation(x, o["rotation_max_degrees"], **co)
        x = random_shear(x, o["shear_max_degrees"], **co)
        x = random_shift(
            x, o["shift_max_fraction"]["w"], o["shift_max_fraction"]["h"], **co
        )
        x = random_zoom(x, o["zoom_range"], **co)
        x = random_brightness(x, o["brightness_range"])

        return x.astype(np.uint8)

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle()

    def _shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
