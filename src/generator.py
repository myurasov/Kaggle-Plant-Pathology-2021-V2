import os
from hashlib import md5

import numpy as np
import pandas as pd
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
        zoom=1,
        shuffle=True,
        batch_size=32,
        image_output_size=(600, 600),
        image_dirs=c["SRC_IMAGE_DIRS"],
        cache_dir=c["WORK_DIR"] + "/images_cache",
        augmentation_options=default_image_augmenation_options,
    ):

        self._df = df
        self._zoom = zoom
        self._shuffle = shuffle
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._image_dirs = image_dirs
        self._image_output_size = image_output_size
        self._augmentation_options = augmentation_options

        # create cache dir for images
        if self._cache_dir is not None:
            os.makedirs(self._cache_dir, exist_ok=True)

        # shuffle data, also repeated after each epoch if needed
        if self._shuffle:
            self._shuffle_samples()

        # calc number of samples and batches
        self._n_samples = self._df.shape[0]
        self._n_batches = self._n_samples // self._batch_size

        # store output parameters
        x0, y0 = self._get_one(0)
        self._x_dtype = x0.dtype
        self._x_shape = x0.shape
        self._y_dtype = y0.dtype
        self._y_shape = y0.shape

    def __len__(self):
        """
        Length in batches
        """
        return self._n_batches

    def __getitem__(self, b_ix):
        """
        Produce batch, by batch index
        """

        assert b_ix < self._n_batches

        b_X = np.zeros(
            (self._batch_size,) + self._x_shape,
            dtype=self._x_dtype,
        )

        b_Y = np.zeros(
            (self._batch_size,) + self._y_shape,
            dtype=self._y_dtype,
        )

        for i in range(self._batch_size):
            b_X[i], b_Y[i] = self._get_one(i + self._batch_size * b_ix)

        return (b_X, b_Y)

    def on_epoch_end(self):
        if self._shuffle:
            self._shuffle_samples()

    def _find_image(self, file_name):

        # find image file in image_dirs
        for image_dir in self._image_dirs:

            # do not add .jpg extension
            path = f"{image_dir}/{file_name}"
            if os.path.exists(path):
                return path

            # add .jpg extension
            path = f"{image_dir}/{file_name}.jpg"
            if os.path.exists(path):
                return path

        raise Exception(f'Can\'t find "{file_name}"')

    def _read_image(self, src_file, zoom=1):
        """Read resized image - either from cache or source file"""

        # unique id for a cache file
        # eg /app/_data/src/train_images/a583cadede382c70.jpg_size=(224,224)_zoom=1.41
        size_cache_key = f"size={tuple(self._image_output_size[:2])}"
        zoom_cache_key = f"zoom={zoom:.2f}"
        file_id_string = f"{src_file}_{size_cache_key}_{zoom_cache_key}"
        cache_id = md5(file_id_string.encode()).hexdigest()

        # cache file location
        cache_file = f"{self._cache_dir}/{cache_id[0]}/{cache_id[1]}/{cache_id}.npy"

        x = None

        # read from cache
        if self._cache_dir is not None:
            if os.path.exists(cache_file):
                x = np.load(cache_file)

        # x wasn't read from cache
        if x is None:
            x = Image.open(src_file)

            x = x.resize(
                (
                    int(self._image_output_size[0] * self._zoom),
                    int(self._image_output_size[1] * self._zoom),
                ),
                resample=Image.BICUBIC,
            )

            # do center crop for zoomed-in versions
            if self._zoom > 1:
                left = (x.width - self._image_output_size[0]) / 2
                top = (x.height - self._image_output_size[1]) / 2
                right = (x.width + self._image_output_size[0]) / 2
                bottom = (x.height + self._image_output_size[1]) / 2
                x = x.crop((left, top, right, bottom))

            x = np.array(x).astype(np.uint8)

            # save to cache
            if self._cache_dir is not None:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.save(cache_file[:-4], x)

        # verify that cached data has the corect dimensions
        # np array has HxWXC layout, unlike PIL Image's WxHxC
        assert x.shape == (self._image_output_size[1], self._image_output_size[0], 3)

        return x

    def _get_one(self, ix):
        """
        Get single item by absolute index
        """

        # x

        image = self._df.image.iloc[ix]
        src_file = self._find_image(image)
        x = self._read_image(src_file, zoom=self._zoom)

        if self._augmentation_options is not None:
            x = self._augment_image(x)

        # y

        y = self._df.Y.iloc[ix].split(",")
        y = np.array(y).astype(np.float16)

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
            "fill_mode": "reflect",
            "cval": 0.0,
        }

        o = self._augmentation_options

        x = random_rotation(x, o["rotation_max_degrees"], **co)
        x = random_shear(x, o["shear_max_degrees"], **co)
        x = random_shift(
            x, o["shift_max_fraction"]["w"], o["shift_max_fraction"]["h"], **co
        )
        x = random_zoom(x, o["zoom_range"], **co)
        x = random_brightness(x, o["brightness_range"])

        return x.astype(np.uint8)

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)


class X2_Generator(keras.utils.Sequence):
    def __init__(self, df, zooms=[1.0, 2.0], shuffle=True, **kwargs):

        self._df = df
        self._shuffle = shuffle

        self._g1 = Generator(df=self._df, zoom=zooms[0], shuffle=False, **kwargs)
        self._g2 = Generator(df=self._df, zoom=zooms[1], shuffle=False, **kwargs)

    def __len__(self):
        return self._g1.__len__()

    def __getitem__(self, ix):
        X1, Y = self._g1.__getitem__(ix)
        X2, Y = self._g2.__getitem__(ix)
        return [X1, X2], Y

    def on_epoch_end(self):
        if self._shuffle:
            self._shuffle

        self._g1.on_epoch_end()
        self._g2.on_epoch_end()

    def _shuffle_samples(self):
        self._df = self._df.sample(frac=1).reset_index(drop=True)
        self._g1._df = self._df
        self._g2._df = self._df
