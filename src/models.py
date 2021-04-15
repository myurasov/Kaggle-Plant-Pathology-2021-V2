import efficientnet.tfkeras as efn
from tensorflow import keras

from src.config import c


class Model_ENBX:
    def __init__(
        self,
        variant,
        n_classes,
        final_activation="softmax",
        augmentation=1,
        input_shape=None,
    ):

        if input_shape is None:
            en_input_sizes = [224, 240, 260, 300, 380, 456, 528, 600]
            self.input_shape = (en_input_sizes[variant], en_input_sizes[variant], 3)
        else:
            self.input_shape = input_shape

        self._variant = variant
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.final_activation = final_activation
        self.base_model = None  # type: keras.Model

    def freeze_base(self):
        self.base_model.trainable = False

    def unfreeze_base(self):
        for layer in self.base_model.layers:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = True

    def augmentation_stage(self, input_tensor):
        x = input_tensor

        # augmentation level
        if self.augmentation >= 1:
            x = keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"
            )(x)
            x = keras.layers.experimental.preprocessing.RandomRotation(
                0.5, fill_mode="reflect"
            )(x)
            x = keras.layers.experimental.preprocessing.RandomZoom(
                height_factor=(0.5, -0.5), width_factor=(0.5, -0.5), fill_mode="reflect"
            )(x)
            x = keras.layers.experimental.preprocessing.RandomContrast(factor=0.33)(x)
            x = keras.layers.experimental.preprocessing.RandomTranslation(
                height_factor=0.5, width_factor=0.5, fill_mode="reflect"
            )(x)

        return x

    def base_stage(self, input_tensor):

        self.base_model = getattr(keras.applications, f"EfficientNetB{self._variant}")(
            input_tensor=input_tensor,
            include_top=False,
            weights="imagenet",
            classes=self.n_classes,
        )

        return self.base_model.output

    def top_stage(self, input_tensor):
        x = input_tensor

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2, seed=c["SEED"])(x)
        x = keras.layers.Dense(
            self.n_classes,
            activation=self.final_activation,
        )(x)

        return x

    def create(self):

        x = input_1 = keras.layers.Input(shape=self.input_shape)
        x = self.augmentation_stage(x)
        x = self.base_stage(x)
        x = output_1 = self.top_stage(x)

        self.model = keras.Model(inputs=[input_1], outputs=[output_1])
        return self.model


class Model_ENBX_NS(Model_ENBX):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def base_stage(self, input_tensor):

        self.base_model = getattr(efn, f"EfficientNetB{self._variant}")(
            input_tensor=input_tensor,
            include_top=False,
            weights="noisy-student",
            classes=self.n_classes,
        )

        return self.base_model.output


class Model_ENBL2(Model_ENBX):
    def __init__(self, **kwargs):
        super().__init__(input_shape=(800, 800, 3), variant=None, **kwargs)

    def base_stage(self, input_tensor):

        # @see https://github.com/xhlulu/keras-noisy-student

        weights_file = keras.utils.get_file(
            "efficientnet-l2_noisy-student_notop.h5",
            "https://github.com/xhlulu/keras-efficientnet-l2/releases/"
            + "download/data/efficientnet-l2_noisy-student_notop.h5",
            cache_subdir="models",
        )

        self.base_model = efn.EfficientNetL2(
            input_tensor=input_tensor,
            include_top=False,
            classes=self.n_classes,
            weights=weights_file,
            # drop_connect_rate=0,
        )

        return self.base_model.output


class Model_ENBX_X2(Model_ENBX):
    """
    EfficientNetBX with 2 inputs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create(self):

        self.base_model = getattr(keras.applications, f"EfficientNetB{self._variant}")(
            include_top=False,
            weights="imagenet",
            classes=self.n_classes,
        )

        input_1 = keras.layers.Input(shape=self.input_shape)
        x1 = self.augmentation_stage(input_1)
        x1 = self.base_model(x1)
        x1 = keras.layers.GlobalAveragePooling2D()(x1)
        x1 = keras.layers.BatchNormalization()(x1)

        input_2 = keras.layers.Input(shape=self.input_shape)
        x2 = self.augmentation_stage(input_2)
        x2 = self.base_model(x2)
        x2 = keras.layers.GlobalAveragePooling2D()(x2)
        x2 = keras.layers.BatchNormalization()(x2)

        x = keras.layers.Concatenate()([x1, x2])
        x = keras.layers.Dropout(0.2, seed=c["SEED"])(x)
        output_1 = keras.layers.Dense(
            self.n_classes,
            activation=self.final_activation,
        )(x)

        self.model = keras.Model(inputs=[input_1, input_2], outputs=[output_1])

        return self.model
