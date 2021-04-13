from tensorflow import keras
import efficientnet.tfkeras as efn


class Model:
    def __init__(self, n_classes, final_activation="softmax", augmentation=1):

        self.n_classes = n_classes
        self.augmentation = augmentation
        self.final_activation = final_activation
        self.base_model = None  # type: keras.Model
        self.input_shape = None

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
        pass

    def top_stage(self, input_tensor):
        x = input_tensor

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
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


class Model_ENB0(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = (224, 224, 3)

    def base_stage(self, input_tensor):

        self.base_model = keras.applications.EfficientNetB0(
            input_tensor=input_tensor,
            include_top=False,
            weights="imagenet",
            classes=self.n_classes,
        )

        return self.base_model.output


class Model_ENB7(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = (600, 600, 3)

    def base_stage(self, input_tensor):

        self.base_model = keras.applications.EfficientNetB7(
            input_tensor=input_tensor,
            include_top=False,
            weights="imagenet",
            classes=self.n_classes,
        )

        return self.base_model.output


class Model_ENB7_NS(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = (600, 600, 3)

    def base_stage(self, input_tensor):

        self.base_model = efn.EfficientNetB7(
            input_tensor=input_tensor,
            include_top=False,
            weights="noisy-student",
            classes=self.n_classes,
        )

        return self.base_model.output


class Model_ENB7_X2(Model):
    """
    EfficientNetB7 with 2 inputs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = (600, 600, 3)

    def create(self):

        base = keras.applications.EfficientNetB7(
            include_top=False,
            weights="imagenet",
            classes=self.n_classes,
        )

        input_1 = keras.layers.Input(shape=self.input_shape)
        x1 = self.augmentation_stage(input_1)
        x1 = base(x1)
        x1 = keras.layers.GlobalAveragePooling2D()(x1)
        x1 = keras.layers.BatchNormalization()(x1)

        input_2 = keras.layers.Input(shape=self.input_shape)
        x2 = self.augmentation_stage(input_2)
        x2 = base(x2)
        x2 = keras.layers.GlobalAveragePooling2D()(x2)
        x2 = keras.layers.BatchNormalization()(x2)

        x = keras.layers.Concatenate()([x1, x2])
        x = keras.layers.Dropout(0.2)(x)
        output_1 = keras.layers.Dense(
            self.n_classes,
            activation=self.final_activation,
        )(x)

        self.model = keras.Model(inputs=[input_1, input_2], outputs=[output_1])

        return self.model
