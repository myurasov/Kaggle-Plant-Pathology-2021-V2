import os

from tensorflow import keras


def lr_logger(self, logs: dict):
    """ Sample logger - adds LR """
    logs.update({"lr": keras.backend.eval(self.model.optimizer.lr)})


def gpu_temp_logger(self, logs: dict):
    stream = os.popen("nvidia-smi stats -i 0 -c 1 | grep temp")
    output = stream.read()
    gpu_temp = int(output[-4:])
    logs.update({"gpu_temp": gpu_temp})


class TensorBoard_Logger(keras.callbacks.TensorBoard):
    def __init__(self, loggers=[lr_logger], **kwargs):
        self._loggers = loggers
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for logger in self._loggers:
            logger(self, logs)
        super().on_epoch_end(epoch, logs)
