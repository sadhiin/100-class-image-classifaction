import os
import time
import tensorflow as tf
from LCIC import logger
from livelossplot import PlotLossesKerasTF


def get_log_path(DIR="Tensorboard/logs/"):
    log_file_name = time.strftime("TB_log_%Y_%m_%d-%H_%M_%S")
    os.makedirs(DIR, exist_ok=True)
    log_path = os.path.join(DIR, log_file_name)
    logger.info(f"Tensorboard log path: {log_path}")
    print(f"Tensorboard log path: {log_path}")
    return log_path


def get_callbacks():
    log_path = get_log_path()

    file_name = os.path.join("artifacts/training/checkpoints", "model_ckpt.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=file_name, save_best_only=True, monitor='val_loss', mode='min'),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True),

        tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1),
        PlotLossesKerasTF()
    ]
    return callbacks


def get_val_callbacks():
    callbacks = [
        PlotLossesKerasTF()
    ]
    return callbacks
