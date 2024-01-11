import tensorflow as tf
from LCIC import logger
from LCIC.utils.common import read_yaml
from LCIC.constants import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from LCIC.entity.data_prep_entity import PreprocessConfig


class DataPreprocessing():
    def __init__(self, config: PreprocessConfig, params_path: Path = PARAMS_FILE_PATH):
        self.config = config
        self.params = read_yaml(params_path)

    def __generator(self):
        _train_datagen = ImageDataGenerator(
            rescale=self.config.rescale,
            shear_range=self.config.shear_range,
            zoom_range=self.config.zoom_range,
            width_shift_range=self.config.width_shift_range,
            height_shift_range=self.config.height_shift_range,
            horizontal_flip=self.config.horizontal_flip,
            validation_split=self.config.validation_split,
            fill_mode=self.config.fill_mode,
            batch_size=self.config.batch_size
        )

        _test_datagen = ImageDataGenerator(rescale=1./255)

        return _train_datagen, _test_datagen

    def get_train_and_valid_set(self):
        train_datagen, val_datagen = self.__generator()

        training_set = train_datagen.flow_from_directory(
            directory=self.config.path,
            target_size=self.params.IMAGE_SIZE,
            color_mode='rgb',
            classes=self.params.CLASSES,
            class_mode='categorical',
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.config.seed,
            interpolation='nearest',
            subset="training"
        )

        validation_set = val_datagen.flow_from_directory(
            directory=self.config.path,
            target_size=self.params.IMAGE_SIZE,
            color_mode='rgb',
            classes=self.params.CLASSES,
            class_mode='categorical',
            batch_size=self.config.batch_size,
            interpolation='nearest',
            subset="validation"
        )

        return training_set, validation_set
