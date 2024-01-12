import tensorflow as tf
from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml, save_json
from LCIC.entity.model_training import ModelTrainingConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPreprocessing():
    def __init__(self,
                 config: ModelTrainingConfig,
                 params_path: Path = PARAMS_FILE_PATH):
        self.config = config
        self.params = read_yaml(params_path)
        # setting the params for to have the target size and classes of the model
        self.params = self.params[self.config.model_name]

        self.train_generator = None
        self.valid_generator = None

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
            directory=self.config.dataset_path,
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
            directory=self.config.dataset_path,
            target_size=self.params.IMAGE_SIZE,
            color_mode='rgb',
            classes=self.params.CLASSES,
            class_mode='categorical',
            batch_size=self.config.batch_size,
            interpolation='nearest',
            subset="validation"
        )

        self.train_generator = training_set
        self.valid_generator = validation_set

        return self.train_generator, self.valid_generator


class Training():
    def __init__(self, config: ModelTrainingConfig,
                 params_path: Path = PARAMS_FILE_PATH) -> None:
        self.config = config
        self.params = read_yaml(params_path)

        self.training_data, self.validation_data = DataPreprocessing(
            config=self.config).get_train_and_valid_set()

        self.trains_steps = self.training_data.samples // self.config.batch_size
        self.validation_steps = self.validation_data.samples // self.config.batch_size
        self.model = tf.keras.load_model(self.config.model_path)
        self.history = None

    @staticmethod
    def save_model(model: tf.keras.Model, path: Path):
        model.save(path)

    def __getoptimizer(self, optimizer_name: str):
        if optimizer_name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=self.params.LEARNING_RATE, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif optimizer_name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=self.params.LEARNING_RATE, rho=0.9)
        elif optimizer_name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=self.params.LEARNING_RATE, momentum=0.0, nesterov=False)

    def train(self, callbacks_list: list = [], save_model: bool = True, gethistory: bool = True):

        self.model.compile(optimizer=self.__getoptimizer(self.params.OPTIMIZER),
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
        classvalues = {value: key for key,
                       value in self.training_data.class_indices.items()}

        save_json(path=CLASS_INDEX_PATH, data=classvalues)
        self.history = self.model.fit(self.training_data,
                                      steps_per_epoch=self.trains_steps,
                                      epochs=self.params.EPOCHS,
                                      validation_data=self.validation_data,
                                      validation_steps=self.validation_steps,
                                      callbacks=callbacks_list)
        if save_model:
            self.save_model(self.model, self.config.trained_model_path)
        if gethistory:
            return self.history
