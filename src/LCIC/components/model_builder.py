import tensorflow as tf
from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml

from LCIC.config.base_model_configuration import BaseModelConfigurationManager
from LCIC.entity.base_model_entity import BaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: BaseModelConfig):
        self.model = None
        self.config = config

    # Note: this model is called at the pipeline file to get the base model.
    def get_base_model(self):
        """
        This function is reposible for loading the pretrained model, which is specified in the config file.
        Also set the class model variable.
        """
        if self.config.pretrained_model == "VGG16":
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.config.pretrained_model == "EFFICIENTNETB2":
            self.model = tf.keras.applications.efficientnet.EfficientNetB2(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

        elif self.config.pretrained_model == "EFFICIENTNETB6":
            self.model = tf.keras.applications.efficientnet.EfficientNetB2(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

        elif self.config.pretrained_model == "INCEPTIONV3":
            self.model = tf.keras.applications.inception_v3.InceptionV3(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.config.pretrained_model == "RESNET101":
            self.model = tf.keras.applications.resnet.ResNet101(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )
        elif self.config.pretrained_model == "DENSENET169":
            self.model = tf.keras.applications.densenet.DenseNet169(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top
            )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        This is a static function that's responsible for saving the model at the specified path.
        path: Path where you want to save the model.
        model: Model which you want to save. object of tensorflow.keras.Model
        """
        model.save(path)

    @staticmethod
    def __prepare_base_model(model, classes, freeze_all, freeze_till, optimizer, learning_rate, dropout_rate):
        """
        This function is responsible for preparing the base model for training, with custome layers.
        model: Model which you want to prepare for training.
        classes: Number of classes in the dataset.
        freeze_all: If you want to freeze all the layers of the model then set it to True.
        freeze_till: If you want to freeze the layers till a certain layer then set it to the layer number.
        optimizer: Optimizer function for training.
        learning_rate: Learning rate for the optimizer.
        dropout_rate: Dropout rate for the custome layers.

        """

        # Freeze the pretrained weights
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False

            no_of_layers = len(model.layers)
            no_of_layers_to_train = int(no_of_layers // 2)

            # Skipping the BN layers
            for layer in model.layers[-no_of_layers_to_train:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model[:freeze_till]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
                else:
                    layer.trainable = False
        logger.info(f"Creating custome layers for the model")
        # Rebuild top
        x = tf.keras.layers.GlobalAveragePooling2D(
            name="avg_pool")(model.output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
        x = tf.keras.layers.Dense(units=128, activation='relu')(x)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation='softmax',
            name="output_layer"
        )(x)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        logger.info("Compileing the model")

        if optimizer.lower() == 'adam':
            full_model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
        elif optimizer.lower() == 'sgd':
            full_model.compile(
                optimizer=tf.keras.optimizers.SGD(
                    learning_rate=learning_rate, momentum=0.9),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
        elif optimizer.lower() == 'rmsprop':
            full_model.compile(
                optimizer=tf.keras.optimizers.RMSprop(
                    learning_rate=learning_rate),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )
        else:
            raise ValueError("Undefine optimizer function")

        full_model.summary()
        logger.info(f"Loaded the base model.")
        return full_model

    def update_base_model(self):
        self.full_model = self.__prepare_base_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            optimizer=self.config.params_optimizer,
            learning_rate=self.config.params_learning_rate,
            dropout_rate=self.config.params_dropout_rate
        )
        self.save_model(path=self.config.updated_base_model_path,
                        model=self.full_model)

        logger.info(
            f"Final model with custome layers is saving at {self.config.updated_base_model_path}")
