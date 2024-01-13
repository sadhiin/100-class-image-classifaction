import tensorflow as tf
from LCIC.constants import *
from LCIC.config.model_eval_configuration import EvalConfigurationManager
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class EvaluationDataPreprocessing():
    def __init__(self,
                config: EvalConfigurationManager,
                params_path: Path = PARAMS_FILE_PATH):
        self.config = config
        self.params = read_yaml(params_path)
        # setting the params for to have the target size and classes of the model
        self.params = self.params[self.config.model_name]

        self.test_dataset = None

    def __generator(self):
        _test_datagen = ImageDataGenerator(rescale=1./255)
        return _test_datagen

    def get_test_data_set(self):
        test_generator = self.__generator()

        test_set = test_gen.flow_from_directory(
            directory=Path.joinpath(self.config.dataset_path, 'test'),
            target_size=self.params.IMAGE_SIZE[:-1],
            color_mode='rgb',
            class_mode='categorical',
            batch_size=self.config.batch_size,
        )

        self.test_dataset = test_set 
        return self.test_dataset


class ModelEvaluation:
    def __init__(self, config: EvalConfigurationManager):
        self.config = config
        self.test_data = EvaluationDataPreprocessing(config=self.config).get_test_data_set()
        self.model = None
        self.score = None

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def save_score(self):
        scores = {'loss': self.score[0], 'accuracy': self.score[1]}
        save_json(path=Path('reports/score.json'), data= scores)
    
    def evaluation(self):
        self.model = self.load_model(self.config.model_path)
        self.score = model.evaluate(self.test_data, batch_size=self.config.batch_size, verbose=1)
        self.save_score()

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            print(self.config.all_params)
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'loss': self.score[0], 'accuracy':self.score[1]})

            # model registry
            if tracking_url_type_store != 'file':
                # register the model
                # There are other ways to use the model registry, which depends on the user case,
                # please refer to the doc for more information:
                # at https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, 'model', registered_model_name=str(self.config.model_name))
            else:
                mlflow.keras.log_model(self.model, "model")