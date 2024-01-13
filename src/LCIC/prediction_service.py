import os
import numpy as np
from LCIC import logger
from LCIC.constants import *
from LCIC.utils.common import read_yaml, load_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionService:

    def __init__(self, filename, config_path: Path = CONFIG_FILE_PATH, params_path: Path = PARAMS_FILE_PATH) -> None:
        self.config = read_yaml(config_path)
        self.config = self.config.model_training

        self.params = read_yaml(params_path)
        self.params = self.params[self.config.model_name]

        self.model = load_model(self.config.trained_model_path)

        self.filename = filename

    def predict(self):
        img = self.filename
        test_image = image.load_img(img, target_size=self.params.IMAGE_SIZE[:-1])

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        result = self.model.predict(test_image)
        result = np.argmax(result, axis=1)
        
        class_idx = load_json(CLASS_INDEX_PATH)
        prediction = class_idx[str(result[0])]

        return [{"image": prediction}]