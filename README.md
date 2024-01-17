# Image Classification Project with MLOps Practices

## Overview

100 Class Image Classification Project with MLOps Practices! This project is designed to demonstrate a robust and scalable approach to building and deploying a deep learning model for image classification. With [100 different classes of sports images](https://www.kaggle.com/datasets/gpiosenka/sports-classification), this project showcases the versatility and complexity of handling a large-scale image classification task.

## Project Structure

```
├── 3.jpg
├── 5.jpg
├── README.md
├── app.py
├── artifacts
|   ├── data
|   |   ├── sports-classification
|   |   |   ├── test
|   |   |   ├── train
│   │   |   └── valid
│   ├── base_model
│   │   ├── base_model.keras
│   │   └── pretrained_model.keras
│   └── training
│       └── checkpoints
│           └── model_ckpt.h5
├── config
│   ├── class_index.json
│   └── config.yaml
├── dvc.yaml
├── input_image.jpg
├── notebooks
│   ├── 01_data_trail.ipynb
│   ├── 02_base_model.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_eval.ipynb
├── params.yaml
├── reports
│   └── score.json
├── requirements.txt
├── setup.py
├── src
│   ├── LCIC
│   │   ├── __init__.py
│   │   ├── components
│   │   │   ├── __init__.py
│   │   │   ├── data_ingestion.py
│   │   │   ├── model_builder.py
│   │   │   ├── model_evaluation.py
│   │   │   └── model_training.py
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── base_model_configuration.py
│   │   │   ├── configuration.py
│   │   │   ├── model_eval_configuration.py
│   │   │   └── model_training_configuration.py
│   │   ├── constants
│   │   │   ├── __init__.py
│   │   ├── entity
│   │   │   ├── __init__.py
│   │   │   ├── base_model_entity.py
│   │   │   ├── data_config_entity.py
│   │   │   ├── model_training_entity.py
│   │   │   └── model_val_entity.py
│   │   ├── main.py
│   │   ├── pipeline
│   │   │   ├── __init__.py
│   │   │   ├── stage_01_data_ingestion.py
│   │   │   ├── stage_02_base_model_building.py
│   │   │   ├── stage_03_model_training.py
│   │   │   └── stage_04_model_eval.py
│   │   ├── prediction_service.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── common.py
│   │       └── keras_callbacks.py
├── static
│   └── prediction.css
├── template.py
└── templates
    └── index.html
```

- **data:** Contains the [datasets](https://www.kaggle.com/datasets/gpiosenka/sports-classification) into training, validation, and test sets.
- **models:** Consists of pretrain model and custome architecture defined in [`model_builder.py`](https://github.com/sadhiin/100-class-image-classifiaction/blob/main/src/LCIC/components/model_builder.py) and the training script [`model_training.py`](https://github.com/sadhiin/100-class-image-classifiaction/blob/main/src/LCIC/components/model_training.py).
- **notebooks:** Jupyter [notebooks](https://github.com/sadhiin/100-class-image-classifiaction/tree/main/notebooks) for data exploration and model evaluation.
- **src:** All the coding implementation and helper scripts for the project.
- **requirements.txt:** Python dependencies required for the project.
- **config.yaml:** Configuration other settings for the project.
- **params.yaml:** Configuration of the model with hyperparameters and other settings.
- **main.py:** Main script to run all the stages of the project.
- **app.py:** For run the flask app.

## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/sadhiin/100-class-image-classifiaction.git
   cd 100-class-image-classifaction
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and organize your dataset into the `data` directory, following the structure mentioned above.

4. Ensure to have the model related parameters in `params.yaml` file.

5. Configure your info (like dataset selected pretrain model) and settings in `config/config.yaml`.

## Training the Model

To train the model, run the following command:

```bash
python src/LCIC/main.py
```
Use dvc for tracking the stages
```bash
dvc repro
```

This will use the configurations specified in `config/config.yaml` and save the trained model in the `artifacts/trained_model` directory.

## Model Evaluation

Automatic model evaluatoin will done on test set by running the `src/LCIC/main.py` or `dvc repro` command.

Fourther if you want's to explore the code can be found at 
Explore model performance by running the Jupyter notebooks in the `notebooks/04_model_eval.ipynb` directory:

```bash
jupyter notebook notebooks/
```
This ensures consistent and reproducible deployments across different environments.

## Continuous Integration/Continuous Deployment (CI/CD)

This project supports CI/CD pipelines to automate testing and deployment processes. Integrate with your preferred CI/CD service for seamless automation.

## Contributing

Feel free to contribute by opening issues or submitting pull requests. We appreciate your feedback and collaboration!

Happy coding! 🚀