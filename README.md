# Fraud Detection in Credit Card Transactions



## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Credit card fraud is a significant issue for both financial institutions and customers. This project aims to leverage machine learning algorithms to detect fraudulent transactions efficiently. The models are trained on a dataset of credit card transactions, which includes various features such as transaction amount, time, and other anonymized parameters.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/path/creditcardfrauddetection.git
   cd creditcardfrauddetection


**Usage**
To run the project, follow these steps:

1. **Prepare the dataset:**

Place your dataset (e.g., creditcard.csv) in a disired directory and locate it through main.py

2. **Configure the settings:**

Modify the classifiers_config.json file to set your preferred classifiers and hyperparameters.

3. **Run the main script:**
python main.py

4. **View the results:**

The progress results will be displayed on console, while performance results including evaluation metrics, will be saved in the results.log file.
## Project Structure

The project is organized into several key directories and files, as described below:

- **Dataset**: The dataset can be obtained from kaggle through the this link https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.


- **utils/logger.py**: Contains functions for logging output to the console and files. This module plays a crucial role in monitoring the execution of the scripts.

- **utils/plotter.py**: Contains functions for plotting graphs and visualizations, which are essential for understanding the distribution of data and the performance of models.

- **preprocessing/preprocess.py**: Contains functions for data loading, cleaning, and preprocessing, including handling class imbalances through resampling techniques.

- **classifiers/classifier_init.py**: Responsible for initializing the classifiers based on the configurations provided in the `classifiers_config.json` file.

- **classifiers/cv.py**: Manages cross-validation processes to evaluate model performance.

- **classifiers/ensemble.py**: Handles the implementation of ensemble methods, combining multiple classifiers to improve prediction accuracy.

- **classifiers/hypertuning.py**: Contains functions for hyperparameter tuning, optimizing model parameters to achieve the best possible performance.

- **classifiers/train.py**: The main training module where models are trained, validated, and tested.

- **classifiers/utils.py**: Utility functions used across different parts of the project, including data splitting, metric calculations, and more.

- **nn_model/model.py**: Contains the implementation of a neural network model specifically designed for this project.

- **main.py**: The main script that ties everything together, executing the entire workflow from data preprocessing to model evaluation.

- **configs/config.json**: Configuration file where you can specify the classifiers, their parameters, cross-validation settings, ensemble methods, and resampling methods.








