# Fraud Detection in Credit Card Transactions

This project focuses on detecting fraudulent transactions in credit card data using various machine learning techniques. The primary goal is to build models that can accurately identify anomalies and flag potentially fraudulent activities while reducing false positives and improving precision.

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

2. **Create and activate a virtual environment:**
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

2. **Install the required dependencies:**
pip install -r requirements.txt

**Usage**
To run the project, follow these steps:

1. **Prepare the dataset:**

Place your dataset (e.g., creditcard.csv) in the Dataset-Approved directory.

2. **Configure the settings:**

Modify the classifiers_config.json file to set your preferred classifiers and hyperparameters.

3. **Run the main script:**
python main.py

4. **View the results:**

The results, including logs and evaluation metrics, will be saved in the results.log file and displayed in the console.

**Project Structure:**
1. Description of Key Files and Directories:
2. Dataset-Approved/: Contains the dataset file.
3. utils/logger.py: Contains functions for logging output.
4. utils/plotter.py: Contains functions for plotting graphs and visualizations.
5. preprocessing/preprocess.py: Contains functions for data loading and preprocessing.
6. models/classifiers.py: Contains functions for training and evaluating classifiers.
7. models/ensemble.py: Contains functions for training and evaluating ensemble classifiers.
8. main.py: The main script to run the project.
9. classifiers_config.json: Configuration file for specifying classifiers , their parameters , crossvalidation and ensemble method.




