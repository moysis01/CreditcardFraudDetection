# cnn hypertuning path  
DIRECTORY_PATH = "C:\\Users\\ke1no\\Downloads\\tuning"
PROJECT_NAME = "creditcard_fraud"

from classifiers.utils import find_best_threshold, adjusted_prediction, calculate_metrics
from nn_model.model import DNNClassifier, load_best_hyperparameters, tune_hyperparameters
from utils.logger import setup_logger
from classifiers.classifier_init import all_classifiers
import time
import pandas as pd
import numpy as np
from utils.plotter import plot_training_history 

logger = setup_logger(__name__)

def train_classifier(clf, X_train, y_train, name):
    """
    Train the classifier.

    Parameters:
    clf: Classifier object.
    X_train (pd.DataFrame or np.ndarray): Training features.
    y_train (pd.Series or np.ndarray): Training labels.
    name (str): Classifier name.

    Returns:
    float: Training duration in seconds.
    """
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training time for {name}: {training_time:.2f} seconds")
    return training_time

def get_predictions(clf, X_test, name):
    """
    Get predicted probabilities or decision function for the classifier.

    Parameters:
    clf: Classifier object.
    X_test (pd.DataFrame or np.ndarray): Test features.
    name (str): Classifier name.

    Returns:
    np.ndarray: Predicted probabilities or decision function values.
    """
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except AttributeError:
        logger.warning(f"{name} does not support predict_proba. Using decision function instead.")
        try:
            y_proba = clf.decision_function(X_test)
        except AttributeError:
            raise AttributeError(f"{name} does not have predict_proba or decision_function.")
    return y_proba

def training(X_train, X_test, y_train, y_test, best_estimators, config):
    """
    Train and evaluate classifiers.

    Parameters:
    X_train (pd.DataFrame or np.ndarray): Training features.
    X_test (pd.DataFrame or np.ndarray): Test features.
    y_train (pd.Series or np.ndarray): Training labels.
    y_test (pd.Series or np.ndarray): Test labels.
    best_estimators (dict): Dictionary of best estimators.
    config (dict): Configuration dictionary.

    Returns:
    dict: Results dictionary with metrics for each classifier.
    """
    results = {}

    for name in config['classifiers']:
        logger.info(f"Starting training for {name}...")
        
        if name == 'Neural Network':
            # Converting DataFrame to NumPy array and reshape for neural network
            X_train_nn = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1)) if isinstance(X_train, pd.DataFrame) else X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test_nn = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1)) if isinstance(X_test, pd.DataFrame) else X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Tuning the hyperparameters if not already tuned
            best_hps = tune_hyperparameters(DIRECTORY_PATH, PROJECT_NAME, X_train_nn, y_train)

            # Initialize DNNClassifier with the best hyperparameters
            dnn_clf = DNNClassifier(input_shape=(29, 1), best_hps=best_hps)
            dnn_clf.fit(X_train_nn, y_train)
            
            # Plotting training history
            if hasattr(dnn_clf.model, 'history'):
                plot_training_history(dnn_clf.model.history, save_dir='plots')

            # Predict and adjust predictions based on the best threshold
            y_proba = dnn_clf.predict_proba(X_test_nn)[:, 1]
            best_threshold = find_best_threshold(y_test, y_proba)
            y_pred_adj = (y_proba >= best_threshold).astype(int)
            metrics = calculate_metrics(y_test, y_pred_adj, y_proba)
            results[name] = metrics

        else:
            try:
                clf = best_estimators.get(name)

                if clf is None:
                    logger.warning(f"No tuned model found for {name}. Using default classifier.")
                    clf = all_classifiers.get(name)
                    if clf is None:
                        raise ValueError(f"Classifier {name} is not available in all_classifiers.")

                #  X_train and X_test being 2D for non-neural network classifiers
                if len(X_train.shape) > 2:
                    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
                    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
                else:
                    X_train_reshaped = X_train
                    X_test_reshaped = X_test

                train_classifier(clf, X_train_reshaped, y_train, name)
                y_proba = get_predictions(clf, X_test_reshaped, name)
                best_threshold = find_best_threshold(y_test, y_proba)
                y_pred_adj = adjusted_prediction(clf, X_test_reshaped, best_threshold)
                metrics = calculate_metrics(y_test, y_pred_adj, y_proba)
                results[name] = metrics

                logger.info(f"Finished training and evaluation for {name}.")

            except Exception as e:
                logger.error(f"An error occurred while training {name}: {e}", exc_info=True)

    return results
