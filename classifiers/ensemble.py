from sklearn.ensemble import VotingClassifier
import pandas as pd
from typing import Dict, Any
from utils.logger import setup_logger
from classifiers.utils import find_best_threshold, adjusted_prediction, calculate_metrics
import numpy as np

# Initialize logger
logger = setup_logger(__name__)

def get_voting_classifier(config: Dict[str, Any], best_estimators: Dict[str, Any]) -> VotingClassifier:
    """
    Create a VotingClassifier based on configuration and best estimators.

    Parameters:
    - config (Dict[str, Any]): Configuration dictionary.
    - best_estimators (Dict[str, Any]): Dictionary of best estimators.

    Returns:
    - VotingClassifier: Configured VotingClassifier instance.
    """
    estimators = [(name, best_estimators[name]) for name in config['classifiers'] if name in best_estimators]

    if not estimators:
        raise ValueError("No valid classifiers found in best_estimators for voting.")
    
    # Voting type
    voting_type = config.get('voting', 'soft').lower()
    if voting_type not in ['soft', 'hard']:
        raise ValueError(f"Invalid voting type '{voting_type}'. Choose 'soft' or 'hard'.")

    return VotingClassifier(estimators=estimators, voting=voting_type)

def summarize_array(array: np.ndarray) -> str:
    """
    Provide a summary of a numpy array.

    Parameters:
    - array (np.ndarray): Array to summarize.

    Returns:
    - str: Summary of the array.
    """
    if array.ndim == 1:
        return (f"Min: {np.min(array):.4f}, "
                f"Max: {np.max(array):.4f}, "
                f"Mean: {np.mean(array):.4f}")
    else:
        return f'Array with shape {array.shape}'

def train_and_evaluate_voting_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_estimators: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train and evaluate a VotingClassifier.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Test features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    - best_estimators (Dict[str, Any]): Dictionary of best estimators.
    - config (Dict[str, Any]): Configuration dictionary.

    Returns:
    - Dict[str, Any]: Dictionary containing evaluation metrics.
    """
    voting_results = {}

    try:
        # Retrieve the VotingClassifier based on configuration
        voting_clf = get_voting_classifier(config, best_estimators)
        logger.info("Training Voting Classifier...")
        voting_clf.fit(X_train, y_train)

        logger.info("Evaluating Voting Classifier...")
        y_pred_voting = voting_clf.predict(X_test)

        # Handle probability estimates
        if hasattr(voting_clf, "predict_proba"):
            y_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
        else:
            logger.warning("Voting classifier doesn't support predict_proba. Using predictions as probability estimates.")
            y_proba_voting = y_pred_voting  # Fallback - might not be ideal

        # Optimize threshold and make adjusted predictions
        best_threshold = find_best_threshold(y_test, y_proba_voting)
        y_pred_adj = adjusted_prediction(voting_clf, X_test, best_threshold)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred_adj, y_proba_voting)
        voting_results["VotingClassifier"] = metrics

        # Log probability statistics
        if 'y_proba_voting' in locals():
            logger.info(f"Probability Estimates: {summarize_array(y_proba_voting)}")

    except ValueError as ve:
        logger.error(f"Error during voting classifier creation or training: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected error during voting classifier evaluation: {e}")

    return voting_results
