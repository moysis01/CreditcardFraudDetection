from sklearn.ensemble import VotingClassifier
import pandas as pd
from typing import Dict, Any
from sklearn.pipeline import FunctionTransformer, Pipeline
from classifiers.utils import find_best_threshold, adjusted_prediction, calculate_metrics
import numpy as np
from nn_model import DNNClassifier
from utils.logger import setup_logger

logger = setup_logger(__name__)

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

def get_voting_classifier(config: Dict[str, Any], best_estimators: Dict[str, Any]) -> VotingClassifier:
    estimators = []
    for name in config['classifiers']:
        if name == 'Neural Network':
            dnn_model = DNNClassifier(input_shape=(29, 1), epochs=200, batch_size=2048, verbose=1)
            reshape_transformer = FunctionTransformer(lambda X: np.expand_dims(X, axis=2), validate=False)
            dnn_pipeline = Pipeline([
                ('reshape', reshape_transformer),
                ('dnn', dnn_model)
            ])
            estimators.append(('Neural Network', dnn_pipeline))
        elif name in best_estimators:
            estimators.append((name, best_estimators[name]))
    
    if not estimators:
        raise ValueError("No valid classifiers found in best_estimators for voting.")
    
    voting_type = config.get('voting', 'soft').lower()
    if voting_type not in ['soft', 'hard']:
        raise ValueError(f"Invalid voting type '{voting_type}'. Choose 'soft' or 'hard'.")
    
    return VotingClassifier(estimators=estimators, voting=voting_type)

def train_and_evaluate_voting_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_estimators: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    voting_results = {}
    
    try:
        voting_clf = get_voting_classifier(config, best_estimators)
        logger.info("Training Voting Classifier...")
        voting_clf.fit(X_train, y_train)
        
        logger.info("Evaluating Voting Classifier...")
        y_pred_voting = voting_clf.predict(X_test)
        
        if hasattr(voting_clf, "predict_proba"):
            y_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
        else:
            logger.warning("Voting classifier doesn't support predict_proba. Using predictions as probability estimates.")
            y_proba_voting = y_pred_voting
        
        best_threshold = find_best_threshold(y_test, y_proba_voting)
        y_pred_adj = adjusted_prediction(voting_clf, X_test, best_threshold)
        
        metrics = calculate_metrics(y_test, y_pred_adj, y_proba_voting)
        voting_results["VotingClassifier"] = metrics
        
        logger.info(f"Probability Estimates: {summarize_array(y_proba_voting)}")

    except ValueError as ve:
        logger.error(f"Error during voting classifier creation or training: {ve}")
    except Exception as e:
        logger.exception(f"Unexpected error during voting classifier evaluation: {e}")
    
    return voting_results
