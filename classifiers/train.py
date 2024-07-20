from classifiers.utils import find_best_threshold, adjusted_prediction, calculate_metrics, evaluate_random_states
from utils.logger import setup_logger
from classifiers.classifier_init import all_classifiers
import time

logger = setup_logger(__name__)

def train_classifier(clf, X_train, y_train, name):
    """
    Train the classifier.

    Parameters:
    clf: Classifier object.
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.
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
    X_test (pd.DataFrame): Test features.
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
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Test features.
    y_train (pd.Series): Training labels.
    y_test (pd.Series): Test labels.
    best_estimators (dict): Dictionary of best estimators.
    config (dict): Configuration dictionary.

    Returns:
    dict: Results dictionary with metrics for each classifier.
    """
    results = {}

    for name in config['classifiers']:
        logger.info(f"Starting training for {name}...")

        try:
            # Get the classifier or pipeline
            clf = best_estimators.get(name)

            if clf is None:
                logger.warning(f"No tuned model found for {name}. Using default classifier.")
                clf = all_classifiers.get(name)
                if clf is None:
                    raise ValueError(f"Classifier {name} is not available in all_classifiers.")

            # Train the classifier
            train_classifier(clf, X_train, y_train, name)

            # Get predicted probabilities or decision function
            y_proba = get_predictions(clf, X_test, name)

            # Find best threshold and make predictions
            best_threshold = find_best_threshold(y_test, y_proba)
            y_pred_adj = adjusted_prediction(clf, X_test, best_threshold)

            # Calculate metrics
            metrics = calculate_metrics(y_test, y_pred_adj, y_proba)
            results[name] = metrics

            logger.info(f"Finished training and evaluation for {name}.")

        except Exception as e:
            logger.error(f"An error occurred while training {name}: {e}", exc_info=True)

    return results
