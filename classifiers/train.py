from classifiers.utils import find_best_threshold, adjusted_prediction, calculate_metrics,evaluate_random_states
from utils.logger import setup_logger
from classifiers.classifier_init import all_classifiers
import time

logger = setup_logger(__name__)

def training(X_train, X_test, y_train, y_test, best_estimators, config):
    results = {}

    for name in config['classifiers']:
        logger.info(f"Starting training for {name}...")
        
        try:
            # Get the classifier or pipeline
            clf = best_estimators.get(name)  # Directly access the best estimator if available

            if clf is None:  # Check if best estimator exists
                logger.warning(f"No tuned model found for {name}. Using default classifier.")
                clf = all_classifiers.get(name)
                if clf is None:
                    raise ValueError(f"Classifier {name} is not available in all_classifiers.")

            # Start training
            start_time = time.time()
            clf.fit(X_train, y_train)  
            end_time = time.time()
            logger.info(f"Training time for {name}: {end_time - start_time:.2f} seconds")
            
            # Get predicted probabilities or decision function
            try:
                y_proba = clf.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class (1)
            except AttributeError:
                logger.warning(f"{name} does not support predict_proba. Using decision function instead.")
                try:
                    y_proba = clf.decision_function(X_test)  # Fallback to decision function
                except AttributeError:
                    raise AttributeError(f"{name} does not have predict_proba or decision_function.")
            
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
