def find_best_threshold(y_test, y_proba, metric='f1'):
    from sklearn.metrics import (precision_recall_curve)
    """Find the best threshold for classification based on a specified metric."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = f1_scores.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold

def adjusted_prediction(clf, X, threshold=0.5):
    """Adjust predictions based on a given threshold."""
    y_pred_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)
    y_pred_adj = (y_pred_prob >= threshold).astype(int)
    return y_pred_adj

def get_stratified_kfold(random_state=25, n_splits=8):
    from sklearn.model_selection import StratifiedKFold
    """Get a StratifiedKFold object with the specified random state and number of splits."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def calculate_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, 
    roc_auc_score,classification_report)
    """Calculate a variety of evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=1),
        'recall': recall_score(y_true, y_pred, zero_division=1),
        'f1_score': f1_score(y_true, y_pred, zero_division=1),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_proba': y_proba,
        'classification_report': classification_report(y_true, y_pred)
    }
    return metrics

def evaluate_random_states(data, best_estimators, config, random_states, metric_weights=None):
    from classifiers import training
    from preprocessing.preprocess import preprocess_data
    from utils.logger import setup_logger
    logger = setup_logger(__name__)
    
    if metric_weights is None:
        metric_weights = {'accuracy': 0.2, 'precision': 0.2, 'recall': 0.2, 'f1_score': 0.2, 'roc_auc': 0.1, 'mcc': 0.1}
    
    best_random_state = None
    best_score = -float('inf')
    random_state_results = {}

    for random_state in random_states:
        logger.info(f"Evaluating random_state={random_state}...")

        try:
            X_train, X_test, y_train, y_test, _, _ = preprocess_data(data, config, random_state=random_state)
            results = training(X_train, X_test, y_train, y_test, best_estimators, config)
            
            combined_score = 0
            for metrics in results.values():
                for metric, weight in metric_weights.items():
                    combined_score += metrics[metric] * weight

            combined_score /= len(results)
            random_state_results[random_state] = combined_score

            if combined_score > best_score:
                best_score = combined_score
                best_random_state = random_state

            logger.info(f"Random_state={random_state} evaluation complete. Combined score: {combined_score:.4f}")

        except Exception as e:
            logger.error(f"An error occurred while evaluating random_state={random_state}: {e}", exc_info=True)

    logger.info(f"Best random_state={best_random_state} with a combined score of {best_score:.4f}")
    return best_random_state, random_state_results


def calculate_n_iter(param_distributions, max_iter=100):
    """
    Calculate the dynamic n_iter based on the number of parameter combinations.

    Parameters:
    param_distributions (dict): Dictionary of parameter distributions.
    max_iter (int): Maximum number of iterations.

    Returns:
    int: Calculated n_iter.
    """
    total_combinations = 1
    for param, values in param_distributions.items():
        total_combinations *= len(values)
    
    # Set n_iter as a fraction of the total combinations, up to a maximum limit
    dynamic_n_iter = min(max_iter, int(total_combinations * 0.95))  # 90% of the total combinations, capped at max_iter
    return dynamic_n_iter
