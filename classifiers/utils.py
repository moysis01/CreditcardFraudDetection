


def find_best_threshold(y_test, y_proba, metric='f1'):
    from sklearn.metrics import precision_recall_curve          # Imports within functions to mitigate the risk of circular error
    """
    Find the best threshold for classification based on a specified metric.

    Parameters:
    y_test (array-like): True labels.
    y_proba (array-like): Predicted probabilities.
    metric (str): Metric to use for finding the best threshold. Defaults to 'f1'.

    Returns:
    float: Best threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = f1_scores.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold

def adjusted_prediction(clf, X, threshold=0.5):
    """
    Adjust predictions based on a given threshold.

    Parameters:
    clf: Classifier object.
    X (pd.DataFrame): Input features.
    threshold (float): Threshold for adjusting predictions. Defaults to 0.5.

    Returns:
    np.ndarray: Adjusted predictions.
    """
    y_pred_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)
    y_pred_adj = (y_pred_prob >= threshold).astype(int)
    return y_pred_adj

def get_stratified_kfold(random_state=25, n_splits=5):
    from sklearn.model_selection import StratifiedKFold
    """
    Get a StratifiedKFold object with the specified random state and number of splits.

    Parameters:
    random_state (int): Random state for reproducibility. Defaults to 25.
    n_splits (int): Number of splits. Defaults to 8.

    Returns:
    StratifiedKFold: StratifiedKFold object.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def calculate_metrics(y_true, y_pred, y_proba):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, classification_report

    """
    Calculate a variety of evaluation metrics.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    y_proba (array-like): Predicted probabilities.

    Returns:
    dict: Dictionary containing various evaluation metrics.
    """
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
    
    # Seting n_iter as a fraction of the total combinations, up to a maximum limit
    dynamic_n_iter = min(max_iter, int(total_combinations * 0.5))  # 50% of the total combinations, capped at max_iter
    return dynamic_n_iter
