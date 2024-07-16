from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef, 
    roc_auc_score, precision_recall_curve,classification_report
)
from sklearn.model_selection import StratifiedKFold
import numpy as np

def find_best_threshold(y_test, y_proba,metric='f1'):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = f1_scores.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold

def adjusted_prediction(clf, X, threshold=0.5):
    y_pred_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)
    y_pred_adj = (y_pred_prob >= threshold).astype(int)
    return y_pred_adj

def get_stratified_kfold(random_state=25, n_splits=8):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def calculate_metrics(y_true, y_pred, y_proba):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=1)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=1)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=1)
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    metrics['y_proba'] = y_proba  # Add y_proba to the metrics
    metrics['classification_report'] = classification_report(y_true, y_pred)
    
    return metrics

