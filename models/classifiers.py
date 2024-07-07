import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils.logger import log_memory_usage, setup_logger
from utils.plotter import plot_feature_importance, plot_roc_pr_curves, save_distribution_plots, save_boxplots, plot_confusion_matrix

# Initialize logger
logger = setup_logger(__name__)

# Define all classifiers
all_classifiers = {
    'Random Forest': RandomForestClassifier(max_depth=100, min_samples_split=5, n_estimators=50)
}

def find_best_threshold(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = f1_scores.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold

def adjusted_prediction(clf, X, threshold=0.5):
    y_pred_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)
    y_pred_adj = (y_pred_prob >= threshold).astype(int)
    return y_pred_adj

def train_and_evaluate(X_train, X_test, y_train, y_test, best_estimators, config):
    results = {}
    for name in config['classifiers']:
        try:
            logger.info("Training %s...", name)
            clf = best_estimators.get(name, all_classifiers[name])
            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

            best_threshold = find_best_threshold(y_test, y_proba)
            y_pred_adj = adjusted_prediction(clf, X_test, best_threshold)

            cm = confusion_matrix(y_test, y_pred_adj)
            TN, FP, FN, TP = cm.ravel()
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if TP + FP != 0 else float('NaN')
            recall = TP / (FN + TP) if FN + TP != 0 else float('NaN')
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else float('NaN')

            results[name] = {
                'classification_report': classification_report(y_test, y_pred_adj),
                'confusion_matrix': cm,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'roc_auc': roc_auc_score(y_test, y_proba),
                'y_proba': y_proba
            }

            plot_confusion_matrix(cm, name)
            plot_roc_pr_curves(y_test, results[name]['y_proba'], name) 

            # Plot feature importance
            if hasattr(clf, 'named_steps') and 'classifier' in clf.named_steps:
                clf = clf.named_steps['classifier']
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                plot_feature_importance(importances, X_train.columns, name)
            elif hasattr(clf, 'coef_'):
                importances = np.abs(clf.coef_[0])  # For linear models like Logistic Regression
                plot_feature_importance(importances, X_train.columns, name)

            logger.info("Evaluating %s...", name)
        except Exception as e:
            logger.error("An error occurred while training %s: %s", name, e)
            continue
    return results

def hyperparameter_tuning(X_train, y_train, config, logger):
    tuned_parameters = config.get('tuned_parameters', {})

    selected_classifiers = config['classifiers']
    best_estimators = {}

    for name in selected_classifiers:
        if name in tuned_parameters:
            logger.info("Hyperparameter tuning for %s...", name)
            pipeline = Pipeline(steps=[('classifier', all_classifiers[name])])
            param_grid = {f'classifier__{key}': value for key, value in tuned_parameters[name].items()}
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='precision',
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=25),
                n_jobs=-1
            )
            log_memory_usage(logger)  # Log memory usage before grid search
            grid_search.fit(X_train, y_train)
            log_memory_usage(logger)  # Log memory usage after grid search
            best_estimators[name] = grid_search.best_estimator_
            logger.info("Best parameters for %s: %s", name, grid_search.best_params_)
        else:
            best_estimators[name] = all_classifiers[name]

    return best_estimators

def cross_validate_models(X, y, config, logger, best_estimators):
    selected_classifiers = config['classifiers']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=25)

    for name in selected_classifiers:
        logger.info("Cross-validating %s...", name)
        log_memory_usage(logger)  # Log memory usage before cross-validation
        clf = best_estimators.get(name, all_classifiers[name])

        accuracies, precisions, recalls, f1_scores = [], [], [], []

        try:
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                cm = confusion_matrix(y_test, y_pred)
                TN, FP, FN, TP = cm.ravel()
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP) if TP + FP != 0 else float('NaN')
                recall = TP / (FN + TP) if FN + TP != 0 else float('NaN')
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else float('NaN')

                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1_score)

            logger.info("Cross-validation accuracies for %s: %s", name, accuracies)
            logger.info("Mean accuracy for %s: %f", name, np.mean(accuracies))
            logger.info("Cross-validation precisions for %s: %s", name, precisions)
            logger.info("Mean precision for %s: %f", name, np.mean(precisions))
            logger.info("Cross-validation recalls for %s: %s", name, recalls)
            logger.info("Mean recall for %s: %f", name, np.mean(recalls))
            logger.info("Cross-validation F1 scores for %s: %s", name, f1_scores)
            logger.info("Mean F1 score for %s: %f", name, np.mean(f1_scores))
            
        except Exception as e:
            logger.error("Error during cross-validation for %s: %s", name, e)
        
        log_memory_usage(logger)  # Log memory usage after cross-validation
