import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from utils.logger import log_memory_usage, setup_logger
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, precision_score

# Set up save path for plots
save_path = 'plots'
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist


"""
{
    "classifiers": [
      "Logistic Regression",
      "Decision Tree",
      "Random Forest",
      "KNN",
      "Gradient Boosting",
      "XGBoost",
      "LightGBM",
      "AdaBoost",
      "Naive Bayes",
      "MLP",
      "CatBoost"
    ]
}

"""

logger = setup_logger(__name__)

all_classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=50),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Naive Bayes': GaussianNB(),
    'MLP': MLPClassifier(),
    'CatBoost': CatBoostClassifier()
}

def plot_feature_importance(importance, features, model_name):
    # Create a dataframe for better plotting
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance for {model_name}')
    plt.tight_layout()
    plt.show()

def plot_roc_pr_curves(y_test, y_proba, classifier_name):
    plt.figure(figsize=(12, 5))
    plot_roc_curve(y_test, y_proba, classifier_name)
    plot_precision_recall_curve(y_test, y_proba, classifier_name)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_proba, classifier_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.legend(loc='lower right')

def plot_precision_recall_curve(y_test, y_proba, classifier_name):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'AP = {precision.mean():.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {classifier_name}')
    plt.legend(loc='lower left')

def find_best_threshold(y_test, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall)
    best_index = f1_scores.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold

def adjusted_prediction(clf, X, threshold=0.5):
    """
    Adjust prediction based on a specified threshold.
    """
    y_pred_prob = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X)
    y_pred_adj = (y_pred_prob >= threshold).astype(int)
    return y_pred_adj

def train_and_evaluate(X_train, X_test, y_train, y_test, best_estimators, config):
    results = {}
    for name in config['classifiers']:
        try:
            logger.info(f"Training {name}...")

            # Use best estimators if available, otherwise use the default classifier
            clf = best_estimators.get(name, all_classifiers[name])

            clf.fit(X_train, y_train)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            
            # Find the best threshold
            best_threshold = find_best_threshold(y_test, y_proba)
            y_pred_adj = adjusted_prediction(clf, X_test, best_threshold)

            # Compute confusion matrix and extract evaluation metrics
            cm = confusion_matrix(y_test, y_pred_adj)
            TN, FP, FN, TP = cm.ravel()
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if TP + FP != 0 else float('NaN')
            recall = TP / (FN + TP) if FN + TP != 0 else float('NaN')
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else float('NaN')

            # Store results
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
            filename = os.path.join(save_path, f"confusion_matrix_{name}.png")
            # Plot confusion matrix
            plt.figure(figsize=(6, 4.75))
            sns.heatmap(pd.DataFrame(cm, index=['Legit', 'Fraud'], columns=['Legit', 'Fraud']), annot=True, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {name}")
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.savefig(filename)  # Save the plot

            # Plot feature importance
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                plot_feature_importance(importances, X_train.columns, name)
            elif hasattr(clf, 'coef_'):
                importances = np.abs(clf.coef_[0])  # For linear models like Logistic Regression
                plot_feature_importance(importances, X_train.columns, name)

            logger.info(f"Evaluating {name}...")
        except Exception as e:
            logger.error(f"An error occurred while training {name}: {e}")
            continue
    return results









def hyperparameter_tuning(X_train, y_train, config,logger):
    tuned_parameters = {
        'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        },
        'Decision Tree': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10]
        },
        'Random Forest': {
        
        },
        'SVM': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 127, 255],
            'max_depth': [10, 20, 30, -1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'base_estimator': [DecisionTreeClassifier(max_depth=1)]
        },
        'Naive Bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        },
        'MLP': {
            'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 25)],
            'activation': ['tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        },
        'CatBoost': {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [3, 5, 7]
        }

    }

    selected_classifiers = config['classifiers']
    best_estimators = {}

    for name in selected_classifiers:
        if name in tuned_parameters:
            logger.info(f"Hyperparameter tuning for {name}...")
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
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            best_estimators[name] = all_classifiers[name]

    return best_estimators

def cross_validate_models(X, y, config, logger, best_estimators):
    selected_classifiers = config['classifiers']
    for name in selected_classifiers:
        logger.info(f"Cross-validating {name}...")
        log_memory_usage(logger)  # Log memory usage before cross-validation
        clf = best_estimators.get(name, all_classifiers[name])
        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            logger.info(f"Cross-validation scores for {name}: {scores}")
            logger.info(f"Mean accuracy for {name}: {scores.mean()}")
        except Exception as e:
            logger.error(f"Error during cross-validation for {name}: {str(e)}")
        log_memory_usage(logger)  # Log memory usage after cross-validation

