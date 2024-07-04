import pandas as pd
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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score
import seaborn as sns
import os

save_path = 'plots'
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

logger = setup_logger(__name__)

all_classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
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


def train_and_evaluate(X_train, X_test, y_train, y_test, config):
    results = {}
    for name, clf in all_classifiers.items():
        if name in config['classifiers']:  # Check if the classifier is listed in the config
            try:
                logger.info(f"Training {name}...")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
                
                # Compute confusion matrix and extract evaluation metrics
                cm = confusion_matrix(y_test, y_pred)
                TN, FP, FN, TP = cm.ravel()
                accuracy = (TP + TN) / (TP + TN + FP + FN)
                precision = TP / (TP + FP) if TP + FP != 0 else float('NaN')
                recall = TP / (FN + TP) if FN + TP != 0 else float('NaN')
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else float('NaN')

                # Store results
                results[name] = {
                    'classification_report': classification_report(y_test, y_pred),
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
                plt.show()
                


                # Plot ROC and precision-recall curves
                plot_roc_pr_curves(y_test, y_proba, name)

                logger.info(f"Evaluating {name}...")
            except Exception as e:
                logger.error(f"An error occurred while training {name}: {e}")
                continue
    return results







def hyperparameter_tuning(X_train, y_train, config):
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
            'n_estimators': [100, 200, 300, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
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
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=tuned_parameters[name],
                scoring='accuracy',
                cv=5,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_estimators[name] = grid_search.best_estimator_
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        else:
            best_estimators[name] = all_classifiers[name]

    return best_estimators



def cross_validate_models(X, y, config, logger):
    selected_classifiers = config['classifiers']
    for name in selected_classifiers:
        logger.info(f"Cross-validating {name}...")
        log_memory_usage(logger)  # Log memory usage before cross-validation
        clf = all_classifiers[name]
        try:
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
            logger.info(f"Cross-validation scores for {name}: {scores}")
            logger.info(f"Mean accuracy for {name}: {scores.mean()}")
        except Exception as e:
            logger.error(f"Error during cross-validation for {name}: {str(e)}")
        log_memory_usage(logger)  # Log memory usage after cross-validation


