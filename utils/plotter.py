import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score, precision_recall_curve
from utils import setup_logger

# Setup logger
logger = setup_logger(__name__)
save_path = 'plots'
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

def plot_roc_curve(y_test, y_proba, classifier_name, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, f"roc_curve_{classifier_name}.png")) 
    plt.close()

def plot_precision_recall_curve(y_test, y_proba, classifier_name, save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'AP = {precision.mean():.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {classifier_name}')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(save_path, f"precision_recall_curve_{classifier_name}.png")) 
    plt.close()

def plot_roc_pr_curves(y_test, y_proba, classifier_name, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plot_roc_curve(y_test, y_proba, classifier_name, save_path)
    plt.subplot(1, 2, 2)
    plot_precision_recall_curve(y_test, y_proba, classifier_name, save_path)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"roc_pr_curves_{classifier_name}.png"))
    plt.close()

def save_distribution_plots(X_train, X_test, feature_names, save_path):
    for column in feature_names:
        plt.figure(figsize=(10, 5))
        sns.histplot(X_train[column], kde=True, color='blue', label='Train')
        sns.histplot(X_test[column], kde=True, color='red', label='Test')
        plt.title(f'Distribution of {column} in Train and Test sets')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'{column}_distribution.png'))
        plt.close()

def save_boxplots(X_train, feature_names, save_path):
    for column in feature_names[:5]:  # Checking the first 5 columns as an example
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=X_train[column])
        plt.title(f'Boxplot of {column} in Train set')
        plt.savefig(os.path.join(save_path, f'{column}_boxplot.png'))
        plt.close()

def plot_feature_importance(clf, feature_names, classifier_name, save_path):
    try:
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importance = clf.coef_[0]
        else:
            raise AttributeError("Classifier does not have feature_importances_ or coef_ attributes.")

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances for {classifier_name}')
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(save_path, f"feature_importance_{classifier_name}.png"))
        plt.close()
    except Exception as e:
        logger.error(f"Error while plotting feature importance for {classifier_name}: {e}", exc_info=True)

def plot_confusion_matrix(cm, classifier_name, save_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legit', 'Fraud'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.savefig(os.path.join(save_path, f"confusion_matrix_{classifier_name}.png"))
    plt.close()

def plot_training_results(results, X_train, y_test, best_estimators, save_path='plots'):
    for name, metrics in results.items():
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            if isinstance(cm, (list, tuple)):
                for i, matrix in enumerate(cm):
                    plot_confusion_matrix(matrix, f"{name}_fold_{i+1}", save_path)
            else:
                plot_confusion_matrix(cm, f"{name}_overall", save_path)
        
        if 'y_probas' in metrics and 'y_preds' in metrics:
            for i, (y_proba) in enumerate(metrics['y_probas']):
                plot_roc_pr_curves(y_test, y_proba, f"{name}_fold_{i+1}", save_path)

        try:
            clf = best_estimators[name]
            if hasattr(clf, 'named_steps') and 'classifier' in clf.named_steps:
                clf = clf.named_steps['classifier']
            if hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_'):
                plot_feature_importance(clf, X_train.columns, name, save_path)
            else:
                logger.warning(f"Classifier {name} does not have feature importances or coefficients. Skipping feature importance plot.")
        except KeyError:
            logger.warning(f"Classifier {name} not found in best_estimators. Skipping feature importance plot.")

def plot_cross_validation_results(cv_results, X_train, feature_names, best_estimators, save_path='plots'):
    """Plot and save the results of cross-validation."""
    os.makedirs(save_path, exist_ok=True)

    for name, metrics in cv_results.items():
        if 'confusion_matrix' in metrics:
            confusion_matrices = metrics['confusion_matrix']
            if isinstance(confusion_matrices, (list, tuple)):
                for i, cm in enumerate(confusion_matrices):
                    plot_confusion_matrix(cm, f"{name}_fold_{i+1}", save_path)
            else:
                plot_confusion_matrix(confusion_matrices, f"{name}_overall", save_path)

        # Removed Precision-Recall and ROC curves plotting for each fold

        try:
            clf = best_estimators[name]
            if hasattr(clf, 'named_steps') and 'classifier' in clf.named_steps:
                clf = clf.named_steps['classifier']
            if hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_'):
                plot_feature_importance(clf, feature_names, name, save_path)
            else:
                logger.warning(f"Classifier {name} does not have feature importances or coefficients. Skipping feature importance plot.")
        except KeyError:
            logger.warning(f"Classifier {name} not found in best_estimators. Skipping feature importance plot.")
