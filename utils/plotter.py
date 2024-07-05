import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

save_path = 'plots'
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

def plot_confusion_matrix(cm, classifier_name):
    plt.figure(figsize=(6, 4.75))
    sns.heatmap(pd.DataFrame(cm, index=['Legit', 'Fraud'], columns=['Legit', 'Fraud']), annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {classifier_name}")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    filename = os.path.join(save_path, f"confusion_matrix_{classifier_name}.png")
    plt.savefig(filename)  # Save the plot
    plt.close()

def plot_feature_importance(importances, feature_names, classifier_name):
    # Create a DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance - {classifier_name}')
    plt.tight_layout()

    # Save the plot
    filename = os.path.join(save_path, f"feature_importance_{classifier_name}.png")
    plt.savefig(filename)
    plt.close()

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

def save_distribution_plots(X_train, X_test, feature_names):
    for column in feature_names:
        plt.figure(figsize=(10, 5))
        sns.histplot(X_train[column], kde=True, color='blue', label='Train')
        sns.histplot(X_test[column], kde=True, color='red', label='Test')
        plt.title(f'Distribution of {column} in Train and Test sets')
        plt.legend()
        filename = os.path.join(save_path, f'{column}_distribution.png')
        plt.savefig(filename)
        plt.close()

def save_boxplots(X_train, feature_names):
    for column in feature_names[:5]:  # Checking the first 5 columns as an example
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=X_train[column])
        plt.title(f'Boxplot of {column} in Train set')
        filename = os.path.join(save_path, f'{column}_boxplot.png')
        plt.savefig(filename)
        plt.close()
