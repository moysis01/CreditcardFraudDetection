import json
from utils import setup_logger
from preprocessing import load_data, preprocess_data
from models.classifiers import train_and_evaluate, plot_roc_pr_curves, hyperparameter_tuning, cross_validate_models
from models.ensemble import train_and_evaluate_voting_classifier

logger = setup_logger(__name__)

def load_config(config_path):
    """Load configuration file from specified path."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info("Configuration loaded successfully.")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from config file: {e}")
        raise
    except FileNotFoundError:
        logger.error("Configuration file not found.")
        raise

if __name__ == "__main__":
    try:
        logger.info("Loading configuration...")
        config = load_config('classifiers_config.json')

        logger.info("Loading dataset...")
        df = load_data('data/creditcard.csv')

        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(df)

        if config.get('cross_validation'):
            logger.info("Cross-validating models...")
            cross_validate_models(X_train, y_train, config)

        if config.get('hyperparameter_tuning'):
            logger.info("Hyperparameter tuning...")
            hyperparameter_tuning(X_train, y_train, config)

        logger.info("Training and evaluating classifiers...")
        results = train_and_evaluate(X_train, X_test, y_train, y_test, config)
        for name, metrics in results.items():
            logger.info(f"Results for {name}:")
            logger.info(f"Classification Report:\n{metrics['classification_report']}")
            logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
            logger.info(f"Accuracy: {metrics['accuracy']}, Precision: {metrics['precision']}, Recall: {metrics['recall']}, F1 Score: {metrics['f1_score']}, ROC AUC: {metrics['roc_auc']}")
            plot_roc_pr_curves(y_test, metrics['y_proba'], name)  # Visualizing ROC and PR curves

        if config.get('ensemble'):
            logger.info("Training and evaluating ensemble voting classifier...")
            voting_results = train_and_evaluate_voting_classifier(X_train, X_test, y_train, y_test, config)
            logger.info("Ensemble Voting Classifier results:")
            logger.info(voting_results['classification_report'])
            logger.info(f"Confusion Matrix:\n{voting_results['confusion_matrix']}")
            logger.info(f"Accuracy: {voting_results['accuracy']}, ROC AUC: {voting_results['roc_auc']}")
            plot_roc_pr_curves(y_test, voting_results['y_proba'], "Ensemble Voting Classifier")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {e}")