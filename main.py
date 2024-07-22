import json
import logging
import numpy as np
from classifiers.cv import cross_validate_models
from classifiers.ensemble import train_and_evaluate_voting_classifier
from classifiers.hypertuning import hyperparameter_tuning
from classifiers.train import training
from classifiers.utils import evaluate_random_states
from utils import setup_logger, log_memory_usage
from preprocessing import load_data, preprocess_data
from classifiers.classifier_init import all_classifiers
from utils.plotter import plot_training_results, plot_cross_validation_results

# Initialize logger
logger = setup_logger(__name__, log_file='results.log', console_level=logging.INFO, file_level=logging.INFO)

def load_config(config_path):
    """Load configuration file from the specified path."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        logger.info("Configuration loaded successfully from %s", config_path)
        return config
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON from config file: %s", e)
        raise
    except FileNotFoundError:
        logger.error("Configuration file not found at %s", config_path)
        raise

def main():
    logger.info("Starting script execution...")
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config('configs/config.json')  # Adjust path if necessary
        log_memory_usage(logger)

        # Load dataset
        logger.info("Loading dataset...")
        path = 'C:\\Users\\ke1no\\OneDrive - Sheffield Hallam University\\Year3\\Disseration\\Dataset-Approved\\creditcard.csv'
        library_path='C:\\Users\\c0003255\\OneDrive - Sheffield Hallam University\\Year3\\Disseration\\Dataset-Approved\\creditcard.csv'
        df = load_data(library_path)
        logger.info("Dataset loaded with shape: %s", df.shape)
        log_memory_usage(logger)
        
        """
        # Define a range of random states and convert to a list
        random_state_range = (0, 100)  # Example range from 0 to 100
        random_states = list(range(random_state_range[0], random_state_range[1] + 1))

        # Find the best random state
        logger.info("Evaluating random states to find the best one...")
        best_random_state, random_state_results = evaluate_random_states(df, all_classifiers, config, random_states)
        logger.info(f"Best random state found: {best_random_state}")

        # Log detailed results for all evaluated random states
        logger.info("Detailed random state evaluation results:")
        for random_state, score in random_state_results.items():
            logger.info(f"Random state {random_state}: Combined score = {score:.4f}")

        # Log the best random state and its score
        logger.info(f"Best random state found: {best_random_state} with a combined score of {random_state_results[best_random_state]:.4f}") """

        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, X, y = preprocess_data(df, config,25)
        logger.info("Data preprocessing complete.")
        log_memory_usage(logger)

        # Hyperparameter tuning
        best_estimators = {}
        if config.get('hyperparameter_tuning'):
            logger.info("Starting hyperparameter tuning...")
            best_estimators = hyperparameter_tuning(X_train, y_train, config, logger)
            logger.info("Hyperparameter tuning complete.")
            log_memory_usage(logger)

        # Use default classifiers for those not in best_estimators
        for name in config['classifiers']:
            if name not in best_estimators:
                logger.info("Using default classifier for %s", name)
                best_estimators[name] = all_classifiers[name]

        # Cross-validation
        if config.get('cross_validation'):
            logger.info("Starting cross-validation...")
            cv_results = cross_validate_models(X, y, config, logger, best_estimators)
            logger.info("Cross-validation complete.")
            log_memory_usage(logger)

            # Log cross-validation results
            for name, metrics in cv_results.items():
                logger.info(f"\nCross-validation results for {name}:")
                for metric_name, values in metrics.items():
                    if metric_name not in ["y_preds", "y_probas"]: 
                        mean_value = np.mean(values)
                        logger.info(f"Results for Cross-validation {name}: Mean {metric_name}: {mean_value:.4f}")

            # Plot cross-validation results
            plot_cross_validation_results(cv_results, X_train, X.columns, best_estimators)

        # Train and evaluate classifiers
        logger.info("Training and evaluating classifiers...")
        results = training(X_train, X_test, y_train, y_test, best_estimators, config)
        logger.info("Training and evaluation complete.")
        log_memory_usage(logger)

        # Log individual classifier results
        for name, metrics in results.items():
            logger.info(f"\nResults for {name}:")
            logger.info(f"Results for {name} Classification Report:\n{metrics.get('classification_report', 'N/A')}")
            logger.info(f"Results for {name} Confusion Matrix:\n{metrics.get('confusion_matrix', 'N/A')}")
            logger.info(f"Results for {name} Accuracy: {metrics.get('accuracy', 'N/A'):.4f}, Precision: {metrics.get('precision', 'N/A'):.4f}, Recall: {metrics.get('recall', 'N/A'):.4f}, F1 Score: {metrics.get('f1_score', 'N/A'):.4f}, ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}, MCC: {metrics.get('mcc', 'N/A'):.4f}")
        plot_training_results(results, X_train, y_test, best_estimators)

        # Ensemble Voting Classifier
        if config.get('ensemble'):
            logger.info("Training and evaluating ensemble voting classifier...")
            voting_results = train_and_evaluate_voting_classifier(X_train, X_test, y_train, y_test, best_estimators, config)
            logger.info("Ensemble voting classifier training complete.")
            log_memory_usage(logger)

            # Log ensemble results
            if "VotingClassifier" in voting_results:
                metrics = voting_results["VotingClassifier"]
                logger.info(f"\nResults for Ensemble Voting Classifier:")
                logger.info(f"Results for Ensemble Voting Classifier Classification Report:\n{metrics.get('classification_report', 'N/A')}")
                logger.info(f"Results for Ensemble Voting Classifier Confusion Matrix:\n{metrics.get('confusion_matrix', 'N/A')}")
                logger.info(f"Results for Ensemble Voting Classifier Accuracy: {metrics.get('accuracy', 'N/A'):.4f}, Precision: {metrics.get('precision', 'N/A'):.4f}, Recall: {metrics.get('recall', 'N/A'):.4f}, F1 Score: {metrics.get('f1_score', 'N/A'):.4f}, ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}, MCC: {metrics.get('mcc', 'N/A'):.4f}")

                # Plot Ensemble results
                plot_training_results(voting_results, X_train, y_test, best_estimators)
                logger.info("Ensemble results plotted.")
                log_memory_usage(logger)

    except Exception as e:
        logger.error("An error occurred in the main execution: %s", e, exc_info=True)
        log_memory_usage(logger)

if __name__ == "__main__":
    main()