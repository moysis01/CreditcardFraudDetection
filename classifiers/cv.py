import time
from imblearn.pipeline import Pipeline as ImbPipeline
from utils.logger import log_memory_usage, setup_logger
from classifiers.utils import get_stratified_kfold, calculate_metrics
from classifiers.classifier_init import all_classifiers
from utils.logger import logging
from preprocessing import sampler_classes, all_scalers
import numpy as np
import pandas as pd

# Initialize logger
logger = setup_logger(__name__)

def cross_validate_models(X: pd.DataFrame, y: pd.Series, config: dict, logger: logging.Logger, best_estimators: dict = None) -> dict:
    selected_classifiers = config['classifiers']
    skf = get_stratified_kfold()

    if not best_estimators:
        best_estimators = {}

    resampling_config = config.get('resampling', {})
    resampling_method = resampling_config.get('method', None)
    resampling_params = resampling_config.get('params', {})
    scaler_name = config.get('scaling', None)

    cv_results = {}

    for name in selected_classifiers:
        logger.info("Cross-validating %s...", name)
        log_memory_usage(logger)

        # Get the classifier or pipeline
        clf = best_estimators.get(name)

        if clf is None:
            logger.warning(f"No tuned model found for {name}. Using default classifier with pipeline.")
            steps = []
            if resampling_method:
                sampler_class = sampler_classes.get(resampling_method)
                if not sampler_class:
                    raise ValueError(f"Invalid resampling method '{resampling_method}' in config.")
                steps.append(('resampler', sampler_class(**resampling_params)))

            if scaler_name:
                scaler_class = all_scalers.get(scaler_name)
                if not scaler_class:
                    raise ValueError(f"Invalid scaler '{scaler_name}' in config.")
                steps.append(('scaler', scaler_class))

            steps.append(('classifier', all_classifiers[name]))
            clf = ImbPipeline(steps=steps)  # Use ImbPipeline when resampling is involved

        # Initialize lists to store metrics for each fold
        accuracies, precisions, recalls, f1_scores, roc_aucs, mccs, y_preds, y_probas = [], [], [], [], [], [], [], []

        for split_index, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            try:
                split_start_time = time.time()  # Start timing for this split
                clf.fit(X_train, y_train)
                split_end_time = time.time()  # End timing for this split
                
                logger.info(f"Cross-validating split {split_index} for {name}: {split_end_time - split_start_time:.2f} seconds")

                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

                metrics = calculate_metrics(y_test, y_pred, y_proba)
                accuracies.append(metrics["accuracy"])
                precisions.append(metrics["precision"])
                recalls.append(metrics["recall"])
                f1_scores.append(metrics["f1_score"])
                roc_aucs.append(metrics["roc_auc"])
                mccs.append(metrics["mcc"])
                y_preds.append(y_pred)
                y_probas.append(y_proba)

            except (ValueError, AttributeError, TypeError) as sk_err:
                logger.error(f"Error during cross-validation split {split_index} for {name}: {sk_err}")
                continue
            except ZeroDivisionError as zde:
                logger.error(f"ZeroDivisionError during cross-validation split {split_index} for {name}: {zde}")
                continue
            except Exception as e:
                logger.exception(f"Unexpected error during cross-validation split {split_index} for {name}: {e}")
                continue

        # Aggregate results for each classifier
        cv_results[name] = {
            "accuracies": accuracies,
            "precisions": precisions,
            "recalls": recalls,
            "f1_scores": f1_scores,
            "roc_aucs": roc_aucs,
            "mccs": mccs,
            "y_preds": y_preds,
            "y_probas": y_probas,
        }

        # Log average metrics for the classifier
        logger.info(f"Average metrics for {name}:")
        for metric_name, values in cv_results[name].items():
            if metric_name not in ["y_preds", "y_probas"]:  # Exclude y_preds and y_probas from averaging
                logger.info(f"  Mean {metric_name}: {np.mean(values):.4f}")

        log_memory_usage(logger)

    return cv_results
