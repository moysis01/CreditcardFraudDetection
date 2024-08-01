import time
from imblearn.pipeline import Pipeline as ImbPipeline
from utils.logger import log_memory_usage, setup_logger
from classifiers.utils import get_stratified_kfold, calculate_metrics
from classifiers.classifier_init import all_classifiers
from utils.logger import logging
from preprocessing import sampler_classes
import numpy as np
import pandas as pd
from nn_model.model import build_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve

# Initialize logger
logger = setup_logger(__name__)

def cross_validate_models(X: pd.DataFrame, y: pd.Series, config: dict, logger: logging.Logger, best_estimators: dict = None) -> dict:
    """
    Perform cross-validation for the selected classifiers.

    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Target labels.
        config (dict): Configuration dictionary containing classifier and resampling settings.
        logger (logging.Logger): Logger instance for logging.
        best_estimators (dict): Dictionary of best estimators if available.

    Returns:
        dict: Dictionary containing cross-validation results.
    """
    selected_classifiers = config['classifiers']
    skf = get_stratified_kfold()

    if not best_estimators:
        best_estimators = {}

    resampling_methods = config.get('resampling', [])
    resampling_params = config.get('resampling_params', {})

    cv_results = {}

    for name in selected_classifiers:
        logger.info(f"Cross-validating {name}...")
        log_memory_usage(logger)

        if name == "Neural Network":
            # Special handling for Neural Network model
            model = build_model()
            epochs = 200
            batch_size = 2048
            early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=0.0001)

            # Initialize lists to store metrics for each fold
            accuracies, precisions, recalls, f1_scores, roc_aucs, mccs, y_preds, y_probas = [], [], [], [], [], [], [], []

            for split_index, (train_index, test_index) in enumerate(skf.split(X, y), 1):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                try:
                    split_start_time = time.time()  # Start timing for this split

                    # Reshape for Conv1D
                    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

                    # Compute class weights
                    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
                    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

                    model.fit(X_train, y_train, validation_data=(X_test, y_test),
                              epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, reduce_lr],
                              class_weight=class_weights_dict, verbose=0)
                    split_end_time = time.time()  # End timing for this split

                    logger.info(f"Cross-validating split {split_index} for {name}: {split_end_time - split_start_time:.2f} seconds")

                    y_proba = model.predict(X_test).ravel()
                    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
                    optimal_idx = np.argmax(precision * recall)
                    optimal_threshold = thresholds[optimal_idx]
                    y_pred = (y_proba >= optimal_threshold).astype(int)

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

            # Aggregate results for the Neural Network
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

            # Log average metrics for the Neural Network
            logger.info(f"Average metrics for {name}:")
            for metric_name, values in cv_results[name].items():
                if metric_name not in ["y_preds", "y_probas"]:  # Exclude y_preds and y_probas from averaging
                    logger.info(f"  Mean {metric_name}: {np.mean(values):.4f}")

            log_memory_usage(logger)
            continue

        # Handling for other classifiers
        clf = best_estimators.get(name)

        if clf is None:
            logger.warning(f"No tuned model found for {name}. Using default classifier with pipeline.")
            steps = []

            # Apply resampling methods if specified
            for method in resampling_methods:
                sampler_class = sampler_classes.get(method.upper())
                if not sampler_class:
                    raise ValueError(f"Invalid resampling method '{method}' specified in config.")
                method_params = {key: resampling_params[key] for key in resampling_params if key in sampler_class._parameters}
                steps.append(('resampler', sampler_class(**method_params)))

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
