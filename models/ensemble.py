from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from typing import Dict, Any
import pandas as pd
import numpy as np
from utils.logger import setup_logger
from models.classifiers import find_best_threshold, adjusted_prediction

logger = setup_logger(__name__)

def get_voting_classifier(config: Dict[str, Any], best_estimators: Dict[str, Any]) -> VotingClassifier:
    estimators = []
    for name in config['classifiers']:
        if name in best_estimators:
            estimators.append((name, best_estimators[name]))
        else:
            logger.error(f"Classifier {name} not found in best_estimators dictionary.")
    
    if not estimators:
        raise ValueError("No valid classifiers specified for the voting classifier.")
    
    # voting type: 'soft' for probabilities, 'hard' for majority voting
    voting_type = config.get('voting', 'soft')
    if voting_type not in ['soft', 'hard']:
        logger.error(f"Invalid voting type specified: {voting_type}. Defaulting to 'soft'.")
        voting_type = 'soft'
    
    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
    return voting_clf

def train_and_evaluate_voting_classifier(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, 
    best_estimators: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    try:
        voting_clf = get_voting_classifier(config, best_estimators)
        logger.info("Training Voting Classifier...")
        voting_clf.fit(X_train, y_train)

        logger.info("Evaluating Voting Classifier...")
        y_pred_voting = voting_clf.predict(X_test)
        
        if hasattr(voting_clf, "predict_proba"):
            y_proba_voting = voting_clf.predict_proba(X_test)[:, 1]
        elif hasattr(voting_clf, "decision_function"):
            y_proba_voting = voting_clf.decision_function(X_test)
        else:
            y_proba_voting = y_pred_voting  # Fallback if probabilities are not available

        best_threshold = find_best_threshold(y_test, y_proba_voting)
        y_pred_adj = adjusted_prediction(voting_clf, X_test, best_threshold)
        
        cm = confusion_matrix(y_test, y_pred_adj)
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if TP + FP != 0 else float('NaN')
        recall = TP / (FN + TP) if FN + TP != 0 else float('NaN')
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else float('NaN')

        # Matthews Correlation Coefficient (MCC)
        MCC_num = (TN * TP) - (FP * FN)
        MCC_denom = np.sqrt((FP + TP) * (FN + TP) * (TN + FP) * (TN + FN))
        MCC = MCC_num / MCC_denom if MCC_denom != 0 else float('NaN')

        results = {
            'classification_report': classification_report(y_test, y_pred_adj),
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc_score(y_test, y_proba_voting),
            'mcc': MCC,
            'y_proba': y_proba_voting
        }
        logger.info("Voting Classifier Evaluation Completed.")
        return results
    except Exception as e:
        logger.error(f"An error occurred while training and evaluating the voting classifier: {str(e)}")
        return {}
