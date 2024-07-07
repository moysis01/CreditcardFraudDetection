from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

def get_voting_classifier(config, best_estimators):
    """
    Creates a voting classifier based on the specified configuration.
    """
    # Ensure all specified classifiers are available in the best_estimators dictionary
    estimators = []
    for name in config['classifiers']:
        if name in best_estimators:
            estimators.append((name, best_estimators[name]))
        else:
            logger.error(f"Classifier {name} not found in best_estimators dictionary.")
    
    if not estimators:
        raise ValueError("No valid classifiers specified for the voting classifier.")
    
    # Configure voting type: 'soft' for probabilities, 'hard' for majority voting
    voting_type = config.get('voting', 'soft')
    if voting_type not in ['soft', 'hard']:
        logger.error(f"Invalid voting type specified: {voting_type}. Defaulting to 'soft'.")
        voting_type = 'soft'
    
    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
    return voting_clf

def train_and_evaluate_voting_classifier(X_train, X_test, y_train, y_test, best_estimators, config):
    """
    Trains and evaluates the voting classifier.
    """
    try:
        # Use best estimators if available
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
        
        results = {
            'classification_report': classification_report(y_test, y_pred_voting),
            'confusion_matrix': confusion_matrix(y_test, y_pred_voting),
            'accuracy': accuracy_score(y_test, y_pred_voting),
            'roc_auc': roc_auc_score(y_test, y_proba_voting),
            'y_proba': y_proba_voting
        }
        return results
    except Exception as e:
        logger.error(f"An error occurred while training and evaluating the voting classifier: {str(e)}")
        return {}
