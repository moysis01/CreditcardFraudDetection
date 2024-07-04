from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from utils.logger import setup_logger
from models.classifiers import all_classifiers

# Setup logger
logger = setup_logger(__name__)

def get_voting_classifier(config):
    """
    Creates a voting classifier based on the specified configuration.
    """
    # Ensure all specified classifiers are available in the all_classifiers dictionary
    estimators = []
    for name in config['classifiers']:
        if name in all_classifiers:
            estimators.append((name, all_classifiers[name]))
        else:
            logger.error(f"Classifier {name} not found in all_classifiers dictionary.")
            continue

    # Configure voting type: 'soft' for probabilities, 'hard' for majority voting
    voting_type = config.get('voting', 'soft')
    voting_clf = VotingClassifier(estimators=estimators, voting=voting_type)
    return voting_clf

def train_and_evaluate_voting_classifier(X_train, X_test, y_train, y_test, config):
    """
    Trains and evaluates the voting classifier.
    """
    try:
        voting_clf = get_voting_classifier(config)
        logger.info("Training Voting Classifier...")
        voting_clf.fit(X_train, y_train)

        logger.info("Evaluating Voting Classifier...")
        y_pred_voting = voting_clf.predict(X_test)
        y_proba_voting = voting_clf.predict_proba(X_test)[:, 1] if hasattr(voting_clf, "predict_proba") else voting_clf.decision_function(X_test)

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

# Example configuration loading (assuming configuration loading function exists)
config = {
    'classifiers': ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier'],
    'voting': 'soft'  # Can be changed to 'hard' if needed
}
