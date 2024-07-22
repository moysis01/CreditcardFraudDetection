from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

all_classifiers = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(n_estimators=174,learning_rate=0.12),
    'MLP': MLPClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True)
}
