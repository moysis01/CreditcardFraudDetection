from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

all_classifiers = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(), #n_estimators=174,learning_rate=0.12,max_depth=9,min_child_weight=9,subsample=0.9,gamma=0.3,colsample_bytree=0.3
    'MLP': MLPClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True)
}
