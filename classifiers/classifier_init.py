from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

all_classifiers = {
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(), #,colsample_bytree=0.3,min_child_weight=9,max_depth=9,subsample=0.9,n_estimators=174,learning_rate=0.12,gamma=0.3
    'MLP': MLPClassifier(), #alpha= 0.0001, hidden_layer_sizes= (150, 125, 100, 75, 50, 25), learning_rate_init= 0.001
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True,kernel='linear'),
    'LogisticRegression': LogisticRegression(max_iter=1000)
}
