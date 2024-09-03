from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from nn_model.model import build_model
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

all_classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'k-NN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(),

    'XGBoost': XGBClassifier(), #colsample_bytree=0.3,min_child_weight=9,max_depth=9,subsample=0.9,n_estimators=174,learning_rate=0.12,gamma=0.3
    'MLP': MLPClassifier(max_iter=1500), #alpha= 0.0001, hidden_layer_sizes= (150, 125, 100, 75, 50, 25), learning_rate_init= 0.001
    'Neural Network': build_model  # neural network placeholder 
    
}
