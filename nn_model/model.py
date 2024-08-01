import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

def build_model(input_shape=(29, 1)):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=(29, 1), epochs=200, batch_size=2048, verbose=1):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        
    def build_model(self):
        return build_model(self.input_shape)
    
    def fit(self, X, y):
        self.model = self.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)
        
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        self.model.fit(X, y, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, 
                       callbacks=[early_stopping, reduce_lr], class_weight=class_weights_dict, verbose=self.verbose)
        return self
    
    def predict(self, X):
        preds = self.model.predict(X)
        return (preds > 0.5).astype("int32")
    
    def predict_proba(self, X):
        preds = self.model.predict(X)
        return np.hstack([1 - preds, preds])

# If train_and_evaluate_neural_network is required, define it as well
def train_and_evaluate_neural_network(X_train, y_train, X_test, y_test):
    model = build_model(input_shape=X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=150, batch_size=1054,
                        callbacks=[early_stopping, reduce_lr],
                        class_weight=class_weights_dict, verbose=1)
    return model, history
