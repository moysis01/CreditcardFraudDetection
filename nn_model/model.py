import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from kerastuner.tuners import Hyperband

def build_model(hp=None, input_shape=(29, 1)):
    """
    Builds a convolutional neural network model with reduced tunable parameters.

    Parameters:
    - hp (HyperParameters): The hyperparameters object used for tuning. If None, default values are used.
    - input_shape (tuple): Shape of the input data.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    if hp is not None:
        filters = hp.Int('filters', min_value=32, max_value=64, step=32)  
        dropout_conv = hp.Float('dropout_conv', min_value=0.2, max_value=0.4, step=0.1)  
        num_conv_layers = 1  
        dense_units = hp.Int('units', min_value=32, max_value=64, step=32) 
        dropout_dense = 0.3  
        learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])  
    else:
        # Default values
        filters = 32
        dropout_conv = 0.3
        num_conv_layers = 1
        dense_units = 32
        dropout_dense = 0.3
        learning_rate = 0.001
    
    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_conv))
    
    if num_conv_layers > 1:
        for i in range(num_conv_layers - 1):
            model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_conv))
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_dense))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def tune_hyperparameters(directory, project_name, X_train, y_train):
    """
    Tunes hyperparameters using Hyperband with reduced trials and fewer parameters.

    Parameters:
    - directory (str): The directory where the Keras Tuner results are saved.
    - project_name (str): The name of the project (subdirectory) within the directory.
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    """
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20, 
        factor=4,  
        directory=directory,
        project_name=project_name,
        overwrite=False  
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping]) 
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def load_best_hyperparameters(directory, project_name):
    """
    Loads the best hyperparameters from a Keras Tuner directory.

    Parameters:
    - directory (str): The directory where the Keras Tuner results are saved.
    - project_name (str): The name of the project (subdirectory) within the directory.

    Returns:
    - HyperParameters: The best hyperparameters found during tuning.
    """
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=30,
        directory=directory,
        project_name=project_name,
        overwrite=False  # Do not overwrite existing tuner files
    )
    
    tuner.reload() 
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps

class DNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=(29, 1), epochs=200, batch_size=2048, verbose=1, 
                 best_hps=None, directory=None, project_name=None, X_train=None, y_train=None):
        """
        Initializes the DNNClassifier.

        Parameters:
        - input_shape (tuple): Shape of the input data.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - verbose (int): Verbosity mode.
        - best_hps (HyperParameters): Best hyperparameters from tuning. If None, defaults will be used.
        - directory (str): Directory where Keras Tuner results are saved.
        - project_name (str): Name of the project (subdirectory) within the directory.
        - X_train (np.ndarray): Training data.
        - y_train (np.ndarray): Training labels.
        """
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.best_hps = best_hps  # Store the best hyperparameters
        self.directory = directory
        self.project_name = project_name
        self.model = None

        # Load best hyperparameters if directory and project_name are provided
        if self.directory and self.project_name and X_train is not None and y_train is not None:
            self.best_hps = load_best_hyperparameters(self.directory, self.project_name)

    def build_model(self):
        """
        Builds the model using the stored best hyperparameters.

        Returns:
        - model (tf.keras.Model): Compiled Keras model.
        """
        return build_model(hp=self.best_hps, input_shape=self.input_shape)

    def fit(self, X, y):
        """
        Fits the model to the data.

        Parameters:
        - X (np.ndarray): Training data.
        - y (np.ndarray): Training labels.

        Returns:
        - self: Fitted classifier.
        """
        self.model = self.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        self.model.fit(X, y, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, 
                       callbacks=[early_stopping, reduce_lr], class_weight=class_weights_dict, verbose=self.verbose)
        return self
    
    def predict(self, X):
        """
        Predicts class labels for samples in X.

        Parameters:
        - X (np.ndarray): Test data.

        Returns:
        - np.ndarray: Predicted class labels.
        """
        preds = self.model.predict(X)
        return (preds > 0.5).astype("int32")

    def predict_proba(self, X):
        """
        Predicts class probabilities for samples in X.

        Parameters:
        - X (np.ndarray): Test data.

        Returns:
        - np.ndarray: Predicted class probabilities.
        """
        preds = self.model.predict(X)
        return np.hstack([1 - preds, preds])


def train_and_evaluate_neural_network(X_train, y_train, X_test, y_test, best_hps=None):
    """
    Trains and evaluates the neural network model.

    Parameters:
    - X_train (np.ndarray): Training data.
    - y_train (np.ndarray): Training labels.
    - X_test (np.ndarray): Test data.
    - y_test (np.ndarray): Test labels.
    - best_hps (HyperParameters): Best hyperparameters from tuning. If None, defaults will be used.

    Returns:
    - model (tf.keras.Model): Trained Keras model.
    - history (History): Keras history object containing training history.
    """
    model = build_model(hp=best_hps, input_shape=X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=150, batch_size=1054,
                        callbacks=[early_stopping, reduce_lr],
                        class_weight=class_weights_dict, verbose=1)
    
    return model, history
