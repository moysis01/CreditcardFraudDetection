import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def build_model():
    model = Sequential()
    model.add(Input(shape=(29, 1)))  # Explicitly add Input layer
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


def train_and_evaluate_neural_network(X_train, y_train, X_test, y_test, save_dir='plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model = build_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=200, batch_size=1000, callbacks=[early_stopping, reduce_lr],
                        class_weight=class_weights_dict)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'model_accuracy.png'))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'model_loss.png'))

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")

    # Predictions and threshold optimization
    y_test_pred = model.predict(X_test).ravel()
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)
    optimal_idx = np.argmax(precision * recall)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal Threshold: {optimal_threshold}')

    y_test_pred_classes = (y_test_pred > optimal_threshold).astype(int)
    y_test_true_classes = y_test.to_numpy().astype(int)

    print(classification_report(y_test_true_classes, y_test_pred_classes))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test_true_classes, y_test_pred_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

    return model
