import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LeakyReLU

def create_simple_cnn_model(input_shape):
    model = Sequential([
        Conv1D(32, kernel_size=3, padding='same', input_shape=input_shape), # kernel 3: capturing local patterns and relationships between  features
        LeakyReLU(alpha=0.1),                                               # padding same: maintains the same output length as input, preserving dimensionality throughout the networks
        MaxPooling1D(pool_size=2),                                          # Pooling Size 2: reduces dimensionality by half, decreasing computational load while retaining essential features.
        Dropout(0.3),
        Conv1D(64, kernel_size=3, padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(32),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(2, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])
    return model
