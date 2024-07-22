import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
                                                                            
                                                      
                                                 
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        LSTM(64),
        LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(32),
        LeakyReLU(alpha=0.1),
        Dropout(0.5),
        Dense(2, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy', sensitivity, specificity])
    return model
