import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from model import build_lstm_model
from preprocess import prepare_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

def lr_schedule(epoch):
    lr = 0.001
    if epoch > 10:
        lr *= 0.1
    return lr

def train_pipeline(all_pieces_tokens, vocab_path, epochs=15, batch_size=32):
    X_train, X_val, y_train, y_val = prepare_sequences(all_pieces_tokens, vocab_path)
    vocab_size = int(np.max(X_train) + 2)
    
    model = build_lstm_model(vocab_size=vocab_size)
    model.build(input_shape=(None, 100))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(os.path.join('models', 'best_bach_lstm.keras'), monitor='val_loss', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stop, lr_scheduler]
    )
    return history, model