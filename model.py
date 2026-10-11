import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

# Contrast Autoregressive LSTMs vs GANs for music generation:
# - LSTMs (Autoregressive): Model sequential dependencies naturally using discrete probability distributions.
#   They are highly stable during training, optimize direct maximum likelihood, and handle discrete tokens cleanly.
# - GANs (Generative Adversarial Networks): Suffer from discrete token gradient backpropagation challenges,
#   often requiring RL policy gradients (like SeqGAN). GANs are harder to stabilize but can capture high-level global patterns.

def build_lstm_model(vocab_size, embedding_dim=128, rnn_units=256):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name='embedding_layer'),
        LSTM(units=rnn_units, return_sequences=True, name='lstm_layer_1'),
        Dropout(0.3, name='dropout_1'),
        LSTM(units=rnn_units, return_sequences=False, name='lstm_layer_2'),
        Dropout(0.3, name='dropout_2'),
        Dense(units=vocab_size, activation='softmax', name='dense_softmax_output')
    ])
    return model