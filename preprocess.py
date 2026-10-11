import numpy as np
import json
import os
from sklearn.model_selection import train_test_split

def prepare_sequences(all_pieces_tokens, vocab_path, sequence_length=100, test_size=0.2):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    unk_idx = token_to_idx.get('UNK', len(token_to_idx))
    
    X, y = [], []
    for piece in all_pieces_tokens:
        indices = [token_to_idx.get(tok, unk_idx) for tok in piece]
        if len(indices) > sequence_length:
            for i in range(len(indices) - sequence_length):
                X.append(indices[i : i + sequence_length])
                y.append(indices[i + sequence_length])
                
    X = np.array(X, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    
    return train_test_split(X, y, test_size=test_size, random_state=42)