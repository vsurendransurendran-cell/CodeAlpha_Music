import numpy as np
import tensorflow as tf

def sample_with_temperature(predictions, temperature=1.0):
    """
    Samples an index from a probability array using temperature scaling.
    """
    predictions = np.asarray(predictions).astype('float64')
    if temperature <= 0.0:
        return np.argmax(predictions)
    
    # Scale predictions (logits or probabilities) using log
    # To avoid log(0), we add a tiny epsilon
    preds = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sequence(model, seed_sequence, length, token_to_idx, idx_to_token, temperature=1.0):
    """
    Generates a sequence of musical tokens starting from a seed sequence.
    """
    unk_idx = token_to_idx.get('UNK', len(token_to_idx))
    input_seq = [token_to_idx.get(tok, unk_idx) for tok in seed_sequence]
    generated_tokens = []
    
    for _ in range(length):
        # Pad or slice input sequence to match sequence length (100)
        if len(input_seq) > 100:
            model_input = input_seq[-100:]
        else:
            model_input = [0] * (100 - len(input_seq)) + input_seq
            
        # Predict probabilities
        preds = model.predict(np.array([model_input]), verbose=0)[0]
        next_idx = sample_with_temperature(preds, temperature)
        
        # Map back to token
        next_tok = idx_to_token.get(str(next_idx), 'UNK')
        generated_tokens.append(next_tok)
        input_seq.append(next_idx)
        
    return generated_tokens