import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def build_lstm_model(input_shape, output_dim):
    """
    Build a simple LSTM model for polyphonic music generation.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def build_lstm_model(input_shape, output_dim):
    """
    Build a simple LSTM model for polyphonic music generation.
    """
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(128))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def sample_with_temperature(preds, temperature=1.0):
    """
    Sample an index from a probability array reweighted by temperature.
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_music(model, seed_sequence, length=100, vocab_size=2831, temperature=1.0, key_signature=None, pitchnames=None):
    """
    Generate a sequence of music notes from the LSTM model with guided generation.
    
    Parameters:
    - model: Trained LSTM model
    - seed_sequence: Initial input sequence (numpy array)
    - length: Number of notes to generate
    - vocab_size: Size of the vocabulary for normalization
    - temperature: Sampling temperature for diversity
    - key_signature: Key signature string for music theory guidance
    - pitchnames: List of pitch names corresponding to vocab indices
    
    Returns:
    - generated sequence as a list of note indices
    """
    from music21 import key as m21key, pitch as m21pitch
    
    generated = []
    sequence = seed_sequence.copy()
    recent_notes = []
    k = m21key.Key(key_signature) if key_signature else None
    scale_pitches = [p.name for p in k.pitches] if k else None
    
    for _ in range(length):
        prediction = model.predict(sequence[np.newaxis, :, :], verbose=0)[0]
        
        # Apply music theory mask to prediction probabilities
        if scale_pitches and pitchnames:
            mask = np.zeros_like(prediction)
            for i, pitch_name in enumerate(pitchnames):
                base_pitch = m21pitch.Pitch(pitch_name)
                if base_pitch.name in scale_pitches:
                    mask[i] = 1
            prediction = prediction * mask
            if np.sum(prediction) == 0:
                prediction = np.ones_like(prediction)
            prediction = prediction / np.sum(prediction)
        
        # Apply repetition penalty
        penalty_strength = 0.7
        for note_idx in recent_notes:
            prediction[note_idx] *= (1 - penalty_strength)
        prediction = prediction / np.sum(prediction)
        
        # Sample next note index with temperature
        index = sample_with_temperature(prediction, temperature)
        generated.append(index)
        
        # Update recent notes list
        recent_notes.append(index)
        if len(recent_notes) > 5:
            recent_notes.pop(0)
        
        # Normalize the predicted index for next input
        normalized_index = index / float(vocab_size)
        # Append normalized index to sequence, remove first element
        sequence = np.append(sequence[1:], [[normalized_index]], axis=0)
    return generated

def train_lstm_model(model, X_train, y_train, epochs=20, batch_size=64):
    """
    Train the LSTM model on the given training data.
    
    Parameters:
    - model: LSTM model to train
    - X_train: Input sequences (numpy array)
    - y_train: Target outputs (numpy array)
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    
    Returns:
    - Trained model
    """
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

if __name__ == "__main__":
    # Example usage: build model with dummy input shape and output dim
    model = build_lstm_model((50, 88), 88)
    model.summary()
