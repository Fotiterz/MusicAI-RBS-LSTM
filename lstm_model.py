import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def build_lstm_model(input_shape, output_dim):
    """
    Build an improved LSTM model for polyphonic music generation.
    
    Improvements:
    - Bidirectional LSTM layers for better context understanding
    - Deeper architecture with more capacity
    - Batch normalization for better training stability
    - Dropout for regularization
    - Configurable learning rate with Adam optimizer
    """
    model = Sequential()
    
    # First bidirectional LSTM layer
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Second bidirectional LSTM layer
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Third LSTM layer
    model.add(LSTM(256))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(output_dim, activation='softmax'))
    
    # Use Adam optimizer with a configurable learning rate
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
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

def generate_music(model, seed_sequence, length=100, vocab_size=3803, temperature=1.0, key_signature=None, pitchnames=None):
    """
    Generate a sequence of music notes from the LSTM model with enhanced guided generation.
    
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
    from music21 import key as m21key, pitch as m21pitch, interval, chord
    import pickle
    import os
    
    # Handle missing pitchnames by loading from file if available
    if pitchnames is None:
        try:
            if os.path.exists('pitchnames.pkl'):
                with open('pitchnames.pkl', 'rb') as f:
                    pitchnames = pickle.load(f)
                print(f"Loaded pitchnames from file, length: {len(pitchnames)}")
            else:
                # Create a default set of pitchnames if file doesn't exist
                print("Warning: No pitchnames provided and no pitchnames.pkl file found.")
                # Create a simple chromatic scale as fallback
                pitchnames = [f"C{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)] + \
                             [f"E{i}" for i in range(1, 6)] + [f"F{i}" for i in range(1, 6)] + \
                             [f"G{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 6)] + \
                             [f"B{i}" for i in range(1, 6)]
                print(f"Created default pitchnames, length: {len(pitchnames)}")
        except Exception as e:
            print(f"Error loading pitchnames: {e}")
            # Create a minimal set as absolute fallback
            pitchnames = [f"C{i}" for i in range(1, 6)]
            print(f"Created minimal fallback pitchnames, length: {len(pitchnames)}")
    
    generated = []
    sequence = seed_sequence.copy()
    recent_notes = []
    recent_intervals = []
    
    # Initialize key and scale information
    k = m21key.Key(key_signature) if key_signature else m21key.Key('C')
    scale_pitches = [p.name for p in k.pitches]
    scale_pitches_with_octave = []
    
    # Generate scale pitches across multiple octaves for better matching
    for octave in range(1, 8):
        for pitch_name in scale_pitches:
            scale_pitches_with_octave.append(f"{pitch_name}{octave}")
    
    # Define common chord progressions in the key
    chord_progressions = {
        'C': ['C', 'G', 'Am', 'F'],  # I-V-vi-IV
        'G': ['G', 'D', 'Em', 'C'],  # I-V-vi-IV
        'D': ['D', 'A', 'Bm', 'G'],  # I-V-vi-IV
        'A': ['A', 'E', 'F#m', 'D'],  # I-V-vi-IV
        'E': ['E', 'B', 'C#m', 'A'],  # I-V-vi-IV
        'F': ['F', 'C', 'Dm', 'Bb'],  # I-V-vi-IV
        'Bb': ['Bb', 'F', 'Gm', 'Eb'],  # I-V-vi-IV
    }
    
    # Get chord progression for the current key if available
    current_progression = chord_progressions.get(k.tonic.name, chord_progressions['C'])
    chord_notes = []
    
    # Convert chord names to notes
    for chord_name in current_progression:
        try:
            c = chord.Chord(chord_name)
            chord_notes.extend([p.name for p in c.pitches])
        except:
            pass
    
    # Combine scale and chord notes for better guidance
    preferred_notes = list(set(scale_pitches + chord_notes))
    
    # Track phrase structure
    phrase_length = 8  # Common musical phrase length
    current_phrase_pos = 0
    
    # Melodic contour guidance
    # 0: neutral, 1: ascending, -1: descending
    melodic_direction = 0
    
    for i in range(length):
        prediction = model.predict(sequence[np.newaxis, :, :], verbose=0)[0]
        
        # Apply music theory mask to prediction probabilities
        if pitchnames and len(pitchnames) > 0:
            # Create a weighted mask that favors notes in the scale/chord
            mask = np.ones_like(prediction) * 0.1  # Base weight for all notes
            
            # Higher weights for notes in the scale
            for idx, pitch_name in enumerate(pitchnames):
                if idx >= len(prediction):
                    continue  # Skip if index is out of bounds
                    
                try:
                    # Handle chord notation (e.g., "0.1.2")
                    if '.' in pitch_name:
                        # For chords, check if any note is in scale
                        chord_notes = pitch_name.split('.')
                        in_scale = False
                        for note in chord_notes:
                            if note.isdigit():
                                continue
                            if any(note in sp for sp in scale_pitches_with_octave):
                                in_scale = True
                                break
                        if in_scale:
                            mask[idx] = 1.0
                    else:
                        # For single notes, extract the pitch name without octave
                        base_pitch = None
                        try:
                            base_pitch = m21pitch.Pitch(pitch_name)
                            # Check if the note is in our scale
                            if base_pitch.name in preferred_notes:
                                mask[idx] = 1.0
                            # Give higher weight to chord tones
                            if base_pitch.name in chord_notes:
                                mask[idx] = 1.5
                        except:
                            # If pitch parsing fails, try to extract letter name
                            if any(letter in pitch_name for letter in ['C', 'D', 'E', 'F', 'G', 'A', 'B']):
                                for note in preferred_notes:
                                    if note in pitch_name:
                                        mask[idx] = 0.8
                                        break
                except Exception as e:
                    # If any error occurs, keep default weight
                    continue
            
            # Apply the mask
            prediction = prediction * mask
            
            # Ensure we have valid probabilities
            if np.sum(prediction) <= 0:
                prediction = np.ones_like(prediction)
            prediction = prediction / np.sum(prediction)
        
        # Apply repetition penalty - avoid repeating the same note too often
        if recent_notes:
            penalty_strength = 0.8
            for note_idx in recent_notes[-3:]:  # Focus on most recent notes
                if note_idx < len(prediction):
                    prediction[note_idx] *= (1 - penalty_strength)
                
            # Encourage melodic direction based on phrase position
            if current_phrase_pos < phrase_length / 2:
                # First half of phrase - tend to ascend
                melodic_direction = 1
            else:
                # Second half of phrase - tend to descend
                melodic_direction = -1
                
            # If we have previous notes, try to guide the melodic contour
            if len(recent_notes) >= 2 and pitchnames and len(pitchnames) > 0:
                last_note_idx = recent_notes[-1]
                if last_note_idx < len(pitchnames):
                    last_pitch_name = pitchnames[last_note_idx]
                    
                    # Try to parse the last pitch
                    try:
                        last_pitch = m21pitch.Pitch(last_pitch_name)
                        
                        # Boost probabilities for notes in the desired direction
                        for idx, pitch_name in enumerate(pitchnames):
                            if idx >= len(prediction):
                                continue  # Skip if index is out of bounds
                                
                            try:
                                current_pitch = m21pitch.Pitch(pitch_name)
                                # Calculate interval direction
                                int_direction = 1 if current_pitch.midi > last_pitch.midi else -1
                                
                                # If this note continues our desired melodic direction
                                if int_direction == melodic_direction:
                                    # Boost its probability
                                    prediction[idx] *= 1.2
                            except:
                                continue
                    except:
                        pass
            
            # Renormalize after all adjustments
            prediction = prediction / np.sum(prediction)
        
        # Sample next note index with temperature
        temperature_adjusted = temperature
        # Use lower temperature at phrase boundaries for more predictable cadences
        if current_phrase_pos == 0 or current_phrase_pos == phrase_length - 1:
            temperature_adjusted = max(0.7, temperature - 0.3)
        
        index = sample_with_temperature(prediction, temperature_adjusted)
        
        # Ensure index is within valid range
        if index >= vocab_size:
            index = index % vocab_size
            
        generated.append(index)
        
        # Update recent notes list
        recent_notes.append(index)
        if len(recent_notes) > 8:  # Track more notes for better context
            recent_notes.pop(0)
        
        # Update phrase position
        current_phrase_pos = (current_phrase_pos + 1) % phrase_length
        
        # Normalize the predicted index for next input
        normalized_index = index / float(vocab_size)
        # Append normalized index to sequence, remove first element
        sequence = np.append(sequence[1:], [[normalized_index]], axis=0)
        
    return generated

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=64, validation_split=0.1):
    """
    Train the LSTM model on the given training data with improved training process.
    
    Parameters:
    - model: LSTM model to train
    - X_train: Input sequences (numpy array)
    - y_train: Target outputs (numpy array)
    - epochs: Maximum number of training epochs
    - batch_size: Batch size for training
    - validation_split: Fraction of training data to use for validation
    
    Returns:
    - Trained model
    """
    # Define callbacks for better training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train with validation split and callbacks
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

if __name__ == "__main__":
    # Example usage: build model with dummy input shape and output dim
    model = build_lstm_model((50, 88), 88)
    model.summary()
