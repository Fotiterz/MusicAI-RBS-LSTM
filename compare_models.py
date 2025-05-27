import os
import pickle
import numpy as np
import tensorflow as tf
from music21 import stream, note, chord, instrument, tempo, key
import matplotlib.pyplot as plt

# Import both the original and improved generation functions
# We'll create a copy of the original function for comparison
def original_generate_music(model, seed_sequence, length=100, vocab_size=3803, temperature=1.0, key_signature=None, pitchnames=None):
    """
    Original music generation function (simplified for comparison).
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
                try:
                    base_pitch = m21pitch.Pitch(pitch_name)
                    if base_pitch.name in scale_pitches:
                        mask[i] = 1
                except:
                    continue
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

def improved_generate_music(model, seed_sequence, length=100, vocab_size=3803, temperature=1.0, key_signature=None, pitchnames=None):
    """
    Import the improved generation function from lstm_model.py
    """
    from lstm_model import generate_music
    return generate_music(model, seed_sequence, length, vocab_size, temperature, key_signature, pitchnames)

def sequence_to_stream(sequence, pitchnames):
    """Convert a sequence of note indices to a music21 stream."""
    output_stream = stream.Stream()
    
    for idx in sequence:
        if idx >= len(pitchnames):
            continue
            
        pitch_name = pitchnames[idx]
        
        # Handle chord notation (contains dots)
        if '.' in pitch_name:
            try:
                # Try to interpret as chord
                chord_notes = []
                for p in pitch_name.split('.'):
                    try:
                        n = note.Note(int(p))
                        chord_notes.append(n)
                    except:
                        # If conversion fails, try as pitch name
                        try:
                            n = note.Note(p)
                            chord_notes.append(n)
                        except:
                            continue
                
                if chord_notes:
                    c = chord.Chord(chord_notes)
                    c.quarterLength = 0.5
                    output_stream.append(c)
                else:
                    # Fallback to rest
                    r = note.Rest()
                    r.quarterLength = 0.5
                    output_stream.append(r)
            except Exception as e:
                print(f"Error creating chord from {pitch_name}: {e}")
                # Fallback to rest
                r = note.Rest()
                r.quarterLength = 0.5
                output_stream.append(r)
        else:
            # Single note
            try:
                n = note.Note(pitch_name)
                n.quarterLength = 0.5
                output_stream.append(n)
            except Exception as e:
                print(f"Error creating note from {pitch_name}: {e}")
                # Fallback to rest
                r = note.Rest()
                r.quarterLength = 0.5
                output_stream.append(r)
    
    return output_stream

def compare_models(output_length=100, key_signature='C', temperature=1.0):
    """
    Compare the original and improved LSTM models.
    
    Parameters:
    - output_length: Number of notes to generate
    - key_signature: Key signature for music generation
    - temperature: Sampling temperature (higher = more random)
    """
    print(f"Comparing original and improved LSTM models with key: {key_signature}, temperature: {temperature}")
    
    # Load the trained model
    model_path = 'lstm_trained_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model with output shape: {model.output_shape}")
    
    # Load pitchnames vocabulary
    vocab_path = 'pitchnames.pkl'
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(vocab_path, 'rb') as f:
        pitchnames = pickle.load(f)
    
    print(f"Loaded vocabulary with {len(pitchnames)} unique pitches")
    
    # Create a seed sequence (all zeros)
    sequence_length = 50  # Default sequence length
    seed_sequence = np.zeros((sequence_length, 1))
    
    # Generate music with original generation function
    print("Generating music with original algorithm...")
    original_sequence = original_generate_music(
        model, 
        seed_sequence, 
        length=output_length,
        vocab_size=len(pitchnames),
        temperature=temperature,
        key_signature=key_signature,
        pitchnames=pitchnames
    )
    
    # Generate music with improved generation function
    print("Generating music with improved algorithm...")
    improved_sequence = improved_generate_music(
        model, 
        seed_sequence, 
        length=output_length,
        vocab_size=len(pitchnames),
        temperature=temperature,
        key_signature=key_signature,
        pitchnames=pitchnames
    )
    
    # Convert sequences to music21 streams
    original_stream = sequence_to_stream(original_sequence, pitchnames)
    improved_stream = sequence_to_stream(improved_sequence, pitchnames)
    
    # Set key signature and tempo
    k = key.Key(key_signature)
    t = tempo.MetronomeMark(number=120)
    piano = instrument.Piano()
    
    original_stream.insert(0, k)
    original_stream.insert(0, t)
    original_stream.insert(0, piano)
    
    improved_stream.insert(0, k)
    improved_stream.insert(0, t)
    improved_stream.insert(0, piano)
    
    # Save to MIDI files
    original_filename = f"original_lstm_output_{key_signature}_{temperature}.mid"
    improved_filename = f"improved_lstm_output_{key_signature}_{temperature}_comparison.mid"
    
    original_stream.write('midi', fp=original_filename)
    improved_stream.write('midi', fp=improved_filename)
    
    print(f"Saved original output to {original_filename}")
    print(f"Saved improved output to {improved_filename}")
    
    # Compare pitch distributions
    compare_pitch_distributions(original_sequence, improved_sequence, pitchnames, key_signature, temperature)
    
    return original_filename, improved_filename

def compare_pitch_distributions(original_sequence, improved_sequence, pitchnames, key_sig, temperature):
    """Compare the pitch distributions between original and improved models."""
    # Count occurrences of each pitch in original sequence
    original_counts = {}
    for idx in original_sequence:
        if idx < len(pitchnames):
            pitch_name = pitchnames[idx]
            # Extract base pitch name without octave
            base_pitch = pitch_name.split('.')[0]
            try:
                n = note.Note(base_pitch)
                base_name = n.pitch.name
            except:
                base_name = base_pitch
                
            if base_name in original_counts:
                original_counts[base_name] += 1
            else:
                original_counts[base_name] = 1
    
    # Count occurrences of each pitch in improved sequence
    improved_counts = {}
    for idx in improved_sequence:
        if idx < len(pitchnames):
            pitch_name = pitchnames[idx]
            # Extract base pitch name without octave
            base_pitch = pitch_name.split('.')[0]
            try:
                n = note.Note(base_pitch)
                base_name = n.pitch.name
            except:
                base_name = base_pitch
                
            if base_name in improved_counts:
                improved_counts[base_name] += 1
            else:
                improved_counts[base_name] = 1
    
    # Get all unique pitch names
    all_pitches = sorted(set(list(original_counts.keys()) + list(improved_counts.keys())))
    
    # Create arrays for plotting
    original_values = [original_counts.get(p, 0) for p in all_pitches]
    improved_values = [improved_counts.get(p, 0) for p in all_pitches]
    
    # Create the comparison plot
    plt.figure(figsize=(14, 7))
    
    x = np.arange(len(all_pitches))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original')
    plt.bar(x + width/2, improved_values, width, label='Improved')
    
    plt.title(f'Pitch Distribution Comparison (Key: {key_sig}, Temp: {temperature})')
    plt.xlabel('Pitch')
    plt.ylabel('Count')
    plt.xticks(x, all_pitches, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'pitch_comparison_{key_sig}_{temperature}.png')
    print(f"Saved pitch distribution comparison to pitch_comparison_{key_sig}_{temperature}.png")

if __name__ == "__main__":
    # Compare models with different keys and temperatures
    compare_models(output_length=200, key_signature='C', temperature=1.0)
    compare_models(output_length=200, key_signature='G', temperature=1.0)
    compare_models(output_length=200, key_signature='C', temperature=0.7)  # More predictable