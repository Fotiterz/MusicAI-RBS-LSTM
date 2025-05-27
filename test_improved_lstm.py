import os
import pickle
import numpy as np
import tensorflow as tf
from music21 import stream, note, chord, instrument, tempo, key
from lstm_model import generate_music
import matplotlib.pyplot as plt

def test_lstm_model(output_length=100, key_signature='C', temperature=1.0):
    """
    Test the improved LSTM model with the existing trained weights.
    
    Parameters:
    - output_length: Number of notes to generate
    - key_signature: Key signature for music generation
    - temperature: Sampling temperature (higher = more random)
    """
    print(f"Testing improved LSTM model with key: {key_signature}, temperature: {temperature}")
    
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
    
    # Generate music with our improved generation function
    generated_sequence = generate_music(
        model, 
        seed_sequence, 
        length=output_length,
        vocab_size=len(pitchnames),
        temperature=temperature,
        key_signature=key_signature,
        pitchnames=pitchnames
    )
    
    print(f"Generated sequence of length {len(generated_sequence)}")
    
    # Convert the generated sequence to a music21 stream
    output_stream = sequence_to_stream(generated_sequence, pitchnames)
    
    # Set key signature and tempo
    k = key.Key(key_signature)
    output_stream.insert(0, k)
    
    t = tempo.MetronomeMark(number=120)
    output_stream.insert(0, t)
    
    # Add instrument
    piano = instrument.Piano()
    output_stream.insert(0, piano)
    
    # Save to MIDI file
    output_filename = f"improved_lstm_output_{key_signature}_{temperature}.mid"
    output_stream.write('midi', fp=output_filename)
    print(f"Saved output to {output_filename}")
    
    # Visualize the pitch distribution
    visualize_pitch_distribution(generated_sequence, pitchnames, key_signature, temperature)
    
    return output_filename

def sequence_to_stream(sequence, pitchnames):
    """Convert a sequence of note indices to a music21 stream."""
    output_stream = stream.Stream()
    
    # Pattern for creating a chord: every 4th note
    chord_pattern = 4
    
    # Keep track of previous notes for potential chord building
    prev_notes = []
    
    for i, idx in enumerate(sequence):
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
                
                # Store for potential chord building
                prev_notes.append(n)
                if len(prev_notes) > 3:
                    prev_notes.pop(0)
                
                # Every chord_pattern notes, try to build a chord from recent notes
                if i % chord_pattern == 0 and i > 0 and len(prev_notes) >= 3:
                    # Only create chord if we're not already in a chord pattern
                    c = chord.Chord(prev_notes)
                    c.quarterLength = 1.0
                    output_stream.append(c)
                else:
                    output_stream.append(n)
            except Exception as e:
                print(f"Error creating note from {pitch_name}: {e}")
                # Fallback to rest
                r = note.Rest()
                r.quarterLength = 0.5
                output_stream.append(r)
    
    return output_stream

def visualize_pitch_distribution(sequence, pitchnames, key_sig, temperature):
    """Visualize the distribution of pitches in the generated sequence."""
    # Count occurrences of each pitch
    pitch_counts = {}
    for idx in sequence:
        if idx < len(pitchnames):
            pitch_name = pitchnames[idx]
            # Extract base pitch name without octave for cleaner visualization
            base_pitch = pitch_name.split('.')[0]  # Take first part if it's a chord
            try:
                # Try to create a pitch object to get the name
                n = note.Note(base_pitch)
                base_name = n.pitch.name
            except:
                # If parsing fails, use the original
                base_name = base_pitch
                
            if base_name in pitch_counts:
                pitch_counts[base_name] += 1
            else:
                pitch_counts[base_name] = 1
    
    # Sort by pitch name
    sorted_pitches = sorted(pitch_counts.items())
    pitch_names = [p[0] for p in sorted_pitches]
    counts = [p[1] for p in sorted_pitches]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(pitch_names, counts)
    plt.title(f'Pitch Distribution (Key: {key_sig}, Temp: {temperature})')
    plt.xlabel('Pitch')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'pitch_distribution_{key_sig}_{temperature}.png')
    print(f"Saved pitch distribution visualization to pitch_distribution_{key_sig}_{temperature}.png")

if __name__ == "__main__":
    # Test with different keys and temperatures
    test_lstm_model(output_length=200, key_signature='C', temperature=1.0)
    test_lstm_model(output_length=200, key_signature='G', temperature=1.0)
    test_lstm_model(output_length=200, key_signature='C', temperature=0.7)  # More predictable