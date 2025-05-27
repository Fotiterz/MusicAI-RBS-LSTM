"""
This file contains a fix for the main.py file to properly handle LSTM model generation.
Copy this code to your main.py file to fix the error.
"""

# Import the necessary libraries for LSTM model
import tensorflow as tf
import numpy as np
import pickle
import os

# Load the LSTM model and pitchnames
def load_lstm_model():
    model_path = 'lstm_trained_model.h5'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model with output shape: {model.output_shape}")
    
    # Load pitchnames vocabulary
    vocab_path = 'pitchnames.pkl'
    pitchnames = None
    if os.path.exists(vocab_path):
        try:
            with open(vocab_path, 'rb') as f:
                pitchnames = pickle.load(f)
            print(f"Loaded pitchnames length: {len(pitchnames)}")
        except Exception as e:
            print(f"Error loading pitchnames: {e}")
    else:
        print("Warning: pitchnames.pkl not found")
    
    return model, pitchnames

# In your /generate route handler, replace the LSTM generation code with:
"""
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    print(f"Received /generate request with data: {data}")
    
    mode = data.get('mode', 'rule')
    tempo = int(data.get('tempo', 120))
    key = data.get('key', 'C')
    length = int(data.get('length', 8))
    instruments = data.get('instruments', ['Piano'])
    
    if mode == 'lstm':
        try:
            # Load the model and pitchnames
            model, pitchnames = load_lstm_model()
            
            # Create a seed sequence
            sequence_length = 50  # Default sequence length
            seed_sequence = np.zeros((sequence_length, 1))
            print(f"Seed sequence shape: {seed_sequence.shape}")
            
            # Import the generate_music function
            from lstm_model import generate_music
            
            # Generate music with our improved generation function
            generated_sequence = generate_music(
                model, 
                seed_sequence, 
                length=length*50, 
                vocab_size=len(pitchnames) if pitchnames else 3803,
                temperature=1.0,
                key_signature=key,
                pitchnames=pitchnames
            )
            
            # Convert the generated sequence to a music21 stream
            output_stream = sequence_to_stream(generated_sequence, pitchnames)
            
            # Set key signature and tempo
            output_stream = add_metadata_to_stream(output_stream, key, tempo, instruments)
            
            # Save to MIDI file
            output_filename = f"lstm_output_{key}_{tempo}.mid"
            output_stream.write('midi', fp=output_filename)
            
            # Convert to MP3 if FluidSynth is available
            mp3_path = convert_midi_to_mp3(output_filename)
            
            return jsonify({
                'success': True,
                'midi_file': output_filename,
                'mp3_file': mp3_path
            })
            
        except Exception as e:
            print(f"Error in LSTM generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    # Rest of your code for rule-based generation...
"""

# Add this helper function to convert sequence to stream
def sequence_to_stream(sequence, pitchnames):
    """Convert a sequence of note indices to a music21 stream."""
    from music21 import stream, note, chord, instrument
    
    output_stream = stream.Stream()
    
    if pitchnames is None:
        # Create a default set of pitchnames if none provided
        print("Warning: No pitchnames provided for sequence_to_stream")
        pitchnames = [f"C{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)] + \
                     [f"E{i}" for i in range(1, 6)] + [f"F{i}" for i in range(1, 6)] + \
                     [f"G{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 6)] + \
                     [f"B{i}" for i in range(1, 6)]
    
    for idx in sequence:
        if idx >= len(pitchnames):
            idx = idx % len(pitchnames)  # Ensure index is within range
            
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

# Add this helper function to add metadata to stream
def add_metadata_to_stream(stream_obj, key_signature, tempo_value, instruments):
    """Add key signature, tempo, and instruments to a music21 stream."""
    from music21 import key, tempo, instrument
    
    # Set key signature
    k = key.Key(key_signature)
    stream_obj.insert(0, k)
    
    # Set tempo
    t = tempo.MetronomeMark(number=tempo_value)
    stream_obj.insert(0, t)
    
    # Add instrument
    if 'Piano' in instruments:
        piano = instrument.Piano()
        stream_obj.insert(0, piano)
    elif 'Violin' in instruments:
        violin = instrument.Violin()
        stream_obj.insert(0, violin)
    elif 'Guitar' in instruments:
        guitar = instrument.Guitar()
        stream_obj.insert(0, guitar)
    else:
        # Default to piano
        piano = instrument.Piano()
        stream_obj.insert(0, piano)
    
    return stream_obj