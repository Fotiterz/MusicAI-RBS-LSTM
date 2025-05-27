from flask import Flask, request, send_file, render_template, jsonify, send_from_directory
import os
import uuid
import atexit
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
import tensorflow as tf
from rule_based import generate_rule_based_music
from utils import save_stream_to_midi, convert_midi_to_mp3
from lstm_model import build_lstm_model, generate_music
from music21 import stream, note, chord

app = Flask(__name__)
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sequence_to_stream(sequence, pitchnames, length=100):
    """
    Convert a sequence of note indices to a music21 stream using pitchnames vocabulary.
    """
    s = stream.Stream()
    
    # Handle case when pitchnames is None
    if pitchnames is None:
        print("Warning: No pitchnames provided for sequence_to_stream. Creating default notes.")
        # Create a simple chromatic scale as default notes
        default_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5']
        
        for idx in sequence[:length]:
            # Use modulo to map any index to our default notes
            note_idx = idx % len(default_notes)
            try:
                n = note.Note(default_notes[note_idx], quarterLength=0.25)
                s.append(n)
            except Exception as e:
                print(f"Error creating default note: {e}")
                n = note.Rest(quarterLength=0.25)
                s.append(n)
        return s
    
    # Normal processing with pitchnames
    for idx in sequence[:length]:
        if idx == 0:
            # Treat 0 as a rest
            n = note.Rest(quarterLength=0.25)
            s.append(n)
            continue
            
        # Handle index out of range
        if idx >= len(pitchnames):
            # Use modulo to get a valid index
            idx = idx % len(pitchnames)
            
        pitch_name = pitchnames[idx]
        
        try:
            if '.' in pitch_name:
                # It's a chord
                chord_notes = pitch_name.split('.')
                try:
                    # Try to interpret as integers (MIDI note numbers)
                    chord_notes = [int(n) for n in chord_notes]
                    c = chord.Chord(chord_notes, quarterLength=0.25)
                    n = c
                except ValueError:
                    # Not integers, treat as note names
                    try:
                        notes = [note.Note(n, quarterLength=0.25) for n in pitch_name.split('.')]
                        n = chord.Chord(notes, quarterLength=0.25)
                    except Exception as e:
                        print(f"Error creating chord from {pitch_name}: {e}")
                        n = note.Rest(quarterLength=0.25)
            else:
                # Fix pitch name format if octave is before pitch letter
                # Example error: '5' instead of 'C5'
                if pitch_name and (pitch_name[0].isdigit()):
                    # Attempt to reorder to pitch letter + octave
                    letters = [c for c in pitch_name if c.isalpha()]
                    digits = [c for c in pitch_name if c.isdigit()]
                    
                    if letters and digits:
                        # Reorder pitch name to letter(s) + digit(s)
                        letters_part = ''.join(letters)
                        digits_part = ''.join(digits)
                        corrected_pitch = letters_part + digits_part
                        
                        try:
                            n = note.Note(corrected_pitch, quarterLength=0.25)
                        except Exception as e:
                            print(f"Error creating note from corrected pitch {corrected_pitch}: {e}")
                            n = note.Rest(quarterLength=0.25)
                    else:
                        # If no letters or no digits, use rest
                        n = note.Rest(quarterLength=0.25)
                else:
                    # Check if pitch_name is a single digit
                    if pitch_name.isdigit():
                        n = note.Rest(quarterLength=0.25)
                    else:
                        try:
                            n = note.Note(pitch_name, quarterLength=0.25)
                        except Exception as e:
                            print(f"Error creating note from {pitch_name}: {e}")
                            n = note.Rest(quarterLength=0.25)
            
            s.append(n)
            
        except Exception as e:
            print(f"Unexpected error processing note {idx} ({pitch_name}): {e}")
            # Add a rest as fallback
            n = note.Rest(quarterLength=0.25)
            s.append(n)
            
    return s

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    print("Received /generate request with data:", request.json)
    data = request.json
    mode = data.get('mode', 'rule')
    tempo = int(data.get('tempo', 120))
    key_sig = data.get('key', 'C')
    length = int(data.get('length', 8))
    instruments = data.get('instruments', ['Piano'])
    if isinstance(instruments, str):
        instruments = [instr.strip() for instr in instruments.split(',') if instr.strip()]
    elif isinstance(instruments, list):
        instruments = [str(instr).strip() for instr in instruments if str(instr).strip()]
    else:
        instruments = ['Piano']
    
    if mode == 'rule':
        music_stream = generate_rule_based_music(tempo, key_sig, length, instruments)
    elif mode == 'lstm':
        # Load trained LSTM model if available, else build new
        import tensorflow as tf
        import pickle
        from music21 import key as m21key, pitch as m21pitch, stream as m21stream, instrument as m21instrument
        model_path = 'lstm_trained_model.h5'
        vocab_path = 'pitchnames.pkl'
        
        # Load the model
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded model with output shape: {model.output_shape}")
        else:
            # If model doesn't exist, we need to create a default one
            # First ensure we have pitchnames
            if os.path.exists(vocab_path):
                with open(vocab_path, 'rb') as f:
                    pitchnames = pickle.load(f)
                print(f"Loaded pitchnames length: {len(pitchnames)}")
            else:
                # Create a default set of pitchnames
                pitchnames = [f"C{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)] + \
                             [f"E{i}" for i in range(1, 6)] + [f"F{i}" for i in range(1, 6)] + \
                             [f"G{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 6)] + \
                             [f"B{i}" for i in range(1, 6)]
                print(f"Created default pitchnames, length: {len(pitchnames)}")
                # Save the default pitchnames
                with open(vocab_path, 'wb') as f:
                    pickle.dump(pitchnames, f)
                
            input_shape = (50, 1)  # Adjusted input shape for trained model
            output_dim = len(pitchnames)
            model = build_lstm_model(input_shape, output_dim)
            print(f"Created new model with output shape: {model.output.shape}")
        
        # Load or create pitchnames vocabulary
        pitchnames = None
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'rb') as f:
                    pitchnames = pickle.load(f)
                print(f"Loaded pitchnames length: {len(pitchnames)}")
            except Exception as e:
                print(f"Error loading pitchnames: {e}")
                # Create a default set as fallback
                pitchnames = [f"C{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)] + \
                             [f"E{i}" for i in range(1, 6)] + [f"F{i}" for i in range(1, 6)] + \
                             [f"G{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 6)] + \
                             [f"B{i}" for i in range(1, 6)]
                print(f"Created default pitchnames, length: {len(pitchnames)}")
        else:
            # Create a default set of pitchnames
            pitchnames = [f"C{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)] + \
                         [f"E{i}" for i in range(1, 6)] + [f"F{i}" for i in range(1, 6)] + \
                         [f"G{i}" for i in range(1, 6)] + [f"A{i}" for i in range(1, 6)] + \
                         [f"B{i}" for i in range(1, 6)]
            print(f"Created default pitchnames, length: {len(pitchnames)}")
            # Save the default pitchnames
            with open(vocab_path, 'wb') as f:
                pickle.dump(pitchnames, f)
        
        # Create a seed sequence with shape (50, 1) filled with zeros
        seed_sequence = np.zeros((50, 1))
        print(f"Seed sequence shape: {seed_sequence.shape}")
        
        # Generate music with our improved generation function
        vocab_size = len(pitchnames) if pitchnames else 3803
        generated_sequence = generate_music(
            model, 
            seed_sequence, 
            length=length*50, 
            vocab_size=vocab_size,
            temperature=1.0,
            key_signature=key_sig,
            pitchnames=pitchnames
        )

        def apply_music_theory_constraints(sequence, pitchnames, key_signature):
            """
            Adjust the generated sequence to fit the key signature scale.
            """
            # If pitchnames is None or empty, return the original sequence
            if not pitchnames:
                print("Warning: No pitchnames provided for music theory constraints. Returning original sequence.")
                return sequence
                
            k = m21key.Key(key_signature)
            scale_pitches = [p.nameWithOctave for p in k.pitches]
            adjusted_sequence = []
            
            for idx in sequence:
                if idx == 0:
                    # Rest note, keep as is
                    adjusted_sequence.append(idx)
                    continue
                    
                # Check if index is valid
                if idx >= len(pitchnames):
                    # Invalid index, use modulo to get a valid index
                    idx = idx % len(pitchnames)
                    
                pitch_name = pitchnames[idx]
                
                # Skip chords (dot-separated pitch names)
                if '.' in pitch_name:
                    adjusted_sequence.append(idx)
                    continue
                    
                # Check if pitch is in scale, if not, adjust to closest scale pitch
                try:
                    base_pitch = m21pitch.Pitch(pitch_name)
                except Exception as e:
                    # If pitch creation fails, keep original
                    print(f"Warning: Failed to create pitch from {pitch_name}: {e}")
                    adjusted_sequence.append(idx)
                    continue
                    
                if base_pitch.nameWithOctave in scale_pitches:
                    adjusted_sequence.append(idx)
                else:
                    try:
                        # Find closest pitch in scale by semitone distance
                        distances = [(abs(base_pitch.midi - m21pitch.Pitch(p).midi), p) for p in scale_pitches]
                        distances.sort(key=lambda x: x[0])
                        closest_pitch_name = distances[0][1]
                        
                        # Map closest pitch name back to index in pitchnames
                        if closest_pitch_name in pitchnames:
                            adjusted_idx = pitchnames.index(closest_pitch_name)
                            adjusted_sequence.append(adjusted_idx)
                        else:
                            # If not found, keep original
                            adjusted_sequence.append(idx)
                    except Exception as e:
                        print(f"Warning: Error adjusting pitch {pitch_name}: {e}")
                        adjusted_sequence.append(idx)
                        
            return adjusted_sequence

        adjusted_sequence = apply_music_theory_constraints(generated_sequence, pitchnames, key_sig)

        # Create a music21 Score with separate parts for each instrument
        music_stream = m21stream.Score()
        instr_map = {
            'Piano': m21instrument.Piano,
            'Violin': m21instrument.Violin,
            'Flute': m21instrument.Flute,
            'Guitar': m21instrument.Guitar,
            'Trumpet': m21instrument.Trumpet,
            'Clarinet': m21instrument.Clarinet,
            'Saxophone': m21instrument.Saxophone,
            'Cello': m21instrument.Violoncello,
        }
        if not instruments:
            instruments = ['Piano']
        for instr_name in instruments:
            part = m21stream.Part()
            instr_class = instr_map.get(instr_name, m21instrument.Piano)
            instr_obj = instr_class()
            part.insert(0, instr_obj)
            # Convert adjusted sequence to stream for this part
            part_stream = sequence_to_stream(adjusted_sequence, pitchnames, length=length*50)
            for elem in part_stream:
                part.append(elem)
            music_stream.append(part)
    # Apply tempo, key signature, and instruments to the music stream
    # Set tempo
    from music21 import tempo, key, instrument
    # Set tempo marking
    # Ensure tempo is a number, convert if string
    try:
        tempo_value = float(tempo)
    except Exception:
        tempo_value = 120.0
    tempo_marking = tempo.MetronomeMark(number=tempo_value)
    music_stream.insert(0, tempo_marking)
    # Insert tempo marking into each part as well
    for part in music_stream.parts:
        part.insert(0, tempo_marking)
    # Set key signature
    try:
        key_signature = key.Key(key_sig)
        music_stream.insert(0, key_signature)
    except Exception:
        pass
    # Set instruments for all parts (if applicable)
    # Since the generated stream may be a single part, assign instrument to all notes
    instr_map = {
        'Piano': instrument.Piano(),
        'Violin': instrument.Violin(),
        'Flute': instrument.Flute(),
        'Guitar': instrument.Guitar(),
        'Trumpet': instrument.Trumpet(),
        'Clarinet': instrument.Clarinet(),
        'Saxophone': instrument.Saxophone(),
        'Cello': instrument.Violoncello(),
    # 'Drums': instrument.DrumKit()  # Removed because music21.instrument has no DrumKit attribute
    }
    selected_instruments = []
    for instr_name in instruments:
        if instr_name in instr_map:
            selected_instruments.append(instr_map[instr_name])
    if selected_instruments:
        # Insert instruments at the beginning of the stream
        for instr_obj in selected_instruments:
            music_stream.insert(0, instr_obj)
    else:
        return jsonify({'error': 'Invalid mode'}), 400
    
    midi_filename = f"{uuid.uuid4()}.mid"
    midi_path = os.path.join(OUTPUT_DIR, midi_filename)
    save_stream_to_midi(music_stream, midi_path)
    
    mp3_filename = midi_filename.replace('.mid', '.mp3')
    mp3_path = os.path.join(OUTPUT_DIR, mp3_filename)
    try:
        convert_midi_to_mp3(midi_path, mp3_path)
    except Exception as e:
        import traceback
        print("Error during MIDI to MP3 conversion:", e)
        traceback.print_exc()
        return jsonify({'error': 'MIDI to MP3 conversion failed: ' + str(e)}), 500
    
    response = {
        'mp3_url': f"/download/{mp3_filename}",
        'midi_url': f"/download/{midi_filename}"
    }
    print("Sending response:", response)
    return jsonify(response)

@app.route('/download/<filename>')
def download_file(filename):
    if filename.lower().endswith('.mp3'):
        mimetype = 'audio/mpeg'
    elif filename.lower().endswith('.mid') or filename.lower().endswith('.midi'):
        mimetype = 'audio/midi'
    else:
        mimetype = 'application/octet-stream'
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True, mimetype=mimetype)

def cleanup_output_files():
    print("Cleaning up output files...")
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

atexit.register(cleanup_output_files)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
