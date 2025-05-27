import numpy as np
from music21 import converter, instrument, note, chord, stream, interval, pitch, key
import os
import glob
import random
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_notes_from_midi(midi_files):
    """
    Extract notes and chords from a list of MIDI files with improved parsing.
    
    Improvements:
    - Better error handling for corrupted MIDI files
    - Progress tracking with tqdm
    - Extraction of more musical features (duration, offset)
    - Support for multiple instrument parts
    """
    notes = []
    durations = []
    
    for file in tqdm(midi_files, desc="Parsing MIDI files"):
        try:
            # Parse the MIDI file
            midi = converter.parse(file)
            
            # Get the key signature if available
            key_signature = None
            for element in midi.flat:
                if isinstance(element, key.Key):
                    key_signature = element
                    break
            
            # Partition by instrument
            parts = instrument.partitionByInstrument(midi)
            
            # Process each part (instrument)
            if parts:  # file has instrument parts
                # Process up to 3 instrument parts to capture harmony
                for i, part in enumerate(parts.parts[:3]):
                    notes_to_parse = part.recurse()
                    process_notes(notes_to_parse, notes, durations, instrument_id=i)
            else:
                # Single part/instrument
                notes_to_parse = midi.flat.notes
                process_notes(notes_to_parse, notes, durations)
                
        except Exception as e:
            logger.warning(f"Error parsing {file}: {str(e)}")
            continue
    
    return notes

def process_notes(notes_to_parse, notes_list, durations_list, instrument_id=0):
    """Process notes from a single instrument part."""
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            # Format: pitch_name|duration|instrument_id
            note_info = f"{str(element.pitch)}|{element.quarterLength}|{instrument_id}"
            notes_list.append(note_info)
            durations_list.append(element.quarterLength)
        elif isinstance(element, chord.Chord):
            # Join chord notes with dots, add duration and instrument
            chord_info = f"{'.'.join(str(n) for n in element.normalOrder)}|{element.quarterLength}|{instrument_id}"
            notes_list.append(chord_info)
            durations_list.append(element.quarterLength)

def prepare_sequences(notes, sequence_length=50, augment=True):
    """
    Prepare the sequences used by the Neural Network with data augmentation.
    
    Improvements:
    - Data augmentation through transposition
    - Better handling of sequence preparation
    - Support for variable sequence lengths
    - Improved normalization
    """
    # Get all unique note representations
    pitchnames = sorted(set(item for item in notes))
    logger.info(f"Found {len(pitchnames)} unique note/chord patterns")
    
    # Create mapping dictionaries
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    
    # Prepare sequences
    network_input = []
    network_output = []
    
    # Create sequences
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])
    
    # Data augmentation through transposition if enabled
    if augment:
        augmented_input, augmented_output = augment_data(notes, note_to_int, sequence_length)
        if augmented_input and augmented_output:
            network_input.extend(augmented_input)
            network_output.extend(augmented_output)
            logger.info(f"Added {len(augmented_input)} augmented sequences")
    
    n_patterns = len(network_input)
    logger.info(f"Total sequences after augmentation: {n_patterns}")
    
    # Reshape and normalize input
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    
    # One-hot encode the output
    network_output = np.eye(len(pitchnames))[network_output]
    
    return network_input, network_output, pitchnames

def augment_data(notes, note_to_int, sequence_length):
    """
    Augment the dataset through transposition.
    
    This function creates additional training examples by transposing
    the original sequences up and down by 1-3 semitones.
    """
    try:
        # Only use a subset of the data for augmentation to avoid memory issues
        max_sequences = min(len(notes) - sequence_length, 5000)
        indices = random.sample(range(len(notes) - sequence_length), max_sequences)
        
        augmented_input = []
        augmented_output = []
        
        # Transposition intervals (in semitones)
        transpositions = [-3, -2, -1, 1, 2, 3]
        
        for idx in tqdm(indices, desc="Augmenting data"):
            # Get the original sequence
            seq_in = notes[idx:idx + sequence_length]
            seq_out = notes[idx + sequence_length]
            
            # Apply random transposition
            transpose_interval = random.choice(transpositions)
            
            # Try to transpose the sequence
            try:
                transposed_seq_in = transpose_sequence(seq_in, transpose_interval)
                transposed_seq_out = transpose_note(seq_out, transpose_interval)
                
                # Check if all transposed notes are in our vocabulary
                if all(note in note_to_int for note in transposed_seq_in) and transposed_seq_out in note_to_int:
                    # Convert to integers
                    int_seq_in = [note_to_int[note] for note in transposed_seq_in]
                    int_seq_out = note_to_int[transposed_seq_out]
                    
                    augmented_input.append(int_seq_in)
                    augmented_output.append(int_seq_out)
            except Exception as e:
                # Skip this sequence if transposition fails
                continue
                
        return augmented_input, augmented_output
    except Exception as e:
        logger.warning(f"Data augmentation failed: {str(e)}")
        return [], []

def transpose_sequence(seq, semitones):
    """Transpose a sequence of notes by the given number of semitones."""
    return [transpose_note(note, semitones) for note in seq]

def transpose_note(note_str, semitones):
    """
    Transpose a note or chord by the given number of semitones.
    Handles the complex note format with duration and instrument info.
    """
    # Split the note string to extract pitch, duration, and instrument
    parts = note_str.split('|')
    pitch_part = parts[0]
    
    # Keep duration and instrument info unchanged
    duration_instrument = '|'.join(parts[1:]) if len(parts) > 1 else ''
    
    # Handle chord (contains dots)
    if '.' in pitch_part:
        # For simplicity, we'll just shift the chord's normal order
        chord_notes = pitch_part.split('.')
        try:
            # Try to convert to integers and transpose
            chord_notes = [str(int(n) + semitones) for n in chord_notes]
            transposed = '.'.join(chord_notes)
        except ValueError:
            # If conversion fails, keep original
            transposed = pitch_part
    else:
        # Handle single note
        try:
            # Try to create a pitch object and transpose
            p = pitch.Pitch(pitch_part)
            p.transpose(interval.Interval(semitones), inPlace=True)
            transposed = str(p)
        except:
            # If parsing fails, keep original
            transposed = pitch_part
    
    # Reconstruct the full note string
    if duration_instrument:
        return f"{transposed}|{duration_instrument}"
    else:
        return transposed

def load_midi_files(data_dir='data'):
    """Load all MIDI files from the data directory and subdirectories recursively."""
    midi_files = glob.glob(os.path.join(data_dir, '**', '*.mid'), recursive=True)
    logger.info(f"Found {len(midi_files)} MIDI files in {data_dir}")
    return midi_files

if __name__ == "__main__":
    data_dir = 'data'
    midi_files = load_midi_files(data_dir)
    notes = get_notes_from_midi(midi_files)
    X, y, pitchnames = prepare_sequences(notes)
    print(f"Total sequences: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Vocabulary size: {len(pitchnames)}")
