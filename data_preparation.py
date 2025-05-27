import numpy as np
from music21 import converter, instrument, note, chord, stream
import os
import glob

def get_notes_from_midi(midi_files):
    """Extract notes and chords from a list of MIDI files."""
    notes = []
    for file in midi_files:
        midi = converter.parse(file)
        print(f"Parsing {file}")
        parts = instrument.partitionByInstrument(midi)
        if parts:  # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def prepare_sequences(notes, sequence_length=50):
    """Prepare the sequences used by the Neural Network."""
    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])
    n_patterns = len(network_input)
    # Reshape input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(len(pitchnames))
    # One hot encode the output vectors
    network_output = np.eye(len(pitchnames))[network_output]
    return network_input, network_output, pitchnames

def load_midi_files(data_dir='data'):
    """Load all MIDI files from the data directory and subdirectories recursively."""
    midi_files = glob.glob(os.path.join(data_dir, '**', '*.mid'), recursive=True)
    return midi_files

if __name__ == "__main__":
    data_dir = 'data'
    midi_files = load_midi_files(data_dir)
    notes = get_notes_from_midi(midi_files)
    X, y, pitchnames = prepare_sequences(notes)
    print(f"Total sequences: {len(X)}")
