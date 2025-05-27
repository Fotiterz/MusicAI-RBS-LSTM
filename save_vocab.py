import pickle
from data_preparation import get_notes_from_midi, load_midi_files

def save_vocabulary(data_dir='data', vocab_path='pitchnames.pkl'):
    midi_files = load_midi_files(data_dir)
    notes = get_notes_from_midi(midi_files)
    pitchnames = sorted(set(notes))
    with open(vocab_path, 'wb') as f:
        pickle.dump(pitchnames, f)
    print(f"Vocabulary saved to {vocab_path} with {len(pitchnames)} entries.")

if __name__ == "__main__":
    save_vocabulary()
