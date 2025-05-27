import os
from lstm_model import build_lstm_model, train_lstm_model
from data_preparation import load_midi_files, get_notes_from_midi, prepare_sequences
from save_vocab import save_vocabulary
import numpy as np

def train_model(data_dir='data', epochs=20, batch_size=64):
    # Regenerate vocabulary dynamically based on current training data
    save_vocabulary(data_dir=data_dir)
    
    midi_files = load_midi_files(data_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}. Please add training data.")
    notes = get_notes_from_midi(midi_files)
    X_train, y_train, pitchnames = prepare_sequences(notes)
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]
    model = build_lstm_model(input_shape, output_dim)
    model = train_lstm_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    model.save('lstm_trained_model.h5')
    print("Model trained and saved as lstm_trained_model.h5")

if __name__ == "__main__":
    train_model()
