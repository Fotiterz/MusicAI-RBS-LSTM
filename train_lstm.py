import os
import pickle
import matplotlib.pyplot as plt
from lstm_model import build_lstm_model, train_lstm_model
from data_preparation import load_midi_files, get_notes_from_midi, prepare_sequences
from save_vocab import save_vocabulary
import numpy as np
import tensorflow as tf

def train_model(data_dir='data', epochs=50, batch_size=32, sequence_length=50):
    """
    Train an improved LSTM model for music generation.
    
    Parameters:
    - data_dir: Directory containing MIDI files for training
    - epochs: Maximum number of training epochs
    - batch_size: Batch size for training
    - sequence_length: Length of input sequences
    """
    print(f"Starting training process with data from {data_dir}")
    
    # Set memory growth for GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using {len(gpus)} GPU(s) for training")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")
    
    # Regenerate vocabulary dynamically based on current training data
    save_vocabulary(data_dir=data_dir)
    
    # Load MIDI files and extract notes
    midi_files = load_midi_files(data_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {data_dir}. Please add training data.")
    
    print(f"Found {len(midi_files)} MIDI files for training")
    notes = get_notes_from_midi(midi_files)
    print(f"Extracted {len(notes)} notes from MIDI files")
    
    # Prepare sequences with the specified sequence length
    X_train, y_train, pitchnames = prepare_sequences(notes, sequence_length=sequence_length)
    print(f"Prepared {len(X_train)} training sequences with shape {X_train.shape}")
    
    # Save pitchnames for later use in generation
    with open('pitchnames.pkl', 'wb') as f:
        pickle.dump(pitchnames, f)
    print(f"Saved vocabulary with {len(pitchnames)} unique pitches")
    
    # Build and train the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]
    print(f"Building model with input shape {input_shape} and output dimension {output_dim}")
    
    model = build_lstm_model(input_shape, output_dim)
    model, history = train_lstm_model(
        model, 
        X_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.1
    )
    
    # Save the trained model
    model.save('lstm_trained_model.h5')
    print("Model trained and saved as lstm_trained_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation accuracy values if available
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as training_history.png")
    
    return model

if __name__ == "__main__":
    train_model()
