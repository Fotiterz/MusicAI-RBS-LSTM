# MusicAI-RBS-LSTM

A music generation system using both rule-based and LSTM approaches.

## Overview

This project implements two different approaches to algorithmic music composition:

1. **Rule-Based System (RBS)**: Generates music based on predefined music theory rules and patterns.
2. **LSTM Neural Network**: Generates music by learning patterns from MIDI training data.

## Recent Improvements

The LSTM model has been significantly enhanced to produce more musically coherent and pleasing output:

### 1. Model Architecture Improvements
- **Bidirectional LSTM layers** for better context understanding
- **Deeper network** with more layers for better feature extraction
- **Batch normalization** for improved training stability
- **Increased model capacity** with more units per layer
- **Improved regularization** with strategic dropout

### 2. Training Process Improvements
- **Early stopping** to prevent overfitting
- **Learning rate scheduling** to optimize convergence
- **Validation split** for better model evaluation
- **Training history visualization** for monitoring progress
- **Improved batch size** for better learning dynamics

### 3. Music Generation Algorithm Improvements
- **Enhanced music theory integration** with key signatures and chord progressions
- **Phrase structure awareness** for more natural musical phrases
- **Melodic contour guidance** for better melodic shapes
- **Adaptive temperature sampling** based on phrase position
- **Improved repetition handling** for more varied output

### 4. Data Preparation and Augmentation
- **Data augmentation through transposition** for more training examples
- **Multi-instrument support** for capturing harmony information
- **Enhanced note representation** with duration and instrument information
- **Better error handling** for robust MIDI file parsing

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/MusicAI-RBS-LSTM.git
cd MusicAI-RBS-LSTM
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download the FluidR3_GM.sf2 soundfont file for MP3 conversion (optional):
```
# On Ubuntu/Debian
sudo apt-get install fluidsynth
```

## Usage

### Web Interface

Run the Flask web server:
```
python main.py
```

Access the web interface at http://localhost:5000

### Training the LSTM Model

To train the LSTM model with your own MIDI files:

1. Place your MIDI files in the `data` directory
2. Run the training script:
```
python train_lstm.py
```

### Testing the LSTM Model

To test the improved LSTM model:
```
python test_improved_lstm.py
```

This will generate music samples with different keys and temperature settings.

## Troubleshooting

If you encounter the error `TypeError: object of type 'NoneType' has no len()` when using the LSTM model:

1. Make sure you have the `pitchnames.pkl` file in the root directory
2. If not, run `python save_vocab.py` to generate it
3. Alternatively, use the fixed version of the code in `main_fix.py`

## Files and Structure

- `main.py`: Flask web application for the user interface
- `lstm_model.py`: LSTM model definition and music generation functions
- `rule_based.py`: Rule-based music generation system
- `data_preparation.py`: Functions for preparing MIDI data for LSTM training
- `train_lstm.py`: Script for training the LSTM model
- `test_improved_lstm.py`: Script for testing the improved LSTM model
- `compare_models.py`: Script for comparing original and improved models
- `save_vocab.py`: Script for saving the vocabulary from MIDI files
- `data/`: Directory containing MIDI files for training
- `static/`: Static files for the web interface
- `templates/`: HTML templates for the web interface

## Future Improvements

1. **Transformer-based architecture** for better long-range dependencies
2. **Style transfer** for generating music in specific composer styles
3. **Conditional generation** based on mood, genre, etc.
4. **Interactive generation** with real-time user input
5. **Multi-track generation** for multiple coordinated instrument parts

## License

This project is licensed under the MIT License - see the LICENSE file for details.