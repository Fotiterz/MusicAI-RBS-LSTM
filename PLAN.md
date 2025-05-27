# Music Generating App Plan

## Information Gathered
- The app should support multiple instruments.
- It should implement both a Rule-Based System and an LSTM model for music generation.
- The user can use the model to generate sound.
- Output should be an mp3 file based on user parameters.
- The generated sound must be polyphonic, cohesive, and harmonic.
- Hardware constraints: Intel i7-7500u, Nvidia 940MX, 4GB RAM, so complexity must be managed carefully.

## Plan

### Project Structure
- MusicGenApp/
  - rule_based.py          # Rule-Based System implementation
  - lstm_model.py          # LSTM model definition, training, and generation
  - data/                  # Dataset for training (optional, or pre-trained model)
  - utils.py               # Utility functions (MIDI handling, mp3 conversion)
  - main.py                # Main app interface (CLI or simple GUI)
  - requirements.txt       # Python dependencies
  - README.md              # Project overview and usage instructions

### Technologies and Libraries
- Python 3.8+
- TensorFlow/Keras for LSTM model
- music21 or pretty_midi for music representation and MIDI handling
- pydub or ffmpeg for mp3 generation from MIDI or audio
- numpy, scipy for data processing
- Optional: simple CLI interface using argparse or a minimal GUI with Tkinter

### Rule-Based System
- Implement music generation rules for harmony, rhythm, and polyphony.
- Support multiple instruments by assigning different MIDI channels or tracks.
- Generate MIDI sequences based on user parameters (e.g., tempo, key, instrument selection).

### LSTM Model
- Define an LSTM model architecture suitable for polyphonic music generation.
- Use a pre-trained model or train on a small dataset to keep resource usage low.
- Allow user to generate music by sampling from the model given input parameters.
- Output MIDI sequences from the model.

### User Parameters
- Tempo (BPM)
- Key signature
- Instrument selection (e.g., piano, violin, drums)
- Length of generated piece (in bars or seconds)
- Generation mode: Rule-Based or LSTM

### Output
- Convert generated MIDI sequences to mp3 files.
- Ensure output is polyphonic, cohesive, and harmonic.
- Save mp3 file to disk and notify user of location.

### Web App Interface
- Use Flask or FastAPI for backend API.
- Frontend with HTML, CSS, and JavaScript for user input and mp3 playback/download.
- Backend endpoints to trigger rule-based or LSTM generation and return mp3 file.

## Dependent Files to be Edited
- All files will be created from scratch.

## Follow-up Steps
- Implement the rule-based system.
- Implement the LSTM model.
- Implement MIDI to mp3 conversion utilities.
- Implement web app interface (backend and frontend).
- Test generation and output.
- Optimize for hardware constraints.
