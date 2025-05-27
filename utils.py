import os
from music21 import midi
from pydub import AudioSegment

def save_stream_to_midi(music_stream, midi_path):
    """
    Save a music21 stream to a MIDI file.
    """
    mf = midi.translate.streamToMidiFile(music_stream)
    mf.open(midi_path, 'wb')
    mf.write()
    mf.close()

def convert_midi_to_mp3(midi_path, mp3_path):
    """
    Convert a MIDI file to MP3 using FluidSynth and pydub.
    Requires FluidSynth installed and soundfont file available.
    """
    import subprocess
    import sys
    # Path to soundfont file (user must have a soundfont file)
    soundfont_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "FluidR3_GM.sf2"))  # Place your .sf2 file here in the MusicGenApp directory
    soundfont_path = soundfont_path.replace("\\", "/")  # Normalize path for FluidSynth on Windows
    if not os.path.exists(soundfont_path):
        raise FileNotFoundError("Soundfont file not found: " + soundfont_path + "\nPlease download it from https://member.keymusician.com/Member/FluidR3_GM/index.html and place it in the MusicGenApp directory.")
    
    # Normalize midi_path for FluidSynth
    midi_path_norm = os.path.abspath(midi_path).replace("\\", "/")
    
    # Generate wav file path
    wav_path = midi_path_norm.replace('.mid', '.wav')
    
    # Correct command argument order: options before soundfont and midi file
    fluidsynth_cmd = ["fluidsynth", "-ni", "-F", wav_path, "-r", "44100", soundfont_path, midi_path_norm]
    print("Running FluidSynth command:", " ".join(fluidsynth_cmd))
    try:
        subprocess.run(fluidsynth_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("FluidSynth error:", e, file=sys.stderr)
        raise RuntimeError("FluidSynth failed to convert MIDI to WAV. Please ensure the soundfont file is valid and FluidSynth is installed correctly.")
    
    # Convert wav to mp3
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    
    # Clean up wav file
    os.remove(wav_path)
