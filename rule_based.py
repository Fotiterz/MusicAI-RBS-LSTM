import music21
import random

def generate_rule_based_music(tempo_bpm=120, key_signature='C', length_bars=8, instruments=['Piano'], variation_level=0.5):
    """
    Generate a simple polyphonic music piece using rule-based system with added variations.
    
    Parameters:
    - tempo_bpm: Tempo in beats per minute
    - key_signature: Key of the piece (e.g., 'C', 'G', 'F#')
    - length_bars: Length of the piece in bars
    - instruments: List of instrument names (e.g., ['Piano', 'Violin'])
    - variation_level: Float between 0 and 1 controlling the amount of variation (0 = no variation, 1 = max variation)
    
    Returns:
    - music21.stream.Stream object representing the music piece
    """
    # Create a score
    score = music21.stream.Score()
    
    # Set tempo
    tempo_indication_template = music21.tempo.MetronomeMark(number=tempo_bpm)
    
    # Set key signature
    key_sig_template = music21.key.Key(key_signature)
    
    # Time signature (4/4)
    time_sig_template = music21.meter.TimeSignature('4/4')
    
    # Define simple chord progression in the key (I - IV - V - I)
    rn_progression = ['I', 'IV', 'V', 'I']
    chords_progression = [music21.roman.RomanNumeral(rn, key_signature) for rn in rn_progression]
    num_chords = len(chords_progression)
    
    # Introduce variation in chord order by shuffling with probability based on variation_level
    if random.random() < variation_level:
        chords_progression = random.sample(chords_progression, len(chords_progression))
    
    # Voice leading helper: find closest pitch in next chord to previous pitch
    def closest_pitch(prev_pitch, chord_pitches):
        if prev_pitch is None:
            return random.choice(chord_pitches)
        closest = min(chord_pitches, key=lambda p: abs(p.midi - prev_pitch.midi))
        return closest
    
    # For each instrument, create a part
    instrument_map = {
        'Piano': music21.instrument.Piano,
        'Violin': music21.instrument.Violin,
        'Flute': music21.instrument.Flute,
        'Clarinet': music21.instrument.Clarinet,
        'Trumpet': music21.instrument.Trumpet,
        'Guitar': music21.instrument.Guitar,
        'Drums': music21.instrument.Percussion,
    }
    
    for instr_name in instruments:
        part = music21.stream.Part()
        instr_class = instrument_map.get(instr_name, music21.instrument.Piano)
        instr_obj = instr_class()
        part.insert(0, instr_obj)
        
        # Insert tempo, key, and time signature at the beginning of the part
        part.insert(0, music21.tempo.MetronomeMark(number=tempo_bpm))
        part.insert(0, music21.key.Key(key_sig_template.tonic.name, key_sig_template.mode))
        part.insert(0, music21.meter.TimeSignature(time_sig_template.ratioString))
        
        bars_generated = 0
        previous_pitches = [None, None, None]  # For voice leading in harmony (3 voices)
        previous_melody_pitch = None
        
        while bars_generated < length_bars:
            chord_obj = chords_progression[bars_generated % num_chords]
            if instr_name == 'Drums':
                drum_pitches = [35, 38, 42, 46]
                # Randomize drum hits based on variation_level
                for pitch_num in drum_pitches:
                    if random.random() < variation_level:
                        n = music21.note.Unpitched()
                        n.midiChannel = 9
                        n.pitches = [music21.pitch.Pitch(midi=pitch_num)]
                        n.quarterLength = random.choice([0.5, 1.0])  # random duration
                        part.append(n)
            else:
                # Harmony: 3 voices with voice leading
                harmony_chord_pitches = chord_obj.pitches
                harmony_notes = []
                for i in range(3):
                    pitch = closest_pitch(previous_pitches[i], harmony_chord_pitches)
                    n = music21.note.Note(pitch)
                    n.quarterLength = 1.0
                    harmony_notes.append(n)
                    previous_pitches[i] = pitch
                
                harmony_chord = music21.chord.Chord(harmony_notes)
                harmony_chord.quarterLength = 4.0
                part.append(harmony_chord)
                
                # Melody: scale-based melodic line with smooth transitions
                scale = key_sig_template.getScale()
                melody_durations = []
                remaining = 4.0
                while remaining > 0:
                    dur = random.choice([0.25, 0.5, 1.0])
                    if dur > remaining:
                        dur = remaining
                    melody_durations.append(dur)
                    remaining -= dur
                
                for dur in melody_durations:
                    scale_pitches = scale.getPitches()
                    if previous_melody_pitch is None:
                        pitch = random.choice(scale_pitches)
                    else:
                        close_pitches = [p for p in scale_pitches if abs(p.midi - previous_melody_pitch.midi) <= 2]
                        if close_pitches:
                            pitch = random.choice(close_pitches)
                        else:
                            pitch = random.choice(scale_pitches)
                    n = music21.note.Note(pitch)
                    n.quarterLength = dur
                    part.append(n)
                    previous_melody_pitch = pitch
            
            bars_generated += 1
        
        score.append(part)
    
    return score

if __name__ == "__main__":
    s = generate_rule_based_music()
    s.show('midi')
