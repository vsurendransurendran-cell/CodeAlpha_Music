import os
from music21 import stream, chord, note

def tokens_to_stream(tokens):
    """
    Converts a list of token strings back into a music21 Stream object.
    """
    s = stream.Stream()
    for tok in tokens:
        if tok == 'Rest' or tok == 'UNK':
            s.append(note.Rest())
        elif tok.startswith('Nt_'):
            pitch_name = tok.split('_')[1]
            s.append(note.Note(pitch_name))
        elif tok.startswith('Ch_'):
            pitches = tok.split('_')[1:]
            s.append(chord.Chord(pitches))
        else:
            s.append(note.Rest())
    return s

def save_midi(stream_obj, output_path):
    """
    Writes a music21 Stream to a MIDI file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stream_obj.write('midi', fp=output_path)
    print(f"Successfully saved MIDI to {output_path}")