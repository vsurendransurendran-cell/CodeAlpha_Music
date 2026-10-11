import os
import json
from music21 import corpus, chord, note

def parse_chorale(score):
    tokens = []
    try:
        chordified = score.chordify()
        for element in chordified.flatten().recurse():
            if isinstance(element, chord.Chord):
                pitches = sorted([str(p.nameWithOctave) for p in element.pitches])
                tokens.append("Ch_" + "_".join(pitches))
            elif isinstance(element, note.Note):
                tokens.append("Nt_" + str(element.pitch.nameWithOctave))
            elif isinstance(element, note.Rest):
                tokens.append("Rest")
    except:
        pass
    return tokens