"""Environment related utils."""
from typing import List, Union

from music_struct import Chord, Note, Piece, Tempo, TimeSignature


def sum_note_durations(note_track: List[Note]):
    dur = 0
    for note in note_track:
        dur = dur + note.duration
    return dur

def find_state_id(time, states: Union[List[Chord], List[TimeSignature], List[Tempo]]):
    state_id = 0
    if time > states[-1].time:
        state_id = len(states) - 1
    else:
        for i in range(len(states)):
            if time > states[i].time:
                continue
            else:
                state_id = i
                break
    return state_id

def find_first_downbeat(piece: Piece):
    first_downbeat = None

    if len(piece.beats) >= 2:
        prev_beat_log = piece.beats[0]
        for beat_log in piece.beats[1:]:
            if prev_beat_log["is_downbeat"] and not beat_log["is_downbeat"]:
                first_downbeat = prev_beat_log["time"]
                break
            prev_beat_log = beat_log

    else:
        print("Piece input should have more than 1 note.")
    return first_downbeat

def build_init_note_track(piece: Piece):
    n = 1
    if len(piece.beats) >= 2:
        prev_beat_log = piece.beats[0]
        for beat_log in piece.beats[1:]:
            if prev_beat_log["is_downbeat"] and not beat_log["is_downbeat"]:
                first_downbeat = prev_beat_log["time"]
                break
            prev_beat_log = beat_log
            n += 1
    else:
        print("Piece input should have more than 1 note.")
    return piece.notes[0:n]
