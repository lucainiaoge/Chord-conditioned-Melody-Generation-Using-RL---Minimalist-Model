import torch
import music21
from music21 import harmony
from config import device, config 
from reward.chord2scale import get_scale_suggestions, UNKNOWN_KEY
from reward.scales import PITCH_NUM_2_NAME
import numpy as np

def find_pos_in_12(pitches):
    return [pitch % 12 for pitch in pitches]

def find_key_simple_root_type(root1, type1, root2, type2):
    key = UNKNOWN_KEY
    # V -> I, V -> i
    if (root1 - root2)%12 == 7 and ("maj" in type1 or "dom" in type1):
        if "maj" in type2:
            key = (root2)%12
        elif "min" in type2:
            key = (root2+3)%12
            
    # IV -> V, iv -> V
    elif (root1 - root2)%12 == 10 and ("maj" in type2 or "dom" in type2):
        if "maj" in type1:
            key = (root2-7)%12
        elif "min" in type1:
            key = (root2-7+3)%12

    # ii -> V
    elif (root1 - root2)%12 == 7 and ("maj" in type2 or "dom" in type2):
        if "minor-seventh" in type1 or "minor-ninth" in type1 or "minor-11th" in type1 or "minor-13th" in type1:
            key = (root2-7)%12
        elif "dim" in type1:
            key = (root2-7+3)%12
    
    # IV/iv -> I, IV/iv -> i
    '''
    elif (root1 - root2)%12 == 5:
        if ("maj" in type1 or "min" in type1) and "maj" in type2:
            key = (root2)%12
        elif ("maj" in type1 or "min" in type1) and "min" in type2:
            key = (root2+3)%12
    '''
    
    return key

def find_key_simple(this_chord, next_chord):
    c1 = music21.chord.Chord(this_chord)
    if len(next_chord) == 0:
        next_chord = this_chord[:]
    c2 = music21.chord.Chord(next_chord)
    symbol1, type1 = harmony.chordSymbolFigureFromChord(c1, includeChordType=True)
    symbol2, type2 = harmony.chordSymbolFigureFromChord(c2, includeChordType=True)
    
    return find_key_simple_root_type(c1.root().midi, type1, c2.root().midi, type2)

def reward_chord_func(this_state, this_action):
    if this_action.pitch == -1:
        pitch_reward = -4
    elif (this_action.pitch % 12) in find_pos_in_12(this_state.chord.pitches):
        pitch_reward = 8
    elif (this_action.pitch % 12) in find_pos_in_12(this_state.next_chord.pitches):
        pitch_reward = 0
    else:
        pitch_reward = -5
    if this_state.beat_pos == 0 and pitch_reward > 0:
        pitch_reward += 6
    return pitch_reward

def get_suggested_scale_weights(this_state, this_action, verbose=False):
    this_chord = this_state.chord.pitches
    next_chord = this_state.next_chord.pitches

    if len(this_chord) == 0:
        return 0
    else:
        c1 = music21.chord.Chord(this_chord)
        root1 = c1.root().midi
        symbol1, type1 = harmony.chordSymbolFigureFromChord(c1, includeChordType=True)

    if len(next_chord) == 0:
        key = UNKNOWN_KEY
    else:
        c2 = music21.chord.Chord(next_chord)
        root2 = c2.root().midi
        symbol2, type2 = harmony.chordSymbolFigureFromChord(c2, includeChordType=True)
        key = find_key_simple_root_type(root1, type1, root2, type2)

    _, suggested_scale_weights, _ = get_scale_suggestions(
        this_state.chord.pitches, key, root_midi_num = root1, chord_type = type1
    )

    scale_weights = suggested_scale_weights[0]
    if verbose:
        chord_symbol_str1 = harmony.chordSymbolFigureFromChord(music21.chord.Chord(this_state.chord.pitches))
        chord_symbol_str2 = harmony.chordSymbolFigureFromChord(music21.chord.Chord(this_state.next_chord.pitches))
        print("_________________")
        print("chord, next chord: ", chord_symbol_str1, chord_symbol_str2)
        if type(key) == int:
            print("estimated key: ", PITCH_NUM_2_NAME[key])
        else:
            print("estimated key: ", key)
        print("scale suggestions: ", get_scale_suggestions(this_state.chord.pitches, key)[0])
        print("pitch decision, prev pitch: ", this_action.pitch, this_state.note_track[-1].pitch)
    return scale_weights

def reward_scale_harmony(this_state, this_action, verbose = False):
    if this_action.pitch < 20 or this_action.pitch >= 80:
        return -4
    else:
        scale_weights = get_suggested_scale_weights(this_state, this_action, verbose)
        if type(scale_weights) == int:
            return 0
        else:
            prev_pitch = this_state.note_track[-1].pitch
            if this_state.beat_pos == 0 or this_state.beat_pos == 0.5:
                pitch_reward = max(scale_weights[this_action.pitch%12]-1, -0.5)*6
            else:
                signs = np.sign(np.array(scale_weights)-0.1)
                pitch_reward = signs[this_action.pitch%12] - 1 + (scale_weights[this_action.pitch%12]-1)*6
            if this_state.beat_pos == 0 and pitch_reward > 0:
                pitch_reward += 6
            return pitch_reward

def reward_chord_note_harmony(this_state, this_action, verbose = False):
    if this_action.pitch < 20 or this_action.pitch >= 80:
        return -4
    else:
        chord_notes = [p%12 for p in this_state.chord.pitches]
        if this_action.pitch % 12 in chord_notes:
            pitch_reward = 6
            if this_state.beat_pos == 0 and pitch_reward > 0:
                pitch_reward += 6
        else:
            pitch_reward = -6
        return pitch_reward

        
def reward_repetition_pitch(pitches_seq, tolerance=2):
    # repetition of single note
    if len(pitches_seq)>config.n_gram:
        pitches_seq = pitches_seq[-config.n_gram:]
    count_repeat = 0
    max_repeat = 0
    for i in range(1, len(pitches_seq)):
        if pitches_seq[i] == pitches_seq[i - 1]:
            count_repeat += 1
            max_repeat = max(max_repeat, count_repeat)
        elif pitches_seq[i]%12 == pitches_seq[i - 1]%12:
            count_repeat += 0.5
            max_repeat = max(max_repeat, count_repeat)
        else:
            count_repeat = 0
    return -max([0,max_repeat-tolerance])

def reward_jumps_func(this_state, this_action, rewards=[-2,5,3,0,-11,-11,-12]):
    prev_pitch = this_state.note_track[-1].pitch
    this_pitch = this_action.pitch
    if prev_pitch == -1 or this_pitch == -1:
        return rewards[0]
    else:
        jump = abs(prev_pitch - this_pitch)
        if jump >= 1 and jump < 5:
            return rewards[1]
        elif jump >= 5 and jump < 9:
            return rewards[2]
        elif jump >= 9 and jump <= 11:
            return rewards[3]
        elif jump == 0:
            return rewards[4]
        elif jump == 12:
            return rewards[5]
        else:
            return rewards[6]

def reward_repetition_duration_func(duration_seq, tolerance=4):
    if len(duration_seq)>config.n_gram:
        duration_seq = duration_seq[-config.n_gram:]
    count_repeat = 0
    max_repeat = 0
    for i in range(1, len(duration_seq)):
        if duration_seq[i] == duration_seq[i-1]:
            count_repeat += 1
            max_repeat = max(max_repeat, count_repeat)
        else:
            count_repeat = 0
    return -max([0,max_repeat-tolerance])

def reward_duration_value_func(this_state, this_action, rewards=[1,0.5,-5,0]):
    if this_action.duration == 24:
        duration_reward = rewards[0]
    elif this_action.duration == 12:
        duration_reward = rewards[1]
    elif this_action.duration<=6 or this_action.duration>=48:
        duration_reward = rewards[2]
    else:
        duration_reward = rewards[3]
    return duration_reward

def reward_downbeat_func(this_state, this_action):
    numerator = this_state.time_signature.numerator
    denominator = this_state.time_signature.denominator
    beat_pos = this_state.beat_pos
    bar_length = numerator*4/denominator*24
    time_to_next_downbeat = round((1-beat_pos)%1*bar_length)
    if this_action.duration > time_to_next_downbeat:
        return -5
    else:
        return 0
        
            
def handcrafted_reward_single_note(this_state, this_action):
    note_track = this_state.note_track
    if len(note_track) == 0:
        return 0, 0
    pitches_seq = torch.tensor([note.pitch for note in note_track] + [this_action.pitch])
    duration_seq = torch.tensor([note.duration for note in note_track] + [this_action.duration])
    reward_chord = reward_chord_note_harmony(this_state, this_action)
    duration_reward = reward_duration_value_func(this_state, this_action, rewards=[0.5,0.5,-5,0])
    pitch_reward = reward_chord
    return pitch_reward, duration_reward

def handcrafted_reward_harmony(
    this_state, 
    this_action, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,0.5,-5,0],
    reward_weights = [[1,1,0],[0.5,0.3,0]]
):
    note_track = this_state.note_track
    if len(note_track) == 0:
        return 0, 0
    pitches_seq = torch.tensor([note.pitch for note in note_track] + [this_action.pitch])
    duration_seq = torch.tensor([note.duration for note in note_track] + [this_action.duration])
    if reward_weights[0][0] > 0:
        reward_harmony = reward_scale_harmony(this_state, this_action)
    else:
        reward_harmony = 0
    reward_jumps = reward_jumps_func(this_state, this_action, rewards=jump_rewards)
    reward_chord_note = reward_chord_note_harmony(this_state, this_action)
    reward_duration_repitition = reward_repetition_duration_func(duration_seq)
    reward_duration_value = reward_duration_value_func(this_state, this_action, rewards=duration_value_rewards)
    reward_downbeat = reward_downbeat_func(this_state, this_action)
    
    pitch_reward = reward_weights[0][0]*reward_harmony + reward_weights[0][1]*reward_jumps + reward_weights[0][2]*reward_chord_note
    duration_reward = reward_weights[1][0]*reward_duration_repitition + reward_weights[1][1]*reward_duration_value + reward_weights[1][2]*reward_downbeat
    return pitch_reward, duration_reward

