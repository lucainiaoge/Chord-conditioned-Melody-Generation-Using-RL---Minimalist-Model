from data.data_utils import chord_to_binary, pitch_num_to_id_np, id_to_pitch_num_np, note_chord_to_midi
from config import config, device, IDS, LEN

import numpy as np
import music21
from music21 import harmony
import os

from reward.reward import find_key_simple, find_key_simple_root_type
from reward.chord2scale import get_scale_suggestions, UNKNOWN_KEY
from reward.scales import PITCH_NUM_2_NAME


def chord_multi_hot_to_pitch_nums(chord_multi_hot):
    pitch_nums = []
    for i in range(len(chord_multi_hot)):
        if chord_multi_hot[i]:
            pitch_nums.append(i)
    return pitch_nums

def rectify_scale(scale):
    prev = 0
    for i in range(len(scale)):
        while scale[i] < prev:
            scale[i] += 12
        prev = scale[i]
    return scale
        
class ScalePolicyBasic:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.this_chord_mem = []
        self.next_chord_mem = []
        self.key = UNKNOWN_KEY
        self.scale_notes = [0]
        self.scale_weights = [0]
        self.scale_pointer = 0
        self.octave_bias = 5*12
        self.prev_progress_percent = 0
        self.duration_id = config.duration_2_id_gen[12]
        self.rest_id = pitch_num_to_id_np(-1)
        self.direction = 1
    
    def forward(self, observation):
        this_chord = list(observation[IDS["chord"]])
        next_chord = list(observation[IDS["next_chord"]])
        progress_percent = observation[IDS["progress_percent"]]
        if abs(progress_percent - self.prev_progress_percent) > 0.9:
            self.reset()
        self.prev_progress_percent = progress_percent
        if len(this_chord) == 0 or sum(this_chord) == 0:
            return [self.rest_id, self.duration_id]
        if not (self.this_chord_mem == this_chord and self.next_chord_mem == next_chord):
            prev_scale_notes = self.scale_notes[:]
            
            self.direction = 1
            self.this_chord_mem = this_chord[:]
            self.next_chord_mem = next_chord[:]
            
            this_chord = chord_multi_hot_to_pitch_nums(this_chord)
            next_chord = chord_multi_hot_to_pitch_nums(next_chord)
        
            c1 = music21.chord.Chord(this_chord)
            root1 = c1.root().midi
            symbol1, type1 = harmony.chordSymbolFigureFromChord(c1, includeChordType=True)
            if len(next_chord) == 0:
                self.key = UNKNOWN_KEY
            else:
                c2 = music21.chord.Chord(next_chord)
                root2 = c2.root().midi
                symbol2, type2 = harmony.chordSymbolFigureFromChord(c2, includeChordType=True)
                self.key = find_key_simple_root_type(root1, type1, root2, type2)
                
            names, self.scale_weights, self.scale_notes = get_scale_suggestions(
                this_chord, self.key, root_midi_num = root1, chord_type = type1
            )
            self.scale_notes = rectify_scale(self.scale_notes[0])
#             prev_pitch = prev_scale_notes[self.scale_pointer%len(prev_scale_notes)]
#             prev_register = int(np.floor(self.scale_pointer/len(prev_scale_notes)))
#             self.scale_pointer = int(np.argmin(abs(np.array(self.scale_notes)-prev_pitch))) + len(self.scale_notes)*prev_register
            self.scale_pointer = 0
            
        octave_bonus = 12*int(np.floor(self.scale_pointer/len(self.scale_notes)))
        if octave_bonus >= 12:
            octave_bonus = 12
            self.direction = -1
        elif octave_bonus <= -12:
            octave_bonus = -12
            self.direction = 1
        pitch = self.octave_bias + self.scale_notes[self.scale_pointer%len(self.scale_notes)] + octave_bonus
        if pitch > config.pitch_max:
            pitch = config.pitch_max
        if pitch < config.pitch_min:
            pitch = config.pitch_min
        self.scale_pointer += 1 * self.direction
        return [pitch_num_to_id_np(pitch), self.duration_id]


def push_duration(this_duration, prev_durations):
    prev_durations[:-1] = prev_durations[1:]
    prev_durations[-1] = this_duration
    return prev_durations
    
class ScalePolicyMarkov:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.prev_progress_percent = 0
        self.this_chord_mem = []
        self.next_chord_mem = []
        self.key = UNKNOWN_KEY
        self.scale_notes = [0]
        self.scale_weights = [2]
        self.prev_pitch = 0
        self.direction = 1
        self.scale_pointer = 0
        self.octave_bias = 5*12
        self.prev_durations = np.ones(4, dtype=int) * 12
        self.duration_template = np.zeros(16, dtype=int)
        self.direction_template = np.zeros(16, dtype=int)
        self.step_template = np.zeros(16, dtype=int)
        self.template_pointer = 0
        self.using_template = False
        self.template_obtained = False
        self.piece_start = False
        self.rest_id = pitch_num_to_id_np(-1)
    
    def forward(self, observation):
        this_chord = list(observation[IDS["chord"]])
        next_chord = list(observation[IDS["next_chord"]])
        beat_pos = float(observation[IDS["beat_pos"]])
        time_signature = config.id_2_time_signature[int(observation[IDS["time_signature"]])]
        progress_percent = observation[IDS["progress_percent"]]
        if abs(progress_percent - self.prev_progress_percent) > 0.9:
            self.reset()
        self.prev_progress_percent = progress_percent
        
        bar_length = time_signature[0]*4/time_signature[1]*24
        if time_signature[0]%4==0 or time_signature[0]%6==0:
            time_to_next_beat = round((0.5-beat_pos)%0.5*bar_length)
        elif time_signature[0]%3 == 0:
            time_to_next_beat = round((1/3-beat_pos)%(1/3)*bar_length)
        else:
            time_to_next_beat = round((1-beat_pos)%1*bar_length)
        
        # randomly determine the next duration
        if time_to_next_beat==12 and np.random.uniform() < 0.8:
            duration = 12
        elif time_to_next_beat==24 and np.random.uniform() < 0.5:
            duration = 24
        elif np.random.uniform() > 0.4:
            duration = self.prev_durations[-1]
        else:
            duration_id = np.random.randint(0, len(config.possible_durations_gen))
            duration = config.id_2_duration_gen[duration_id]
        
        # exception: no chord
        if len(this_chord) == 0 or sum(this_chord) == 0:
            duration = 12
            duration_id = config.duration_2_id_gen[duration]
            return [self.rest_id, duration_id]
        
        # decide the scale by chords
        if not (self.this_chord_mem == this_chord and self.next_chord_mem == next_chord):
            prev_scale_len = len(self.scale_notes[:])
            
            self.this_chord_mem = this_chord[:]
            self.next_chord_mem = next_chord[:]
            this_chord = chord_multi_hot_to_pitch_nums(this_chord)
            next_chord = chord_multi_hot_to_pitch_nums(next_chord)
            c1 = music21.chord.Chord(this_chord)
            root1 = c1.root().midi
            symbol1, type1 = harmony.chordSymbolFigureFromChord(c1, includeChordType=True)
            if len(next_chord) == 0:
                self.key = UNKNOWN_KEY
            else:
                c2 = music21.chord.Chord(next_chord)
                root2 = c2.root().midi
                symbol2, type2 = harmony.chordSymbolFigureFromChord(c2, includeChordType=True)
                self.key = find_key_simple_root_type(root1, type1, root2, type2)
                
            names, _, scale_notes = get_scale_suggestions(
                this_chord, self.key, root_midi_num = root1, chord_type = type1
            )
            self.scale_notes = rectify_scale(scale_notes[0])
            self.scale_weights = np.ones_like(self.scale_notes)
            self.chord_notes = c1.normalOrder
            for i in range(len(self.scale_notes)):
                if self.scale_notes[i]%12 in self.chord_notes:
                    self.scale_weights[i] += 1
            self.scale_pointer = int(self.scale_pointer * len(self.scale_notes) / prev_scale_len)
            
            
        
        if self.using_template and self.template_obtained:
            duration = self.duration_template[self.template_pointer]
            self.direction = self.direction_template[self.template_pointer]
            step = self.step_template[self.template_pointer]
            self.template_pointer += 1
            if self.template_pointer == len(self.duration_template):
                self.using_template = False
                self.template_pointer = 0
        else:
            # randomly invert the direction
            if np.random.uniform() < 0.25:
                self.direction = -1 * self.direction
                if self.scale_pointer > len(self.scale_notes) and np.random.uniform() < 0.8:
                    self.direction = -1
                elif self.scale_pointer < -len(self.scale_notes) and np.random.uniform() < 0.8:
                    self.direction = 1
                
            # randomly determine the step of next note
            tmp_rand = np.random.uniform()
            if tmp_rand > 0.6:
                step = 1
            elif tmp_rand > 0.2:
                step = 2
            elif tmp_rand > 0.1:
                step = 3
            else:
                step = 0
                    
            if self.template_obtained:
                beat_pos_good = (beat_pos < 0.001)
                if np.random.uniform() > 0.8 and beat_pos_good:
                    self.using_template = True
                    self.template_pointer = 0
        
        self.scale_pointer += step * self.direction
        
        # consider bounds of pitches
        octave_bonus = 12*int(self.scale_pointer/len(self.scale_notes))
        if octave_bonus > 12:
            self.direction = -1
            self.scale_pointer -= self.scale_pointer%len(self.scale_notes) - 1
        elif octave_bonus < -12:
            self.direction = 1
            self.scale_pointer += len(self.scale_notes) - self.scale_pointer%len(self.scale_notes) + 1
        
        # consider the important notes
        if np.random.uniform()<0.5 or abs(beat_pos)<=0.01 or progress_percent >= 0.95:
            note_weight = self.scale_weights[self.scale_pointer%len(self.scale_notes)]
            while note_weight < max(self.scale_weights):
                if self.direction == -1:
                    self.scale_pointer += -1
                else:
                    self.scale_pointer += 1
                note_weight = self.scale_weights[self.scale_pointer%len(self.scale_notes)]
            
        # log the rhythm template and scale template
        if beat_pos <= 0.001:
            self.piece_start = True
        if self.piece_start and not self.template_obtained:
            self.duration_template[self.template_pointer] = duration
            self.direction_template[self.template_pointer] = self.direction
            self.step_template[self.template_pointer] = step
            self.template_pointer += 1
            if self.template_pointer == len(self.duration_template):
                self.template_obtained = True
        
        # finally decide the pitch and duration
        octave_bonus = 12*int(np.floor(self.scale_pointer/len(self.scale_notes)))
        pitch = self.octave_bias + self.scale_notes[self.scale_pointer%len(self.scale_notes)] + octave_bonus
        if pitch > config.pitch_max:
            pitch = config.pitch_max
        if pitch < config.pitch_min:
            pitch = config.pitch_min
        self.prev_pitch = pitch
        self.prev_durations = push_duration(duration, self.prev_durations)
        duration_id = config.duration_2_id_gen[duration]
        return [pitch_num_to_id_np(pitch), duration_id]
    

policies = []
def serial_scale_policy(observations, mode = "basic"):
    N_observation = len(observations)
    if N_observation != len(policies):
        for i in range(N_observation):
            if mode=="basic":
                policies.append(ScalePolicyBasic())
            elif mode=="markov":
                policies.append(ScalePolicyMarkov())
    actions = [0]*N_observation
    for i in range(N_observation):
        actions[i] = policies[i].forward(observations[i])
    
    return np.array(actions)

def serial_scale_markov_policy(observations):
    return serial_scale_policy(observations, mode = "markov")

def serial_scale_basic_policy(observations, mode = "basic"):
    return serial_scale_policy(observations, mode = "basic")

def scale_markov_policy(observation):
    return serial_scale_policy([observation], mode = "markov")[0]

def scale_basic_policy(observation, mode = "basic"):
    return serial_scale_policy([observation], mode = "basic")[0]
