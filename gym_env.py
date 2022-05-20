from environment.environment import ComposerAction, ComposerState, ComposerEnvironmentConditional
from data.data_utils import chord_to_binary, pitch_num_to_id_np, id_to_pitch_num_np, note_chord_to_midi, note_chord_to_json
from music_struct import Note, Piece, Chord, TimeSignature, Tempo
from data import NottinghamDataset
from config import config, device, action_space, observation_space, IDS, LEN

import numpy as np
import gym
from gym import spaces
from typing import List
import collections
import math
import numpy as np

# SB5-related tutorials
# colab notebooks: https://github.com/araffin/rl-tutorial-jnrr19
# document: https://stable-baselines3.readthedocs.io/en/master/
# github: https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3
# examples: https://stable-baselines.readthedocs.io/en/master/guide/examples.html

class PosEncoder:
    def __init__(self, dim = 16, max_len: int = 10872):
        self.pe = np.zeros((max_len, dim), dtype="float32")
        for pos in range(max_len):
            for i in range(0, dim, 2):
                self.pe[pos, i] = math.sin(pos / (10000**((2*i)/dim)))
                self.pe[pos, i + 1] = math.cos(pos / (10000**((2*(i+1))/dim)))
    def encode(self, pos):
        return self.pe[pos,:]

PE = PosEncoder(dim = config.pe_dim, max_len = 10872)
    
def composer_state_to_dict_observation(state:ComposerState, pos_encoder = PE):
    observation = {
        "pitch_track": np.zeros(config.n_gram, dtype="int64"),
        "duration_track": np.zeros(config.n_gram, dtype="int64"),
        "beat_pos_track": np.zeros(config.n_gram, dtype="float32"),
        "time": np.zeros(1),
        "time_signature": np.zeros(1, dtype="int64"),
        "tempo": np.zeros(1),
        "beat_pos": np.zeros(1),
        "time_to_go": np.zeros(1),
        "max_time": np.zeros(1),
        "pos_encoding": np.zeros(config.pe_dim),
        "progress_percent": np.zeros(1),
        "chord": np.zeros(12),
        "next_chord": np.zeros(12),
    }
    for t in range(len(state.note_track)):
        if t>=8:
            break
        pitch = state.note_track[-t-1].pitch
        if pitch > 0:
            observation["pitch_track"][-1-t] = pitch%12
        else:
            observation["pitch_track"][-1-t] = 12
        observation["duration_track"][-1-t] = config.duration_2_id[state.note_track[-t-1].duration]
        observation["beat_pos_track"][-1-t] = state.beat_pos_track[-t-1]
    meter_id = config.time_signature_2_id[(state.time_signature.numerator, state.time_signature.denominator)]
    observation["time"] = np.array([state.time], dtype="float32")
    observation["time_signature"] = np.array([meter_id], dtype="int8")
    observation["tempo"] = np.array([state.tempo.qpm], dtype="float32")
    observation["beat_pos"] = np.array([state.beat_pos], dtype="float32")
    observation["time_to_go"] = np.array([state.time_to_go], dtype="float32")
    observation["max_time"] = np.array([state.max_time], dtype="float32")
    observation["pos_encoding"] = PE.encode(round(state.max_time * state.progress_percent))
    observation["progress_percent"] = np.array([state.progress_percent], dtype="float32")
    observation["chord"] = np.array(chord_to_binary(state.chord.pitches), dtype="int8")
    observation["next_chord"] = np.array(chord_to_binary(state.next_chord.pitches), dtype="int8")
    
    return collections.OrderedDict(sorted(observation.items()))

def composer_dict_observation_to_array(observation):
    obs_array = np.zeros(LEN).astype(np.float)
    for key, ids in IDS.items():
        obs_array[ids] = np.array(observation[key]).astype(np.float)
    return obs_array

def state2array(state):
    return composer_dict_observation_to_array(
        composer_state_to_dict_observation(
            state
        )
    )

class GymComposerEnvironmentConditional(gym.Env):
    def __init__(
        self, 
        reward_func, 
        aug=False, 
        overfit=False, 
        cut_len=16, 
        training=True, 
        no_render = False, 
        out_dir="results/",
        permute = True,
        fixed_len = None
    ):
        super(GymComposerEnvironmentConditional, self).__init__()
        self.reward_func = reward_func
        self.out_dir = out_dir
        self.overall_counter = 0
        
        self.dataset = NottinghamDataset(training=training, aug=aug, overfit=overfit)
        self.dataset_length = self.dataset.__len__()
        if permute:
            self.g_data_perm = np.random.permutation(self.dataset_length)
        else:
            self.g_data_perm = np.arange(self.dataset_length)
        self.g_data_counter = 0
        self.piece = self.dataset.__getitem__(self.g_data_perm[self.g_data_counter])
        self.env = ComposerEnvironmentConditional(self.piece.chords, self.piece.tempos, self.piece.time_signatures)
        self.overall_counter += 1
        
        self.cut_len = cut_len
        self.prev_state = None
        self.no_render = no_render
        self.training = training
        
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.done = False
        self.progress_percent = 0
        
        self.fixed_len = fixed_len
        
    def reset(self):
        self.done = False
        self.progress_percent = 0
        self.g_data_counter += 1
        if self.g_data_counter >= self.dataset_length:
            self.g_data_counter = 0
            self.g_data_perm = np.random.permutation(self.dataset_length)
        self.piece = self.dataset.__getitem__(self.g_data_perm[self.g_data_counter])
        self.overall_counter += 1
        init_state = self.env.reset(self.piece.chords, self.piece.tempos, self.piece.time_signatures)
        self.prev_state = init_state
        observation = composer_state_to_dict_observation(init_state)
        return composer_dict_observation_to_array(observation)
    
    def step(self, action):
        pitch = id_to_pitch_num_np(action[0])
        duration = config.id_2_duration_gen[action[1]]
        note_in = Note(**{"pitch":pitch, "duration":duration})
        reward = self.reward_func(self.prev_state, note_in)
        reward = reward[0] + reward[1]
        state = self.env.step(note_in)
        self.prev_state = state
        observation = composer_state_to_dict_observation(state)
        observation = composer_dict_observation_to_array(observation)
        done = state.done
        self.done = done
        self.progress_percent = state.progress_percent
        info = {}
        if self.done and (not self.training) and (not self.no_render):
            self.render()
        if not (self.fixed_len is None):
            if state.n_step >= self.fixed_len:
                done = True
                self.done = True
            elif done == True:
                done = False
                self.done = False
        return observation, reward, done, info
        
    def render(self, mode='console'):
        note_list = self.prev_state.note_track
        chord_list = self.env.chords
        time_signature_list = self.env.time_signatures
        title = self.piece.title
        name = self.piece.name
        out_path = self.out_dir + "_" + str(self.overall_counter) + "_" + title + name
        note_chord_to_midi(note_list, chord_list, time_signature_list, out_path + ".mid")
        note_chord_to_json(note_list, chord_list, time_signature_list, 
                           title=title, out_path=out_path + ".json"
        )
    def close(self):
        pass
