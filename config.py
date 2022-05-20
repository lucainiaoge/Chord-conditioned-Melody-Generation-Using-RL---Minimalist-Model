import torch
import gym
from gym import spaces
import numpy as np
import collections

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Config:
    def __init__(self):
        # model
        self.pitch_embedding_dim = 8
        self.chord_embedding_dim = 64
        self.duration_embedding_dim = 8
        self.time_signature_embedding_dim = 32
        self.decoder_hidden_dim = 128
        self.reward_hidden_dim = 64
        self.decoder_layer_num = 1
        self.note_pred_hidden_dim = 64
        self.pitch_pred_hidden_dim = 64
        self.duration_pred_hidden_dim = 32
        self.value_pred_hidden_dim = 32
        self.spiral_embedding = False
        # path
        self.train_root_path = "data/nottingham/train"
        self.train_overfit_path = "data/nottingham/overfit"
        self.val_root_path = "data/nottingham/val"
        # data
        self.max_input_length = 76
        self.max_output_length = 453
        # pitch
        self.pitch_min = 49  # 55-6
        self.pitch_max = 93  # 88+5
        self.pitch_num = self.pitch_max - self.pitch_min + 1 + 1  # both ends and rest note
        self.possible_pitches = [-1] + [p for p in range(self.pitch_min, self.pitch_max+1)]
        # duration
        self.possible_durations = [3,4,6,
            8,9,
            12,16,18,
            24,32,36,42,
            48,54,60,
            72,84,
            96,108,
            120,144,168,180,192,240,480,
        ]
        self.duration_2_id = {v: i for i, v in enumerate(self.possible_durations)}
        self.id_2_duration = {i: v for i, v in enumerate(self.possible_durations)}
        self.num_duration_types = len(self.id_2_duration)
        
#         self.possible_durations_gen = [6,12,18,24,36,48,72,96]
        self.possible_durations_gen = [12,24]
        self.duration_2_id_gen = {v: i for i, v in enumerate(self.possible_durations_gen)}
        self.id_2_duration_gen = {i: v for i, v in enumerate(self.possible_durations_gen)}
        self.gen_dur_id_2_duration_id = {i: self.duration_2_id[self.id_2_duration_gen[i]] for i in self.id_2_duration_gen}
        self.num_duration_types_gen = len(self.id_2_duration_gen)
        
        # time signature
        self.filter_signature = True
        self.time_signatures_filtered = [(4, 4), (3, 4), (2, 4), (6, 8)]
        self.time_signatures_all = [
            (4, 4),
            (3, 4),
            (2, 4),
            (6, 8),
            (2, 2),
            (3, 8),
            (1, 8),
            (9, 8),
            (6, 4),
            (1, 4),
            (3, 2),
            (1, 16),
            (5, 4),
        ]
        if self.filter_signature:
            self.possible_time_signatures = self.time_signatures_filtered
        else:
            self.possible_time_signatures = self.time_signatures_all
        self.time_signature_2_id = {
            v: i for i, v in enumerate(self.possible_time_signatures)
        }
        self.id_2_time_signature = {
            i: v for i, v in enumerate(self.possible_time_signatures)
        }
        self.num_time_signature_types = len(self.possible_time_signatures)
        # training
        self.max_training_timesteps = 800000
        self.batch_size_per_gpu = 4
        self.num_workers = 8
        self.num_epochs = 10
        self.lr = 0.01
        self.weight_decay = 0.001
        self.momentum = 0.9
        self.grad_norm_clip = 10
        self.eps_clip = 0.2
        self.log_per_num_piece = 2
        self.print_per_num_piece = 2
        self.save_model_per_num_piece = 50
        self.validate_per_step = 80000
        self.gamma = 0.5

        self.r_step_batch_num = 4
        self.g_step_batch_num = 4
        self.r_step_updates = 4
        
        self.greedy_epsilon = 0.5
        self.epsilon_dec = 5e-3
        
        self.dqn_batchs_size = 16
        self.clear_buf_step = 32

        self.n_gram = 16
        self.pe_dim = 8

config = Config()

array_len = 0

beg_pitch_track = array_len
array_len += config.n_gram
end_pitch_track = array_len

beg_duration_track = array_len
array_len += config.n_gram
end_duration_track = array_len

beg_beat_pos_track = array_len
array_len += config.n_gram
end_beat_pos_track = array_len

beg_time = array_len
array_len += 1 # time
end_time = array_len

beg_time_signature = array_len
array_len += 1 # time_signature
end_time_signature = array_len

beg_beat_pos = array_len
array_len += 1 # beat_pos
end_beat_pos = array_len

beg_time_to_go = array_len
array_len += 1 # time_to_go
end_time_to_go = array_len

beg_max_time = array_len
array_len += 1 # max_time
end_max_time = array_len

beg_pos_encoding = array_len
array_len += config.pe_dim # pos_encoding
end_pos_encoding = array_len

beg_progress_percent = array_len
array_len += 1 # progress_percent
end_progress_percent = array_len

beg_chord = array_len
array_len += 12
end_chord = array_len

beg_next_chord = array_len
array_len += 12
end_next_chord = array_len

IDS = {
    "pitch_track": np.arange(beg_pitch_track, end_pitch_track),
    "duration_track": np.arange(beg_duration_track, end_duration_track),
    "beat_pos_track": np.arange(beg_beat_pos_track, end_beat_pos_track),
    "time": np.arange(beg_time, end_time),
    "time_signature": np.arange(beg_time_signature, end_time_signature),
    "beat_pos": np.arange(beg_beat_pos, end_beat_pos),
    "time_to_go": np.arange(beg_time_to_go, end_time_to_go),
    "max_time": np.arange(beg_max_time, end_max_time),
    "pos_encoding": np.arange(beg_pos_encoding, end_pos_encoding),
    "progress_percent": np.arange(beg_progress_percent, end_progress_percent),
    "chord": np.arange(beg_chord, end_chord),
    "next_chord": np.arange(beg_next_chord, end_next_chord),
}
IDS = collections.OrderedDict(sorted(IDS.items()))
LEN = array_len


action_space = spaces.MultiDiscrete([config.pitch_num, config.num_duration_types_gen])
'''
observation_space = spaces.Dict({
            "pitch_track": spaces.MultiDiscrete([13]*config.n_gram),
            "duration_track": spaces.MultiDiscrete([config.num_duration_types]*config.n_gram),
            "beat_pos_track": spaces.Box(low=0, high=1, shape=(config.n_gram,)),
            "time": spaces.Box(low=-1, high=np.inf, shape=(1,)),
            "time_signature": spaces.MultiDiscrete([1]),
            "tempo": spaces.Box(low=0, high=300, shape=(1,)),
            "beat_pos": spaces.Box(low=0, high=1, shape=(1,)),
            "time_to_go": spaces.Box(low=0, high=768, shape=(1,)),
            "max_time": spaces.Box(low=0, high=12000, shape=(1,)),
            "pos_encoding": spaces.Box(low=-1, high=1, shape=(config.pe_dim,)),
            "progress_percent": spaces.Box(low=0, high=1, shape=(1,)),
            "chord": spaces.MultiBinary(12),
            "next_chord": spaces.MultiBinary(12),
#             "prev_note_chroma": spaces.MultiDiscrete([13]),
#             "prev_note_octave": spaces.Box(low=0, high=12, shape=(1,)),
        })
'''
observation_space = spaces.Box(low=-1, high=np.inf, shape=(LEN,))