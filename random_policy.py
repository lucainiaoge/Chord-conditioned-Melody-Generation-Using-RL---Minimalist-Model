from data.data_utils import chord_to_binary, pitch_num_to_id_np, id_to_pitch_num_np, note_chord_to_midi
from config import config, device, IDS, LEN

import numpy as np
import os

class RandomPolicy:
    def __init__(self):
        self.reset()
    
    def reset(self):
        pass
    
    def forward(self, observation):
        pitch = np.random.randint(config.pitch_min, config.pitch_max)
        duration_id = config.possible_durations_gen[
            np.random.randint(config.num_duration_types_gen)
        ]
        duration_id = config.duration_2_id_gen[duration_id]
        return [pitch_num_to_id_np(pitch), duration_id]