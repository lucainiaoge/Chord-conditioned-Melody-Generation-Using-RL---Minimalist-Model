import os
import json
import numpy as np
from functools import partial
import argparse

from music_struct import Piece, Chord, Note, TimeSignature, Tempo
from gym_env import state2array
from data import NottinghamDataset
from environment.environment import ComposerAction, ComposerState, ComposerEnvironmentConditional
from reward.reward import handcrafted_reward_single_note, handcrafted_reward_harmony

def compute_tonal_vector(pitch, rs = [1,1,0.5]):
    tonal_vec = np.zeros(6)
    if pitch >= 0:
        tonal_vec[0] = rs[0]*np.sin(pitch*7*np.pi/6)
        tonal_vec[1] = rs[0]*np.cos(pitch*7*np.pi/6)
        tonal_vec[2] = rs[1]*np.sin(pitch*3*np.pi/2)
        tonal_vec[3] = rs[1]*np.cos(pitch*3*np.pi/2)
        tonal_vec[4] = rs[2]*np.sin(pitch*2*np.pi/3)
        tonal_vec[5] = rs[2]*np.cos(pitch*2*np.pi/3)
    return tonal_vec

def get_chroma(pitches):
    chroma = []
    for p in pitches:
        if p%12 not in chroma and p>=0:
            chroma.append(p%12)
    return chroma

def compute_tonal_centroid(pitches, weights = None, rs = [1,1,0.5]):
    if len(pitches) == 0:
        return np.zeros(6)
    else:
        if weights is None:
            weights = np.ones_like(np.array(pitches))
        else:
            weights = np.array(weights)
        weights = weights/np.sum(weights)
        centroid = np.zeros(6)
        for i in range(len(pitches)):
            centroid += weights[i]*compute_tonal_vector(pitches[i], rs = rs)
        return centroid

def compute_CTnCTR(pitches, chord):
    if len(pitches) == 0:
        return 0
    chroma_chord = get_chroma(chord)
    n_chord_tone = 0
    n_non_chord_tone = 0
    for p in pitches:
        if p > 0 and p%12 in chroma_chord:
            n_chord_tone += 1
        else:
            n_non_chord_tone += 1
    return n_chord_tone / (n_chord_tone + n_non_chord_tone)

def compute_CMTD_and_CTnCTR(piece:Piece, rs = [1,1,0.5]):
    melody_time_curser = 0
    i_note = 0
    chord_time_curser = 0
    chord_list = piece.chords
    note_list = piece.notes
    current_melody_pitches = []
    current_note_weights = []
    tonal_distances = []
    TD_weights = []
    num_chord_notes = []
    num_melody_notes = []
    
    if len(chord_list) > 1:
        chord_time_curser = chord_list[0].time
        for i_chord in range(len(chord_list)):
            this_chord = chord_list[i_chord]
            chord_time_curser += this_chord.duration
            while melody_time_curser < chord_time_curser and i_note<len(note_list):
                current_melody_pitches.append(note_list[i_note].pitch)
                current_note_weights.append(note_list[i_note].duration)
                melody_time_curser += note_list[i_note].duration
                i_note += 1
            this_chord_TC = compute_tonal_centroid(this_chord.pitches, None, rs = rs)
            this_melody_TC = compute_tonal_centroid(current_melody_pitches, current_note_weights, rs = rs)
            tonal_distances.append(np.linalg.norm(this_chord_TC-this_melody_TC))
            TD_weights.append(this_chord.duration)
            local_CTnCTR = compute_CTnCTR(current_melody_pitches, this_chord.pitches)
            num_chord_notes.append(int(local_CTnCTR * len(current_melody_pitches)))
            num_melody_notes.append(len(current_melody_pitches))
            
        tonal_distances = np.array(tonal_distances)
        TD_weights = np.array(TD_weights)/sum(TD_weights)
        CMTD = np.dot(tonal_distances, TD_weights)
        CTnCTR = sum(num_chord_notes) / sum(num_melody_notes)
        return CMTD, CTnCTR
        
    else:
        return None, 0

    
class MusicPieceEvaluator:
    def __init__(self, reward_func_list, test_json_dir, aug=False):
        self.test_dataset = NottinghamDataset(
            training=False, aug=False, overfit=False,
            test_dir=test_json_dir, logger=None
        )
        self.dataset_length = self.test_dataset.__len__()
        self.reward_func_list = reward_func_list
        self.n_reward_func = len(reward_func_list)
        self.pitch_reward_result_list = [np.empty(0)]*self.n_reward_func
        self.dur_reward_result_list = [np.empty(0)]*self.n_reward_func
        self.tonal_distance_list = []
        self.CTnCTR_list = []
    
    def run(self, rs = [1,1,0.5]):
        for i_piece in range(self.dataset_length):
            try:
                piece = self.test_dataset[i_piece]
                note_list = piece.notes
                CMTD, CTnCTR = compute_CMTD_and_CTnCTR(piece, rs=rs)
                self.tonal_distance_list.append(CMTD)
                self.CTnCTR_list.append(CTnCTR)

                env = ComposerEnvironmentConditional(piece.chords, piece.tempos, piece.time_signatures)
                state = env.reset(piece.chords, piece.tempos, piece.time_signatures)
                pitch_reward_bufs = [0]*self.n_reward_func
                dur_reward_bufs = [0]*self.n_reward_func
                for i_note in range(len(note_list)):
                    action = note_list[i_note]
                    for i_reward in range(self.n_reward_func):
                        this_pitch_reward, this_dur_reward = self.reward_func_list[i_reward](state, action)
                        pitch_reward_bufs[i_reward] = float(i_note * pitch_reward_bufs[i_reward] + this_pitch_reward) / (i_note+1)
                        dur_reward_bufs[i_reward] = float(i_note * dur_reward_bufs[i_reward] + this_dur_reward) / (i_note+1)
                    state = env.step(action)

                for i_reward in range(self.n_reward_func):
                    self.pitch_reward_result_list[i_reward] = np.append(self.pitch_reward_result_list[i_reward], pitch_reward_bufs[i_reward])
                    self.dur_reward_result_list[i_reward] = np.append(self.dur_reward_result_list[i_reward], dur_reward_bufs[i_reward])
                
            except Exception as e:
                print(e)
            
        return self.pitch_reward_result_list, self.dur_reward_result_list, self.tonal_distance_list, self.CTnCTR_list
    
    def get_ave_result(self):
        self.ave_pitch_rewards = [rewards.mean() for rewards in self.pitch_reward_result_list]
        self.ave_dur_rewards = [rewards.mean() for rewards in self.dur_reward_result_list]
        self.ave_MCTD = np.array(self.tonal_distance_list).mean()
        self.ave_CTnCTR = np.array(self.CTnCTR_list).mean()
        result_dict = {
            "ave_pitch_rewards": self.ave_pitch_rewards,
            "ave_dur_rewards": self.ave_dur_rewards,
            "ave_MCTD": self.ave_MCTD,
            "ave_CTnCTR": self.ave_CTnCTR
        }
        return result_dict

reward_scale_note = partial(
    handcrafted_reward_harmony, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,0.5,-5,0],
    reward_weights = [[1,1,0],[0.5,0.3,0]]
)

reward_chord_note = partial(
    handcrafted_reward_harmony, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,0.5,-5,0],
    reward_weights = [[0,1,1],[0.5,0.3,0]]
)

reward_scale_note_no_harmony = partial(
    handcrafted_reward_harmony, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,0.5,-5,0],
    reward_weights = [[0,1,0],[0.5,0.3,0]]
)

reward_scale_note_no_interval = partial(
    handcrafted_reward_harmony, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,0.5,-5,0],
    reward_weights = [[1,0,0],[0.5,0.3,0]]
)

reward_scale_note_4th_only = partial(
    handcrafted_reward_harmony, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,-5,-5,-5],
    reward_weights = [[1,1,0],[0.5,0.3,0]]
)

reward_scale_note_no_rep = partial(
    handcrafted_reward_harmony, 
    jump_rewards = [-2,5,3,0,-12,-12,-14],
    duration_value_rewards = [1,0.5,-5,-5],
    reward_weights = [[1,1,0],[0,0.3,0]]
)

reward_func_list = [
    reward_scale_note,
    reward_chord_note,
    reward_scale_note_no_interval,
    reward_scale_note_no_harmony,
    reward_scale_note_4th_only,
    reward_scale_note_no_rep
]

def main(args):
    eval_runner = MusicPieceEvaluator(
        reward_func_list, args.json_dir, aug=False
    )
    eval_runner.run()
    ave_results = eval_runner.get_ave_result()
    print("--start--")
    print("reward SN pitch: ", ave_results['ave_pitch_rewards'][0])
    print("reward CN pitch: ", ave_results['ave_pitch_rewards'][1])
    print("reward NI pitch: ", ave_results['ave_pitch_rewards'][2])
    print("reward NH pitch: ", ave_results['ave_pitch_rewards'][3])
    print("reward 4TH pitch: ", ave_results['ave_pitch_rewards'][4])
    print("reward NR pitch: ", ave_results['ave_pitch_rewards'][5])
    print("----")
    print("reward SN rhythm: ", ave_results['ave_dur_rewards'][0])
    print("reward CN rhythm: ", ave_results['ave_dur_rewards'][1])
    print("reward NI rhythm: ", ave_results['ave_dur_rewards'][2])
    print("reward NH rhythm: ", ave_results['ave_dur_rewards'][3])
    print("reward 4TH rhythm: ", ave_results['ave_dur_rewards'][4])
    print("reward NR rhythm: ", ave_results['ave_dur_rewards'][5])
    print("--end--")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', type=str)
    args = parser.parse_args()
    main(args)
    