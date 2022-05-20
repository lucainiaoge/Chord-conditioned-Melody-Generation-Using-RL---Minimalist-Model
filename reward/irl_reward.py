import torch
import torch.nn as nn
from config import config, device, action_space, observation_space, IDS, LEN
from imitation.rewards.reward_nets import RewardNet
from imitation.util import networks
import gym
from gym import spaces

class HarmonyPolicyNet(nn.Module):
    def __init__(self, emb_dim=64):
        super(HarmonyPolicyNet, self).__init__()
        self.harmony_fc = nn.Sequential(nn.Linear(12*2+1, 32), nn.ReLU(), nn.Linear(32, emb_dim))
    def forward(self, observation):
        chord = observation[:,IDS["chord"]]  # (B,12) of multi hots
        next_chord = observation[:,IDS["next_chord"]]  # (B,12) of multi hots
        beat_pos = observation[:,IDS["beat_pos"]] # (B,1) of real numbers
        
        emb_harmony = torch.cat([chord, next_chord], dim=-1) #(B,12,2)
        emb_harmony = emb_harmony.flatten(start_dim=1) #(B,24)
        emb = torch.cat([emb_harmony, beat_pos], dim=-1)
        return self.harmony_fc(emb)

class NgramRewardFeature(nn.Module):
    def __init__(self):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(NgramRewardFeature, self).__init__()
        self.n_gram = 16
        self.harmony_policy_net = HarmonyPolicyNet(emb_dim=8)
        
        #debugging
        self.chroma_embedding = nn.Embedding(13, 8)
        self.duration_embedding = torch.nn.Embedding(config.num_duration_types+1, 4)
        self.note_feature_fc = nn.Sequential(
            nn.Linear(8+1+1+1+1, 8), nn.BatchNorm1d(8), nn.ReLU()
        )
        self.final_fc = nn.Sequential(
            nn.Linear(8+(8+2+4)*2, 16), nn.ReLU(), 
            nn.Linear(16, 16), nn.ReLU()
        )
    
    def get_chroma_octave(self, pitches):
        chroma = pitches%12
        chroma[pitches==-1] = 12
        octave = torch.div(pitches, 12, rounding_mode='floor')
        return chroma, octave
    
    def get_pitch_embedding(self, pitch):
        height = (pitch.float()-30) / 60
        chroma, octave = self.get_chroma_octave(pitch)
        pitch_embs = self.chroma_embedding(chroma)
        pitch_embs = torch.cat([pitch_embs, octave.unsqueeze(-1)/6, height.unsqueeze(-1)], dim=-1)
        return pitch_embs
    
    def forward(self, observation, action) -> torch.Tensor:
        # data preparation
        prev_pitches = observation[:,IDS["pitch_track"]].long() # (B,L)
        prev_durations = observation[:,IDS["duration_track"]].long() # (B,L)
        prev_beat_pos = observation[:,IDS["beat_pos_track"]] # (B,L)
        beat_pos = observation[:,IDS["beat_pos"]]  # (B,1) of real numbers
        max_time = observation[:,IDS["max_time"]]/24/10  # (B,1) of real numbers
        progress_percent = observation[:,IDS["progress_percent"]]  # (B,1) of real numbers
        time_to_go = observation[:,IDS["time_to_go"]]/24  # (B,1) of real numbers
        batch_size = prev_pitches.shape[0]

        # pitch and duration embeddings
        
        harmony_embs = self.harmony_policy_net(observation)
        
        prev_pitch_emb = self.get_pitch_embedding(prev_pitches[:,-1])
        action_pitch_emb = self.get_pitch_embedding(action[:,0].long())
        
        prev_dur_emb = self.duration_embedding(prev_durations[:,-1])
        action_dur_emb = self.duration_embedding(action[:,1].long())
                                             
        prev_note_emb =  torch.cat(
            (prev_pitch_emb, prev_dur_emb),
            dim=-1,
        )
        action_emb =  torch.cat(
            (action_pitch_emb, action_dur_emb),
            dim=-1,
        )
            
        emb_input = torch.cat(
            (harmony_embs, beat_pos, time_to_go, max_time, progress_percent),
            dim=-1,
        )
        z = self.note_feature_fc(emb_input)
        feat = self.final_fc(torch.cat(
            (z, prev_note_emb, action_emb), dim=-1)
        )
        return feat


class CustomerRewardNet(RewardNet):
    def __init__(self, 
        observation_space: gym.Space,
        action_space: gym.Space,
        **kwargs,
    ):
        super().__init__(observation_space, action_space)
        self.reward_feat = NgramRewardFeature()
        full_build_mlp_kwargs = {
            "hid_sizes": (16, 16),
        }
        full_build_mlp_kwargs.update(
            {
                # we do not want these overridden
                "in_size":16,
                "out_size": 1,
                "squeeze_output": True,
            },
        )

        self.mlp = networks.build_mlp(**full_build_mlp_kwargs)
        
    def forward(self, state, action, next_state, done) -> torch.Tensor:
        rew_feat = self.reward_feat(state, action)
        rew = self.mlp(rew_feat)
        return rew


    
    