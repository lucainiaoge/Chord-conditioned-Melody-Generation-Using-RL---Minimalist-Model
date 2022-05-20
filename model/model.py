import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn import TransformerEncoderLayer, TransformerEncoder
import gym
from gym import spaces
import math
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from config import config, device, action_space, observation_space, IDS, LEN
from data.data_utils import pitch_num_to_id, id_to_pitch_num
from .model_utils import SpiralEmbedding4d, SpiralEmbeddingChord, PositionalEncoding

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class HarmonyPolicyNet(nn.Module):
    def __init__(self, emb_dim=64):
        super(HarmonyPolicyNet, self).__init__()
        self.harmony_fc = nn.Sequential(nn.Linear(12*2+1, 64), nn.ReLU(), 
                                       nn.Linear(64, 64), nn.ReLU(),nn.Linear(64, emb_dim))
    def forward(self, observation):
        chord = observation[:,IDS["chord"]]  # (B,12) of multi hots
        next_chord = observation[:,IDS["next_chord"]]  # (B,12) of multi hots
        beat_pos = observation[:,IDS["beat_pos"]] # (B,1) of real numbers
        
        emb_harmony = torch.cat([chord, next_chord], dim=-1) #(B,12,2)
        emb_harmony = emb_harmony.flatten(start_dim=1) #(B,24)
        emb = torch.cat([emb_harmony, beat_pos], dim=-1)
        return self.harmony_fc(emb)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return x
    
class NgramFeatureExtractorModule(nn.Module):
    def __init__(self, features_dim=64):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(NgramFeatureExtractorModule, self).__init__()
        self.n_gram = 16
        self.harmony_policy_net = HarmonyPolicyNet()
        
        #debugging
        self.chroma_embedding = nn.Embedding(13, config.pitch_embedding_dim)
        self.chroma_pe = PositionalEncoding(config.pitch_embedding_dim)
        self.duration_embedding = torch.nn.Embedding(config.num_duration_types+1, config.duration_embedding_dim)
        self.duration_pe = PositionalEncoding(config.duration_embedding_dim)
        self.pitch_emb_fc = nn.Sequential(nn.Linear((config.pitch_embedding_dim+1)*self.n_gram, 64), nn.ReLU(), 
                                             nn.Linear(64, 64))
        self.duration_emb_fc = nn.Sequential(nn.Linear((config.duration_embedding_dim+1)*self.n_gram, 64), nn.ReLU(), 
                                                nn.Linear(64, 16))
        
        self.note_feature_fc = nn.Sequential(
            nn.Linear(64+64+16+1+1+1+1+config.pe_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.final_fc = nn.Sequential(
            nn.Linear(64+config.pitch_embedding_dim+1, features_dim), nn.BatchNorm1d(features_dim), nn.ReLU()
        )
    
    def get_chroma_octave(self, pitches):
        chroma = pitches%12
        chroma[pitches==-1] = 12
        octave = torch.div(pitches, 12, rounding_mode='floor')
        return chroma, octave
    
    def get_pitch_dur_embedding(self, prev_pitches, prev_durations, prev_beat_pos):
        batch_size = prev_pitches.shape[0]
        chroma, octave = self.get_chroma_octave(prev_pitches)
        
        pitch_embs = self.chroma_embedding(chroma)
        pitch_embs = self.chroma_pe(pitch_embs)
        pitch_embs = torch.cat([pitch_embs, octave.unsqueeze(-1)/6], dim=-1)
        
        duration_embs = self.duration_embedding(prev_durations)
        duration_embs = self.duration_pe(duration_embs)
        duration_embs = torch.cat([duration_embs, prev_beat_pos.unsqueeze(-1)], dim=-1)
        
        pitch_embs = torch.flatten(pitch_embs, start_dim=1)
        duration_embs = torch.flatten(duration_embs, start_dim=1)
        return pitch_embs, duration_embs
    
    def forward(self, observation) -> torch.Tensor:
        # data preparation
        prev_pitches = observation[:,IDS["pitch_track"]].long() # (B,L)
        prev_durations = observation[:,IDS["duration_track"]].long() # (B,L)
        prev_beat_pos = observation[:,IDS["beat_pos_track"]] # (B,L)
        beat_pos = observation[:,IDS["beat_pos"]]  # (B,1) of real numbers
        max_time = observation[:,IDS["max_time"]]/24/10  # (B,1) of real numbers
        pos_encoding = observation[:,IDS["pos_encoding"]]  # (B,dim) of real numbers
        progress_percent = observation[:,IDS["progress_percent"]]  # (B,1) of real numbers
        time_to_go = observation[:,IDS["time_to_go"]]/24  # (B,1) of real numbers
        batch_size = prev_pitches.shape[0]

        # pitch and duration embeddings
        pitch_embs, duration_embs = self.get_pitch_dur_embedding(prev_pitches, prev_durations, prev_beat_pos)
        pitch_embs = self.pitch_emb_fc(pitch_embs)
        duration_embs = self.duration_emb_fc(duration_embs)
        
        harmony_embs = self.harmony_policy_net(observation)
        
        prev_note_chroma = prev_pitches[:,-1]%12
        prev_note_chroma[prev_pitches[:,-1]==-1] = 12
        prev_note_chroma = self.chroma_embedding(prev_note_chroma)
        prev_note_pitch = (prev_pitches[:,-1].float()-30) / 60
        prev_note_emb =  torch.cat(
            (prev_note_chroma, prev_note_pitch.unsqueeze(-1)),
            dim=-1,
        )
        
        # overall embedding
        emb_input = torch.cat(
            (harmony_embs, pitch_embs, duration_embs,
             beat_pos, time_to_go, max_time, progress_percent, pos_encoding),
            dim=-1,
        )
        z = self.note_feature_fc(emb_input)
        z = self.final_fc(torch.cat(
            (z, prev_note_emb), dim=-1)
        )
        return z
    
    
class NgramFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=64):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(NgramFeatureExtractor, self).__init__(observation_space, features_dim=features_dim)
        self.feature_extractor = NgramFeatureExtractorModule(features_dim)
    
    def forward(self, observation) -> torch.Tensor:
        return self.feature_extractor(observation)

class CustomNgramNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param features_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        features_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNgramNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions (before the last action/value net, which is defined in policy.py in SB3)
        # will be used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(nn.Linear(features_dim, last_layer_dim_pi), nn.ReLU())
        # Value network
        self.value_net = nn.Sequential(nn.Linear(features_dim, last_layer_dim_vf), nn.ReLU())

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNgramNetwork(self.features_dim)

