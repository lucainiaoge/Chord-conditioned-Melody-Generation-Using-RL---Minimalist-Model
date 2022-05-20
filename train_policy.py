import time
from datetime import date
import os
import torch
import argparse
import gym
from gym import wrappers
import numpy as np

from reward.reward import handcrafted_reward_single_note, handcrafted_reward_harmony
from gym_env import GymComposerEnvironmentConditional
from model import NgramFeatureExtractor, CustomActorCriticPolicy
from config import config, device

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


torch.cuda.empty_cache()


def get_exp_dirs(checkpoint_dir = "results/", model = "PPO"):
    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    current_time = time.localtime()
    time_str = time.strftime("%H.%M.%S", current_time)
    subdir = "exp_" + date_str + "_" + time_str + "_" + model + "/"
    result_dir = os.path.join(checkpoint_dir, subdir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    model_path = os.path.join(result_dir, model+"_SB3/")
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    log_path = os.path.join(result_dir, "logs/")
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    render_path = result_dir
    return model_path, log_path, render_path

class RL_Trainer_SB3:
    def __init__(
        self, 
        reward_func, 
        aug = True, 
        overfit = False, 
        cut_len = 8, 
        features_dim = 64, 
        checkpoint_dir = "results/", 
        device = "cpu", 
        load_model_pth = None,
        model = "PPO",
        n_env = 4
    ):
        model_path, log_path, render_path = get_exp_dirs(checkpoint_dir, model)
        
        self.reward_func = reward_func
        self.venv = DummyVecEnv([lambda: GymComposerEnvironmentConditional(
            reward_func, aug, overfit, cut_len, training = True, out_dir=render_path
        )] * n_env)
        self.eval_env = GymComposerEnvironmentConditional(
            reward_func, aug, overfit = False, cut_len=cut_len, training = False, out_dir=render_path, 
        )
        policy_kwargs = dict(
            features_extractor_class=NgramFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
        )
        if "PPO" in model:
            self.model = PPO(
                CustomActorCriticPolicy, 
                self.venv, 
                policy_kwargs=policy_kwargs, 
                ent_coef=0.05,#0.05,
                verbose = 0, 
                seed=1,
                device = device
            )
        elif "A2C" in model:
            self.model = A2C(
                CustomActorCriticPolicy, 
                self.venv, 
                policy_kwargs=policy_kwargs, 
                ent_coef=0.05,
                verbose = 0, 
                seed=1,
                device = device
            )
        
        self.checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=model_path)
        self.eval_callback = EvalCallback(self.eval_env, best_model_save_path=model_path,
                             log_path=log_path, eval_freq=2000,
                             deterministic=True, render=False)

    
    def train(self, step = 100000):
        self.model.learn(step, callback=[self.checkpoint_callback,self.eval_callback])

def main(args):
    reward_func = handcrafted_reward_harmony
    reward_func = partial(
        handcrafted_reward_harmony, 
        jump_rewards = [-2,5,3,0,-12,-12,-14],
        duration_value_rewards = [1,0.5,-5,-5],
        reward_weights = [[1,1,0],[0.5,0.3,0]]
    )
    trainer = RL_Trainer_SB3(
        reward_func, 
        aug = args.aug, 
        overfit = args.overfit, 
        cut_len = config.n_gram,
        device = device,
        model = args.model_name,
        n_env = args.n_env
    )
    trainer.train(args.train_steps)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, default="PPO")
    parser.add_argument('--train_steps', type=int, default=500000)
    parser.add_argument('--aug', type=bool, default=True)
    parser.add_argument('--overfit', type=bool, default=False)
    parser.add_argument('--n_env', type=int, default=4)
    args = parser.parse_args()
    print(args)
    # main(args)