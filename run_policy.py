import time
from datetime import date
import os
import sys
import argparse

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from config import config, device, IDS, LEN
from gym_env import GymComposerEnvironmentConditional
from reward.reward import handcrafted_reward_single_note, handcrafted_reward_harmony
from random_policy import RandomPolicy
from rule_based_policy import scale_basic_policy, scale_markov_policy

class PolicyRunner:
    def __init__(
        self, 
        callable_policy, 
        reward_func = None,
        save_path = "results/", 
        policy_name = "exp", 
        aug=False, 
        training=False, 
        max_step = 1000,
        permute = False
    ):
        
        self.policy = callable_policy
        if reward_func is None:
            self.reward_func = handcrafted_reward_single_note
        else:
            self.reward_func = reward_func
        
        today = date.today()
        date_str = today.strftime("%b-%d-%Y")
        current_time = time.localtime()
        time_str = time.strftime("%H.%M.%S", current_time)
        subdir = policy_name + "_" + date_str + "_" + time_str + "/"
        result_dir = os.path.join(save_path, subdir)
        self.render_path = result_dir
        
        if not os.path.exists(self.render_path):
            os.mkdir(self.render_path)
            
        self.env = GymComposerEnvironmentConditional(
                    self.reward_func, 
                    aug, overfit=False, cut_len=config.n_gram, 
                    training = training, out_dir=self.render_path, permute=permute
                )
        
        self.max_step = max_step
        self.rewards = []
        self.dones = []
    
    def run(self, max_pieces = None, SB3_policy = None, deterministic = True):
        if SB3_policy is None:
            if not max_pieces:
                max_pieces = self.env.dataset_length
            for i_data in range(self.env.dataset_length):
                if i_data > max_pieces:
                    break
                observation = self.env.reset()
                done = False
                step_cnt = 0
                while not done and step_cnt < self.max_step:
                    action = self.policy(observation)
                    if len(action[0])==2 and action[1] is None:
                        action = action[0]
                    observation, reward, done, info = self.env.step(action)
                    self.rewards.append(reward)
                    self.dones.append(done)
                    step_cnt += 1
                self.env.render()
        else:
            
            episode_rewards, episode_lengths = evaluate_policy(
                SB3_policy,
                self.env,
                n_eval_episodes=self.env.dataset_length,
                render=True,
                deterministic=deterministic,
                return_episode_rewards=False
            )
            self.rewards = episode_rewards/episode_lengths
            self.dones = [True]
        
        return self.rewards, self.dones


def main(args):
    policy_name = args.policy_name
    if "scale" in policy_name:
        if "markov" in policy_name:
            callable_policy = scale_markov_policy
            print("Running scale markov policy")
        else:
            callable_policy = scale_basic_policy
            print("Running scale basic policy")
    elif "random" in policy_name:
        callable_policy = RandomPolicy().forward
        print("Running random policy")
    else:
        callable_policy = None
        policy = PPO.load(args.policy_ckpt)
        policy.policy.eval()

    policy_runner = PolicyRunner(
        callable_policy, 
        reward_func = handcrafted_reward_single_note,
        save_path = args.save_path, 
        policy_name = policy_name
    )
    if callable_policy is None:
        rewards, done = policy_runner.run(SB3_policy = policy, deterministic = True)
    else:
        rewards, done = policy_runner.run()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_name', type=str, default="scale_basic")
    parser.add_argument('save_path', type=str, default="results/")
    parser.add_argument('policy_ckpt', type=str)
    args = parser.parse_args()
    main(args)
    