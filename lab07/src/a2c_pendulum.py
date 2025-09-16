#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import warnings
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
import os
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        hidden_dim = 64
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, out_dim)
        self.actor_std = nn.Linear(hidden_dim, out_dim)
        initialize_uniformly(self.actor_mean)
        initialize_uniformly(self.actor_std)
        
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.actor_mean(x)
        std = F.softplus(self.actor_std(x)) + 1e-5
        dist = Normal(mean, std)

        pre_tanh = dist.rsample()                 # reparameterization
        tanh_u = torch.tanh(pre_tanh)
        action_scale = 2.0
        action = tanh_u * action_scale            # [-2, 2]

        # log-det of the transform a = scale * tanh(u)
        # = sum ( log(scale) + log(1 - tanh(u)^2) )
        # add eps for stability
        eps = 1e-6
        log_det_per_dim = torch.log((1.0 - tanh_u.pow(2)).clamp(min=eps)) + torch.log(torch.tensor(action_scale, device=pre_tanh.device))
        log_det = log_det_per_dim.sum(dim=-1)
        #############################

        return action, dist, pre_tanh, log_det


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        hidden_dim = 64
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        initialize_uniformly(self.critic)


        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.critic(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist, pre_tanh, log_det = self.actor(state)

        if self.is_test:
            # deterministic: use mean, but apply same squash+scale
            mean = dist.mean
            action_det = torch.tanh(mean) * 2.0
            return action_det.cpu().detach().numpy()

        # train: compute correct log_prob (on pre_tanh) and store
        log_prob = dist.log_prob(pre_tanh).sum(dim=-1) - log_det
        self.transition = [state, log_prob]

        return action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        ############TODO#############
        # value_loss = ?
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_value = self.critic(next_state)
            target = reward + self.gamma * next_value * (1 - done)

        value = self.critic(state)
        value_loss = F.mse_loss(value, target)

        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        # policy_loss = ?

        advantage = (target - value).detach()
        policy_loss = -(log_prob * advantage).mean()

        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()
    
    def train(self):
        """Train the agent."""
        self.is_test = False
        
        for ep in tqdm(range(1, self.num_episodes)): 
            actor_losses, critic_losses, scores = [], [], []
            state, _ = self.env.reset(seed=self.seed + ep)
            score = 0
            done = False
            while not done:
                # self.env.render()  # Render the environment
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                self.total_step += 1
                # W&B logging
                wandb.log({
                    "step": self.total_step,
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    tqdm.write(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": ep,
                        "return": score,
                        "total_steps": self.total_step
                    })  
                    # save checkpoint for this episode
                    ckpt_dir = "checkpoints/task1"
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_ep{ep:04d}.pt")

                    # save only model weights (actor + critic) to avoid storing extra metadata
                    actor_state = {k: v.cpu() for k, v in self.actor.state_dict().items()}
                    critic_state = {k: v.cpu() for k, v in self.critic.state_dict().items()}
                    torch.save({
                        "actor_state_dict": actor_state,
                        "critic_state_dict": critic_state,
                        "total_step": self.total_step,
                    }, ckpt_path)
                    tqdm.write(f"Saved checkpoint: {ckpt_path}")

    def test(self, video_folder: str, seed: int = None) -> float:
        """Test the agent and record one episode to video_folder. Returns episode total reward.
        If seed is provided, temporarily set self.seed to that value for the run.
        """
        prev_mode = self.is_test
        prev_seed = self.seed
        if seed is not None:
            self.seed = seed
        self.is_test = True

        tmp_env = self.env
        # self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0.0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # restore
        self.env = tmp_env
        self.is_test = prev_mode
        self.seed = prev_seed
        return float(score)

    def evaluate(self, start_seed: int = 0, n_seeds: int = 21, ckpt_path: str = None, video_dir: str = "videos/task1") -> Tuple[float, list]:
        """Evaluate deterministic policy by loading actor/critic weights (if provided) and
        running `test()` for seeds start_seed..start_seed+n_seeds-1, saving each episode video.
        Returns (average_reward, rewards_list).
        """

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        
        # Print checkpoint info
        total_step = ckpt.get("total_step", "unknown")
        print(f"Loaded checkpoint from step: {total_step}")

        os.makedirs(video_dir, exist_ok=True)
        rewards = []

        for s in range(start_seed, start_seed + n_seeds):
            subdir = os.path.join(video_dir, f"seed_{s}")
            os.makedirs(subdir, exist_ok=True)
            score = self.test(video_folder=subdir, seed=s)
            rewards.append(score)
            print(f"seed {s}: return = {score:.3f}  video -> {subdir}")

        avg = float(np.mean(rewards)) if rewards else 0.0
        print(f"Average over seeds {start_seed}..{start_seed + n_seeds - 1}: avg_return={avg:.3f}")
        return avg, rewards
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    # Filter out gymnasium deprecation warnings and video overwrite warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=4e-4)
    parser.add_argument("--critic-lr", type=float, default=4e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=0) # entropy can be disabled by setting this to 0
    parser.add_argument("--mode", choices=["train","evaluate"], default="train")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt to load for evaluation")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    
    agent = A2CAgent(env, args)
    if args.mode == "train":
        agent.train()
    else:
        avg, rewards = agent.evaluate(start_seed=args.seed, n_seeds=21, ckpt_path=args.ckpt)
        print(f"Average reward over {len(rewards)} seeds: {avg:.3f}")