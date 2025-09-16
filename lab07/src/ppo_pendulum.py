#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 2: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random
from collections import deque
from typing import Deque, List, Tuple
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

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        std_min: int = 0,
        std_max: int = 2.0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        hidden_dim = 64
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_mean = nn.Linear(hidden_dim, out_dim)
        self.actor_std = nn.Linear(hidden_dim, out_dim)
        self.std_min = std_min
        self.std_max = std_max
        init_layer_uniform(self.actor_mean)
        init_layer_uniform(self.actor_std)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        # Remeber to initialize the layer weights
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.actor_mean(x)
        std = F.softplus(self.actor_std(x)) + 1e-5
        std = torch.clamp(std, self.std_min, self.std_max)  # clamp std to avoid numerical issues
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
        init_layer_uniform(self.critic)
        #############################


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.critic(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float
) -> List:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    num_steps = len(rewards)
    gae_returns = [0] * num_steps  # Pre-allocate list

    for step in reversed(range(num_steps)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns[step] = gae + values[step]  # Assign value directly

    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # learning rate
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        
        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False
        
        self.obs_dim = obs_dim

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action_post, dist, pre_tanh, log_det = self.actor(state)   # action_post is tanh(pre_tanh)*scale
        # deterministic post-tanh mean for test, otherwise sampled post-tanh action
        if self.is_test:
            mean_pre = dist.mean
            selected_action = torch.tanh(mean_pre) * 2.0
        else:
            selected_action = action_post

        log_prob = dist.log_prob(pre_tanh).sum(dim=-1) - log_det

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(pre_tanh.detach())
            self.values.append(value.detach())
            self.log_probs.append(log_prob.detach())

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses, critic_losses, entropy_losses = [], [], []

        for state, pre_tanh, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist, _, _ = self.actor(state)
            eps = 1e-6
            tanh_u = torch.tanh(pre_tanh)
            log_det_per_dim = torch.log((1.0 - tanh_u.pow(2)).clamp(min=eps)) + torch.log(torch.tensor(2.0, device=pre_tanh.device))
            log_det = log_det_per_dim.sum(dim=-1)
            log_prob = dist.log_prob(pre_tanh).sum(dim=-1) - log_det  
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            actor_loss = -torch.min(ratio * adv,
                                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv).mean()
            entropy_loss = self.entropy_weight* dist.entropy().mean()
            actor_loss -= entropy_loss
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            target_value = self.critic(state)
            critic_loss = F.mse_loss(target_value, return_)

            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        entropy_loss = sum(entropy_losses) / len(entropy_losses)

        return actor_loss, critic_loss, entropy_loss 

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        for ep in tqdm(range(1, self.num_episodes)):
            score = 0
            print("\n")
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed + episode_count)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    tqdm.write(f"Episode {episode_count}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "episode": episode_count,
                        "return": score,
                        "total_steps": self.total_step
                    })  
                    score = 0
                    ckpt_dir = "checkpoints/task2"
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

            actor_loss, critic_loss, entropy_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            # Log losses to wandb
            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "episode": episode_count,
            })

        # termination
        self.env.close()

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

    def evaluate(self, start_seed: int = 0, n_seeds: int = 21, ckpt_path: str = None, video_dir: str = "videos/task2") -> Tuple[float, list]:
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
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=5e-3)
    parser.add_argument("--discount-factor", type=float, default=0.9)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=256)
    parser.add_argument("--update-epoch", type=int, default=64)
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", help="Run mode: train or evaluate")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt to load for evaluation")
    parser.add_argument("--video-dir", type=str, default="./videos/task2", help="Directory to save test videos")
    parser.add_argument("--num-test-episodes", type=int, default=20, help="Number of episodes for testing/evaluation")
    args = parser.parse_args()

    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # only initialize wandb during training
    if args.mode == "train":
        wandb.init(project="DLP-Lab7-PPO-Pendulum", name=args.wandb_run_name, save_code=True)

    agent = PPOAgent(env, args)
    if args.mode == "train":
        agent.train()
    else:
        avg, rewards = agent.evaluate(start_seed=args.seed, n_seeds=21, ckpt_path=args.ckpt)
        print(f"Average reward over {len(rewards)} seeds: {avg:.3f}")