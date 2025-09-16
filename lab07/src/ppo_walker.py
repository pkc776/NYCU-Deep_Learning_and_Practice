#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
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
from tqdm import tqdm
import os  # added for checkpoint directories

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.hidden_dim = 256
        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mean_layer = nn.Linear(self.hidden_dim, out_dim)
        self.log_std = nn.Parameter(torch.ones(out_dim) * -0.5)
        self.activate = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, state: torch.Tensor):
        s = self.activate(self.fc1(state))
        s = self.activate(self.fc2(s))
        mean = self.mean_layer(s)               
        # Clamp log_std to prevent extreme values
        log_std_clamped = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = log_std_clamped.exp()
        base = Normal(mean, std)                 
        z = base.rsample()                       
        a = torch.tanh(z)                        
        # log_prob with tanh correction (per-dim)
        logp = base.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        ent = base.entropy()                    
        return a, logp, ent, mean, std

class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden_dim = 256
        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.activate = nn.Tanh()

        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        s = self.activate(self.fc1(state))
        s = self.activate(self.fc2(s))
        value = self.fc3(s)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    num_steps = len(rewards)
    gae_returns = [0] * num_steps  # Pre-allocate list

    for step in reversed(range(num_steps)):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        gae_returns[step] = gae + values[step]  # Assign value directly

    #############################
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

class RunningMeanStd:
    """Running mean and variance for observation normalization (OpenAI-style)."""
    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def update(self, x):
        x = np.array(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x, clip_range: float = 10.0):
        x = np.array(x, dtype=np.float64)
        std = np.sqrt(self.var + 1e-8)
        norm = (x - self.mean) / std
        return np.clip(norm, -clip_range, clip_range)

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
        # state normalization (OpenAI-style running mean/std)
        self.state_normalize = getattr(args, "state_normalize", False)
        if self.state_normalize:
            self.obs_rms = RunningMeanStd(env.observation_space.shape[0])
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
        # Learning rate scheduler (linear decay from 0 to 3M steps)
        self.max_training_steps = 3_000_000
        self.update_count = 0  # Track number of model updates
        lr_lambda = lambda it: max(0.0, 1.0 - min(it * self.rollout_len, self.max_training_steps) / self.max_training_steps)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda)

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
        # checkpoint / milestone setup
        self.checkpoint_dir = args.checkpoint_dir
        self.step_checkpoint_dir = os.path.join(self.checkpoint_dir, "step")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.step_checkpoint_dir, exist_ok=True)
        self.milestone_steps = [1_000_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000]
        self._saved_milestones = set()
        
        # Evaluation setup
        self.eval_interval = 10_000  # Every 10k steps
        self._last_eval_step = 0

    def _save_checkpoint(self, path: str):
        # Save only actor & critic weights plus state normalization if enabled
        actor_state = {k: v.cpu() for k, v in self.actor.state_dict().items()}
        critic_state = {k: v.cpu() for k, v in self.critic.state_dict().items()}
        
        checkpoint = {
            "actor_state_dict": actor_state,
            "critic_state_dict": critic_state,
            "total_step": self.total_step,
        }
        
        # Save state normalization parameters if enabled
        if self.state_normalize:
            checkpoint["obs_rms_mean"] = self.obs_rms.mean
            checkpoint["obs_rms_var"] = self.obs_rms.var
            checkpoint["obs_rms_count"] = self.obs_rms.count
            
        torch.save(checkpoint, path)
        tqdm.write(f"Saved checkpoint: {path}")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.state_normalize and (not self.is_test):
            try:
                self.obs_rms.update(state)
            except Exception:
                pass
        norm_state = self.obs_rms.normalize(state) if self.state_normalize else state
        state_tensor = torch.FloatTensor(norm_state).to(self.device)

        action, logp_perdim, ent_perdim, _, _ = self.actor(state_tensor)
        logp = logp_perdim.sum(dim=-1, keepdim=True)        # [B,1]
        ent  = ent_perdim.sum(dim=-1, keepdim=True)

        selected_action = action if not self.is_test else torch.tanh(self.actor(state_tensor)[3])  # test 時用 mean->tanh

        if not self.is_test:
            value = self.critic(state_tensor)
            self.states.append(state_tensor)
            self.actions.append(selected_action.detach())
            self.values.append(value.detach())
            self.log_probs.append(logp.detach())
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

    def _atanh(self, x, eps=1e-6):
        x = x.clamp(-1+eps, 1-eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def update_model(self, next_state: np.ndarray):
        if self.state_normalize:
            next_state = self.obs_rms.normalize(next_state)
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(next_value, self.rewards, self.masks, self.values, self.gamma, self.tau)

        states  = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)                        # tanh 後
        returns = torch.cat(returns).detach()
        values  = torch.cat(self.values).detach()
        log_probs_old = torch.cat(self.log_probs).detach()       # [B,1]

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses, entropy_losses = [], [], []
        for state, action, old_value, old_logp, return_, adv in ppo_iter(
            self.update_epoch, self.batch_size, states, actions, values, log_probs_old, returns, advantages
        ):
            # 重新算新 log_prob：z = atanh(a)
            _, _, ent_perdim, mean, std = self.actor(state)
            base = Normal(mean, std)
            z = self._atanh(action)
            logp_perdim = base.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            logp = logp_perdim.sum(dim=-1, keepdim=True)               # [B,1]
            ratio = (logp - old_logp).exp()

            # PPO-Clip actor loss（含 entropy）
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
            entropy = ent_perdim.sum(dim=-1, keepdim=True).mean()
            entropy_loss = self.entropy_weight * entropy  # weighted entropy loss
            actor_loss = -(torch.min(surr1, surr2).mean() + entropy_loss)

            # Critic：建議加 value clipping（更穩）
            v_pred = self.critic(state)
            v_pred_clip = old_value + (v_pred - old_value).clamp(-0.2, 0.2)
            critic_unclipped = F.mse_loss(v_pred, return_)
            critic_clipped   = F.mse_loss(v_pred_clip, return_)
            critic_loss = 0.5 * torch.max(critic_unclipped, critic_clipped)

            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []
        
        # Update learning rate schedulers
        self.update_count += 1
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return sum(actor_losses)/len(actor_losses), sum(critic_losses)/len(critic_losses), sum(entropy_losses)/len(entropy_losses)


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
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)

                # milestone step checkpoints
                if self.total_step in self.milestone_steps and self.total_step not in self._saved_milestones:
                    milestone_path = os.path.join(self.step_checkpoint_dir, f"checkpoint_step_{self.total_step}.pt")
                    self._save_checkpoint(milestone_path)
                    self._saved_milestones.add(self.total_step)

                state = next_state
                score += reward[0][0]

                # Check if it's time for evaluation (every 10k steps)
                if self.total_step - self._last_eval_step >= self.eval_interval:
                    print(f"\nRunning evaluation at step {self.total_step}...")
                    # Save current checkpoint for evaluation
                    eval_ckpt_path = os.path.join(self.checkpoint_dir, f"eval_checkpoint_step_{self.total_step}.pt")
                    self._save_checkpoint(eval_ckpt_path)
                    
                    # Run evaluation
                    try:
                        avg_reward, rewards = self.evaluate(start_seed=0, n_seeds=20, 
                                                           ckpt_path=eval_ckpt_path,
                                                           video_dir=f"videos/task3/eval_step_{self.total_step}")
                        print(f"Evaluation at step {self.total_step}: avg_reward = {avg_reward:.3f}")
                        
                        # Log evaluation results to wandb
                        wandb.log({
                            "eval_avg_reward": avg_reward,
                            "eval_step": self.total_step
                        })
                        
                        self._last_eval_step = self.total_step
                    except Exception as e:
                        print(f"Evaluation failed at step {self.total_step}: {e}")
                        # Continue training even if evaluation fails

                # Check if we've reached 3M+1 steps
                if self.total_step >= 3_000_001:
                    print(f"\nReached 3M+1 steps ({self.total_step}), ending training...")
                    # Save final checkpoint
                    final_ckpt_path = os.path.join(self.checkpoint_dir, f"final_checkpoint_step_{self.total_step}.pt")
                    self._save_checkpoint(final_ckpt_path)
                    # Close environment and return
                    self.env.close()
                    return

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset(seed=self.seed + episode_count)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    wandb.log({"score": score, "total_step": self.total_step})
                    # regular episode checkpoint
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{ep:04d}.pt")
                    self._save_checkpoint(ckpt_path)
                    score = 0

            actor_loss, critic_loss, entropy_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            # Log losses to wandb
            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy_loss": entropy_loss,
                "episode": episode_count,
                "actor_lr": self.actor_scheduler.get_last_lr()[0],
                "critic_lr": self.critic_scheduler.get_last_lr()[0],
                "total_step": self.total_step,
            })

        # termination
        self.env.close()

    def test(self, video_folder: str = None, seed: int = None) -> float:
        prev_mode, prev_seed = self.is_test, self.seed
        if seed is not None:
            self.seed = seed
        self.is_test = True
        video_folder = None 
        # 只有要錄影才要求 rgb_array，否則不要建 GL context
        render_mode = "rgb_array" if video_folder else None
        # render_mode = None
        test_env = gym.make("Walker2d-v4", render_mode=render_mode)

        if video_folder:
            try:
                test_env = gym.wrappers.RecordVideo(test_env, video_folder=video_folder)
            except Exception as e:
                print(f"RecordVideo init failed ({e}); falling back to no video.")
                # 重新建一個不渲染的環境
                test_env.close()
                test_env = gym.make("Walker2d-v4")  # no render_mode

        state, _ = test_env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)
        done = False
        score = 0.0

        while not done:
            action = self.select_action(state).reshape(self.action_dim,)
            next_state, reward, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            state = np.expand_dims(next_state, axis=0)
            score += reward

        test_env.close()
        self.is_test, self.seed = prev_mode, prev_seed
        return float(score)


    def evaluate(self, start_seed: int = 0, n_seeds: int = 21, ckpt_path: str = None, video_dir: str = "videos/task3") -> Tuple[float, list]:
        """Evaluate deterministic policy by loading actor/critic weights and testing multiple seeds."""
        if ckpt_path is None:
            raise ValueError("ckpt_path is required for evaluation")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        
        # Print checkpoint info
        total_step = ckpt.get("total_step", "unknown")
        print(f"Loaded checkpoint from step: {total_step}")
        
        # Load state normalization parameters if available and enabled
        if self.state_normalize and "obs_rms_mean" in ckpt:
            self.obs_rms.mean = ckpt["obs_rms_mean"]
            self.obs_rms.var = ckpt["obs_rms_var"] 
            self.obs_rms.count = ckpt["obs_rms_count"]
            print("Loaded state normalization parameters")

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
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=0.01) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2048)  
    parser.add_argument("--update-epoch", type=int, default=5)
    # state normalization (OpenAI-style)
    parser.add_argument("--state-normalize", action="store_true", dest="state_normalize", help="Enable running mean/std observation normalization (updates during training)")
    # checkpoint options
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/task3", help="Directory to save checkpoints")
    # evaluation options
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt to load for evaluation")
    parser.add_argument("--video-dir", type=str, default="./videos/task3", help="Directory to save evaluation videos")
    parser.add_argument("--eval-n-seeds", type=int, default=21, help="Number of seeds for evaluation")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v4", render_mode=None)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    if args.mode == "train":
        wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    else:
        # still init wandb in eval for consistency? optional: skip if not desired
        wandb.init(project="DLP-Lab7-PPO-Walker", name=f"eval-{args.wandb_run_name}", mode="disabled")
    
    agent = PPOAgent(env, args)

    if args.mode == "train":
        agent.train()
    else:
        avg, rewards = agent.evaluate(start_seed=args.seed, n_seeds=args.eval_n_seeds, ckpt_path=args.ckpt, video_dir=args.video_dir)
        print(f"Average reward over {len(rewards)} seeds: {avg:.3f}")
