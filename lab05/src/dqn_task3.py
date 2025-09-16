# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym 
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0 

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        if error is None:
            priority = self.max_priority
        else:
            priority = (abs(error) + self.eps) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.max_priority = max(self.max_priority, priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        current_len = len(self.buffer)
        prios = self.priorities[:current_len]

        prios = np.maximum(prios, self.eps)
        prob_sum = prios.sum()
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            probs = np.ones(current_len, dtype=np.float32) / current_len
        else:
            probs = prios / prob_sum

        indices = np.random.choice(current_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # IS weights
        weights = (current_len * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, errors):
        for idx, e in zip(indices, errors):
            p = (abs(e) + self.eps) ** self.alpha
            self.priorities[idx] = p
            self.max_priority = max(self.max_priority, p)

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.prev_frame = None

    def _gray84(self, obs):
        g = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        cropped = g[34:194, :]       # 160x160
        return cv2.resize(cropped, (84,84), interpolation=cv2.INTER_AREA)

    def preprocess(self, obs):
        cur = self._gray84(obs)
        if self.prev_frame is None:
            self.prev_frame = cur
        m = np.maximum(cur, self.prev_frame)  # Max over last two frames
        self.prev_frame = cur
        return m

    def reset(self, obs):
        self.prev_frame = None
        f = self.preprocess(obs)
        self.frames = deque([f for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        f = self.preprocess(obs)
        self.frames.append(f)
        return np.stack(self.frames, axis=0)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.args = args
        
        # action wrapper
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Multi-Step Return 
        self.n_step = args.n_step

        # criterion
        self.criterion = nn.SmoothL1Loss(reduction="none")

        # PER beta annealing
        if args.PER:
            self.per_beta_start = 0.4
            self.per_beta_frames = 500000
        
        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        if self.args.PER:
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4)
        else:
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir

        # Epsilon decay
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_decay_rate = (self.epsilon_min / self.epsilon_start) ** (1 / self.epsilon_decay_steps)

        os.makedirs(self.save_dir, exist_ok=True)

        if args.PER:
            self.memory = PrioritizedReplayBuffer(args.memory_size)
        else:
            self.memory = deque(maxlen=args.memory_size)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        # divide by 255.0 to normalize the state
        state_tensor = torch.from_numpy(np.array(state)).float().div_(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()
    
    def _push_transition(self, s, a, r, ns, d):
        if self.args.PER:
            self.memory.add((s, a, r, ns, d), error=None)
        else:
            self.memory.append((s, a, r, ns, d))

            
    def run(self, episodes=4000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0
            
            if hasattr(self.args, 'max_env_steps') and self.env_count >= self.args.max_env_steps:
                print(f"Reached maximum environment steps ({self.args.max_env_steps}). Stopping training.")
                break
                
            # Initialize n-step buffer if using Multi-Step Return
            if self.args.MSR:
                nstep_buffer = deque(maxlen=self.n_step)

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                
                if self.args.MSR:
                    nstep_buffer.append((state, action, reward, next_state, done))

                if self.args.MSR and len(nstep_buffer) == self.n_step:
                    # calculate n-step return
                    R = sum([nstep_buffer[i][2] * (self.gamma ** i) for i in range(self.n_step)])
                    s, a, _, _, _ = nstep_buffer[0]
                    ns, _, _, _, d = nstep_buffer[-1]
                    self._push_transition(s, a, R, ns, d)
                elif not self.args.MSR:
                    self._push_transition(state, action, reward, next_state, done)
                    
                for _ in range(self.train_per_step):
                    self.train()
                 
                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                # if self.env_count % self.target_update_frequency == 0:
                #     self.target_net.load_state_dict(self.q_net.state_dict())
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
                    
            if self.args.MSR:
                # process remaining n-step transitions  
                while len(nstep_buffer) > 0:
                    R = sum((self.gamma ** i) * nstep_buffer[i][2] for i in range(len(nstep_buffer)))
                    s, a = nstep_buffer[0][0], nstep_buffer[0][1]
                    ns, d = nstep_buffer[-1][3], nstep_buffer[-1][4]
                    self._push_transition(s, a, R, ns, d)
                    nstep_buffer.popleft()
                    
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 3 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 3 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            # divide by 255.0 to normalize the state
            state_tensor = torch.from_numpy(np.array(state)).float().div_(255.0).unsqueeze(0).to(self.device)
           
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Exponential decay for epsilon based on env_count
        if self.env_count <= self.epsilon_decay_steps:
            self.epsilon = self.epsilon_start * (self.epsilon_decay_rate ** self.env_count)
            # 確保不會低於最小值（由於浮點數精度問題）
            self.epsilon = max(self.epsilon, self.epsilon_min)
        else:
            self.epsilon = self.epsilon_min
            
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        if self.args.PER:
            t = min(1.0, self.train_count / self.per_beta_frames)
            self.memory.beta = self.per_beta_start + t * (1.0 - self.per_beta_start)
            
        if self.args.PER:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            # weights = np.ones(self.batch_size, dtype=np.float32)  # Uniform weights for non-PER
        
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        
        # Convert states and next_states to float32 and normalize by dividing by 255.0
        states = torch.from_numpy(np.array(states).astype(np.float32)).div_(255.0).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).div_(255.0).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates

        with torch.no_grad():
            if self.args.DDQN:
                next_actions = self.q_net(next_states).argmax(1)
                target_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                target_q_values = self.target_net(next_states).max(1)[0]

        if self.args.MSR:
            gamma = self.gamma ** self.n_step
        else:
            gamma = self.gamma
        expected_q_values = rewards + (1 - dones) * gamma * target_q_values

        td_errors = q_values - expected_q_values 
    
        per_sample_loss = self.criterion(q_values, expected_q_values)  # shape [B]
        
        if self.args.PER:
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            loss = (per_sample_loss * weights_tensor).mean()
            self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

        # monitor train loss and Q value statistics
        wandb.log({
            "Train Loss": loss.item(),
            "Q Mean": q_values.mean().item(),
            "Q Std": q_values.std().item()
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results/task2")
    parser.add_argument("--wandb-run-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999977)
    parser.add_argument("--epsilon-min", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=int, default=900000, help="Number of env steps for linear epsilon decay")
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    
    # maximum environment steps
    parser.add_argument("--max-env-steps", type=int, default=100000000, help="Maximum total environment steps")
    
    # enhancements for Task 3
    parser.add_argument("--DDQN", action="store_true", help="Enable Double DQN")
    parser.add_argument("--PER", action="store_true", help="Enable Prioritized Experience Replay")
    parser.add_argument("--MSR", action="store_true", help="Enable Multi-Step Return")
    parser.add_argument("--n-step", type=int, default=3, help="Number of steps for Multi-Step Return")
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong-exDQN", name=args.wandb_run_name, save_code=True, config=args)
    agent = DQNAgent(args=args)
    agent.run()