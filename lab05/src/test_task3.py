import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
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

    def forward(self, x):
        return self.network(x)
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
        
        
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env = Pong3Action(gym.make("ALE/Pong-v5", render_mode="rgb_array",repeat_action_probability = 0.0))
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    avg_reward = 0.0
    
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            frame = env.render()
            frames.append(frame)

            state_tensor = torch.from_numpy(np.array(state)).float().div_(255.0).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
            frame_idx += 1

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        avg_reward += total_reward
    avg_reward /= args.episodes
    print(f"Average reward over seed {args.seed}: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation")
    args = parser.parse_args()
    evaluate(args)
