import torch
import numpy as np
import gymnasium as gym
import cv2
import imageio
import argparse
import os
from collections import deque
import ale_py

gym.register_envs(ale_py)  

class DQN(torch.nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions)
        )
    def forward(self, x):
        return self.network(x)

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized
    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)
    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
        with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
            for f in frames:
                video.append_data(f)
        print(f"Saved episode {ep} with total reward {total_reward} → {out_path}")
        avg_reward += total_reward
    avg_reward /= args.episodes
    print(f"Average reward over 20 episode: {avg_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task2")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)