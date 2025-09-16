import torch
import numpy as np
import gymnasium as gym
import argparse
import os
import imageio

class DQN(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions)
        )
    def forward(self, x):
        return self.network(x)
    
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    num_actions = env.action_space.n

    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    avg_reward = 0.0

    for ep in range(args.episodes):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Episode {ep} total reward: {total_reward}")
        avg_reward += total_reward
    avg_reward /= args.episodes
    print(f"Average reward over 20 episode: {avg_reward:.2f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)