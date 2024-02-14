import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import imageio
import pathlib
import os

from matplotlib import animation
from trainagent import agent_cart

example_path = 'example_run'
if not os.path.exists(example_path):
    pathlib.Path(example_path).mkdir(parents=True, exist_ok=True)

frames = []
env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset(seed=7896)
agent_cart.policy_net.eval()
state, info = env.reset()
state = torch.tensor(state)
done = False
while not done:
    frames.append(env.render())
    action = torch.argmax(agent_cart.policy_net(state)).item()
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = torch.tensor(next_state)
env.close()

imageio.mimsave(os.path.join(example_path, 'run800.gif'), frames, fps=30)
