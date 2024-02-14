import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensordict
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random

from agent.net import DQN
from agent.agentcart import Agent_CartPole
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from itertools import count

# Setup game
env = gym.make("CartPole-v1")

# Set random number seeds
def set_random_seeds(env, seed):
      torch.manual_seed(seed)  # PyTorch
      np.random.seed(seed)  # NumPy
set_random_seeds(env, seed = 798)

# Initialize agent
agent_cart = Agent_CartPole(state_dim = 4,
                            action_dim = 2,
                            intial_exploration = 1,
                            final_exploration = 0.01,
                            final_exploration_frame = 10000,
                            size_memory = 10000,
                            batch_size = 128,
                            gamma = 0.99,
                            tau = 0.005,
                            learning_rate = 0.0001)

# Train agent
num_episodes = 800
durations_episodes = []
for i_episode in range(num_episodes):

    # Initialize environment
    state, info = env.reset(seed=42)

    for t in count():

      state = torch.tensor(state)
      action = agent_cart.agent_act(state)
      next_state, reward, terminated, truncated, _ = env.step(action)
      is_done = terminated or truncated
      non_final = not is_done

      # Store transition pair
      agent_cart.agent_cache(state, action, next_state, reward, non_final)

      # Update model with experience
      agent_cart.agent_update()

      # Update state
      state = next_state

      # Update target network parameters (using soft update)
      agent_cart.update_target()
      
      if is_done:
        durations_episodes.append(t + 1)
        break 

# Plot result
plt.title('Return per episode of training')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.plot(np.arange(len(durations_episodes)) + 1, durations_episodes)
