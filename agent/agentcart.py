import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensordict
import numpy as np

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from agent.net import DQN


class Agent_CartPole:

    def __init__(self, state_dim, action_dim, intial_exploration, final_exploration, final_exploration_frame, size_memory, batch_size, gamma, tau, learning_rate):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize networks
        self.policy_net = DQN(self.state_dim, self.action_dim)
        self.target_net = DQN(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Exploration parameters
        self.intial_exploration =  intial_exploration
        self.final_exploration =  final_exploration
        self.final_exploration_frame =  final_exploration_frame
        self.eps_threshold = self.intial_exploration
        self.exploration_decrement = (self.intial_exploration - self.final_exploration) / self.final_exploration_frame

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(size_memory))
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad=True)

    # Given a state the agent chooses an action based on an epsilon-greedy policy
    ############################################################################
    def agent_act(self, state):

        self.eps_threshold = max(self.final_exploration, self.eps_threshold )

        sample = np.random.rand()

        # Exploit
        if sample > self.eps_threshold:
            action_ind = torch.argmax(self.policy_net(state)).item()

        # Explore
        else:
            action_ind =  np.random.randint(self.action_dim)

        # Decrement exploration_threshold
        self.eps_threshold -= self.exploration_decrement

        return action_ind
      ############################################################################

    # Add experience to memory
    ############################################################################
    def agent_cache(self, state, action, next_state, reward, non_final):

        state = torch.tensor(state)
        action = torch.tensor([action])
        next_state = torch.tensor(next_state)
        reward = torch.tensor([reward])

        self.memory.add(TensorDict({"state" : state, "action" : action, "next_state" : next_state,"reward" : reward, "non_final" : non_final}, batch_size=[]))
    ############################################################################

    # Update policy network
    ############################################################################
    def agent_update(self):

        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch = self.memory.sample(self.batch_size)

        # Calculate state action values at current state
        action_batch = batch['action'].reshape(self.batch_size,1)
        state_action_batch = self.policy_net(batch['state']).gather(1, action_batch)

        # Compute state action values for new state
        next_state_action_batch = torch.zeros(self.batch_size)
        non_final_batch = batch['non_final'].squeeze()
        with torch.no_grad():
            max_values =  torch.max(self.target_net(batch['next_state']),1).values
            next_state_action_batch[non_final_batch] = max_values[non_final_batch]

        # Expected state action values for current state
        exp_state_action_batch = batch['reward'].squeeze() + (self.gamma * next_state_action_batch)

        # Compute Huber loss
        loss = self.loss_fn(state_action_batch, exp_state_action_batch.reshape(self.batch_size, 1))

        # Reset gradients
        self.optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Update parameters
        self.optimizer.step()
    ############################################################################

    # Update target network
    ############################################################################
    def update_target(self):

        self.target_net_state_dict = self.target_net.state_dict()
        self.policy_net_state_dict = self.policy_net.state_dict()
        for i in self.policy_net_state_dict:
            self.target_net_state_dict[i] = self.policy_net_state_dict[i]*self.tau + self.target_net_state_dict[i]*(1 - self.tau)
        self.target_net.load_state_dict(self.target_net_state_dict)
    ############################################################################



# class Agent_CartPole:

#     def __init__(self, state_dim, action_dim, intial_exploration, final_exploration, final_exploration_frame):

#         self.state_dim = state_dim
#         self.action_dim = action_dim

#         # Initialize networks
#         self.policy_net = DQN(self.state_dim, self.action_dim)
#         self.target_net = DQN(self.state_dim, self.action_dim)
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#         # Setup exploration parameters
#         self.intial_exploration =  intial_exploration
#         self.final_exploration =  final_exploration
#         self.final_exploration_frame =  final_exploration_frame
#         self.eps_threshold = self.intial_exploration
#         self.exploration_decrement = (self.intial_exploration - self.final_exploration) / self.final_exploration_frame

#     # Given a state the agent chooses an action based on an epsilon-greedy policy.
#     def agent_act(self, state):

#         self.eps_threshold = max(self.final_exploration, self.eps_threshold )

#         sample = np.random.rand()

#         #Exploit
#         if sample > self.eps_threshold:
#             action_ind = torch.argmax(self.policy_net(state)).item()

#         #Explore
#         else:
#             action_ind =  np.random.randint(self.action_dim)

#         #Decrement exploration_threshold
#         self.eps_threshold -= self.exploration_decrement

#         return action_ind
    
    
    
# class Agent_CartPole(Agent_CartPole):

#     def __init__(self, state_dim, action_dim, intial_exploration, final_exploration, final_exploration_frame, size_memory):
#         super().__init__(state_dim, action_dim, intial_exploration, final_exploration, final_exploration_frame)
#         self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(size_memory))

#     # Adding experience to memory
#     def agent_cache(self, state, action, next_state, reward, non_final):

#         state = torch.tensor(state)
#         action = torch.tensor([action])
#         next_state = torch.tensor(next_state)
#         reward = torch.tensor([reward])

#         self.memory.add(TensorDict({"state" : state, "action" : action, "next_state" : next_state,"reward" : reward, "non_final" : non_final}, batch_size=[]))



# class Agent_CartPole(Agent_CartPole):

#     def __init__(self, state_dim, action_dim, intial_exploration, final_exploration, final_exploration_frame, size_memory, batch_size, gamma, tau, learning_rate):
#         super().__init__(state_dim, action_dim, intial_exploration, final_exploration, final_exploration_frame, size_memory)
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.tau = tau
#         self.loss_fn = nn.SmoothL1Loss()
#         self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad=True)

#     def agent_update(self):

#         if len(self.memory) < self.batch_size:
#             return

#         #Sample a batch
#         batch = self.memory.sample(self.batch_size)

#         #Calculating state action values at current state
#         action_batch = batch['action'].reshape(self.batch_size,1)
#         state_action_batch = self.policy_net(batch['state']).gather(1, action_batch)

#         #Compute state action values for new state
#         next_state_action_batch = torch.zeros(self.batch_size)
#         non_final_batch = batch['non_final'].squeeze()
#         with torch.no_grad():
#             max_values =  torch.max(self.target_net(batch['next_state']),1).values
#             next_state_action_batch[non_final_batch] = max_values[non_final_batch]

#         #Expected state action values for current state
#         exp_state_action_batch = batch['reward'].squeeze() + (self.gamma * next_state_action_batch)

#         #Compute Huber loss
#         loss = self.loss_fn(state_action_batch, exp_state_action_batch.reshape(self.batch_size, 1))

#         #Reset gradients
#         self.optimizer.zero_grad()

#         #Compute gradients
#         loss.backward()

#         #Update parameters
#         self.optimizer.step()

#     def update_target(self):

#         self.target_net_state_dict = self.target_net.state_dict()
#         self.policy_net_state_dict = self.policy_net.state_dict()
#         for i in self.policy_net_state_dict:
#             self.target_net_state_dict[i] = self.policy_net_state_dict[i]*self.tau + self.target_net_state_dict[i]*(1 - self.tau)
#         self.target_net.load_state_dict(self.target_net_state_dict)

