# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# EDITED BY: Merlijn Sevenhuijsen  | merlijns@kth.se   | 200104073275
# EDITED BY: Hugo Westergård       | hugwes@kth.se     | 200011289659

# Load packages
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
from collections import deque, namedtuple
from DQN_NN import NeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)

def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def exp_decay(eps_min, eps_max, Z, k):
    """ Function used to compute the epsilon value used for epsilon-greedy
        exploration
    """
    epsilon_k = max(eps_min, eps_max * (eps_min/eps_max) ** ((k-1)/(Z-1)))
    return epsilon_k

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters for the episodic Deep Q Network algorithm with epsilon-greedy exploration
N_episodes = 400                                # Number of episodes
discount_factor = 0.99                          # Value of the discount factor
n_ep_running_average = 50                       # Running average of 50 episodes
n_actions = env.action_space.n                  # Number of available actions
dim_state = len(env.observation_space.high)     # State dimensionality
BUFFER_size = 109000                             # Maximum number of experiences we are storing
BATCH_SIZE = 64                                # Number of experiences to sample in each training step
EPSILON_MAX = 0.99                              # Initial value of epsilon in epsilon-greedy
EPSILON_MIN = 0.05                              # Minimum value of epsilon
TARGET_UPDATE = int(BUFFER_size / BATCH_SIZE)   # How often to update the target network
                                                # update target network
CLIPPING_VALUE = 1.                              # Gradient clipping value
LEARNING_RATE = 0.0005                          # Learning rate
size_layers_hidden = 64                         # Number of hidden neurons
Z = 0.9 * N_episodes                            # Number of episodes for epsilon to decay to eps_min

# Index for counting when to update the target network

Experience = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

buffer = ExperienceReplayBuffer(maximum_length=BUFFER_size)
network = NeuralNetwork(dim_state, n_actions, size_layers_hidden)
network_target = NeuralNetwork(dim_state, n_actions, size_layers_hidden)
network.load_state_dict(network_target.state_dict())

# Optimal network found so far
opt_network = NeuralNetwork(dim_state, n_actions, size_layers_hidden)

# Adam optimizer with learning rate
optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE)

# Index for the episode
target_update_counter = 0

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    state = env.reset()                    # Reset environment, returns
    total_episode_reward = 0.              # initial state
    t = 0
    done = False                           # Boolean variable used to indicate
                                           # if an episode terminated
    # epsilon exponentially decayed
    epsilon = exp_decay(EPSILON_MIN, EPSILON_MAX, Z, i)

    while not done:
        # env.render()                       # Render the environment, remove this
                                           # line if you run on Google Colab
        # Create state tensor, remember to use single precision (torch.float32)
        state_tensor = torch.tensor([state], requires_grad=False, dtype=torch.float32)

        # Compute output of the network
        values = network(state_tensor)

        # With probability epsilon select a random action, otherwise select the action
        # with highest Q-value
        if np.random.rand() < epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = values.max(1)[1].item()

        # Take an action and observe the new state and reward
        next_state, reward, done, _ = env.step(action)

        # Append experience to the buffer
        exp = Experience(state, action, reward, next_state, done)
        buffer.append(exp)

        ### TRAINING ###
        # Perform training if we have more than BATCH_SIZE elements in the buffer
        if len(buffer) >= BATCH_SIZE:
            # Sample a batch of 3 elements
            states, actions, rewards, next_states, done_2 = buffer.sample_batch(n=BATCH_SIZE)

            # Convert everything to tensors
            states = torch.tensor(states, requires_grad=False, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, requires_grad=False, dtype=torch.float32)
            done_2 = torch.tensor(done_2, dtype=torch.int32)

            # Get the values of our network for the current states
            values = network(states)
            values = values.gather(1, actions.view(-1,1))

            # Compute the target values
            target_values = rewards + (1 - done_2) * discount_factor * network_target(next_states).max(1)[0]
            target_values = target_values.view(-1, 1)
            # Compute the loss
            loss = nn.MSELoss()(target_values, values)

            # Perform backward pass (backpropagation)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), CLIPPING_VALUE)

            # Update the network's parameters
            optimizer.step()

        # Update state
        state = next_state
        total_episode_reward += reward
        t += 1
        target_update_counter += 1

        # Update the target network every TARGET_UPDATE steps
        if target_update_counter  == TARGET_UPDATE:
            network_target.load_state_dict(network.state_dict())
            target_update_counter = 0

    # If the best network so far is found save it, if the episoderewardlist is not empty
    if len(episode_reward_list) != 0 and total_episode_reward > max(episode_reward_list):
        opt_network.load_state_dict(network.state_dict())

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))

# Save the optimal network
torch.save(opt_network, 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
