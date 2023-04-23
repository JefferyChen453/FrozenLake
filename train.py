import gymnasium as gym
import random
import numpy as np

# Based on Qlearning or SARSA
policy = "SARSA"

# Initialize the non-slippery Frozen Lake environment
environment = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")
environment.reset()
environment.render()

# Import matplotlib to plot the outcomes
import matplotlib.pyplot as plt
# plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})

# We re-initialize the Q-table
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hyperparameters
episodes = 1000       # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1.0          # Amount of randomness in the action selection
epsilon_decay = 0.001  # Fixed amount to decrease

# List of outcomes to plot
outcomes = []

# print('Q-table before training:')
# print(qtable)

# Training
for _ in range(episodes):
    state, _ = environment.reset()
    done = False
    action = np.argmax(qtable[state])
    
    # By default, we consider our outcome to be a failure
    outcomes.append("Failure")

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:
        # Generate a random number between 0 and 1
        rnd = np.random.random()
        
        if policy == "Qlearning":
            
            # If random number < epsilon, take a random action
            if rnd < epsilon:
                action = environment.action_space.sample()
            # Else, take the action with the highest value in the current state
            else:
                action = np.argmax(qtable[state])
                
            # Implement this action and move the agent in the desired direction
            state_new, reward, done, _, info = environment.step(action)

            # Update Q(s,a)
            qtable[state, action] = qtable[state, action] + \
                                    alpha * (reward + gamma * np.max(qtable[state_new]) - qtable[state, action])
            
            state = state_new
        
        
        elif policy == "SARSA":
            
            # action = np.argmax(qtable[state])
            state_new, reward, done, _, info = environment.step(action)
            # If random number < epsilon, take a random action
            if rnd < epsilon:
                action_new = environment.action_space.sample()
            # Else, take the action with the highest value in the current state
            else:
                action_new = np.argmax(qtable[state])

            # Update Q(s,a)
            qtable[state, action] = qtable[state, action] + \
                                    alpha * (reward + gamma * qtable[state_new][action_new] - qtable[state, action])

            state = state_new
            action = action_new


        # If we have a reward, it means that our outcome is a success
        if reward:
            outcomes[-1] = "Success"
        
    epsilon = max(epsilon - epsilon_decay, 0)

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

# Plot outcomes
plt.figure(figsize=(6, 3))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
plt.bar(range(len(outcomes)), outcomes, width=1.0)
plt.show()
