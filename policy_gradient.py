import torch.nn as nn
from torch.autograd import Variable
import unicodedata
import string
import torch,ipdb
import torch.nn.functional as F
from envs.nn_env import NnEnv
import torch.optim as optim

import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np

class Policy(nn.Module):
    def __init__(self, num_hyperparams=4, hidden_size = 20, num_layers=3):
        super(Policy, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm = nn.LSTM(input_size=num_hyperparams, hidden_size=hidden_size, num_layers=num_layers)
        self.affine1 = nn.Linear(hidden_size, num_hyperparams)
        self.saved_actions = []
        self.rewards = []
        self.rewards_discounted = []

    
    def forward(self, input):
        output, (self.h, self.c) = self.lstm(input, (self.h, self.c))
        output = output.view(-1, output.size(2))
        output = self.affine1(output)
        return F.softmax(output)
    
    def initHidden(self):
        batch = 1
        self.h = Variable(torch.randn(self.num_layers, batch, self.hidden_size)) # (num_layers, batch, hidden_size)
        self.c = Variable(torch.randn(self.num_layers, batch, self.hidden_size))

def select_action(state):
    state = torch.from_numpy(state)
    state = state.float()

    probs = policy(Variable(state.resize_(1,1,4)))
    action = probs.multinomial()
    policy.saved_actions.append(action)
    return action.data[0,0]

def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)

    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    policy.rewards_discounted.extend(rewards)  
    del policy.rewards[:]

def train_policy():
    for action, r in zip(policy.saved_actions, policy.rewards_discounted):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards_discounted[:]
    del policy.saved_actions[:]

if __name__ == '__main__':
    # hyperparameters for rnn 
    n_hidden = 128
    num_hyperparams=4
    policy = Policy(num_hyperparams, n_hidden, num_hyperparams)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    gamma=0.99
    num_epochs=5


    # values for the architecture of cnn sampled
    num_layers=2
    num_hyperparams_per_layer=4
    num_episodes=2
    
    for epoch in range(num_epochs):
        for i in range(num_episodes):
            # create an environment
            env = NnEnv()
            policy.initHidden()
            observation = env.reset()
            done=False
            
            for t in range(10000): # Don't infinite loop while learning
                # forward through the rnn
                action = select_action(observation)
                print observation, action,
                observation, reward, done, info = env.step(action)
                policy.rewards.append(reward)
                print reward
                if done:
                    finish_episode()
                    break

        train_policy()      


    
   

