import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from agent import *
import os, time

class model(nn.Module):
    def __init__(self, gridSize, actions, lr=.001):
        super(model, self).__init__()
        self.gridSize, self.actions, self.lr = gridSize, actions, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.lin1 = nn.Linear(32*height*width, 512)
        self.lin2 = nn.Linear(512, 64)
        self.lin3 = nn.Linear(64, self.actions)

        #self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, X):
        sh = X.shape
        #assert len(sh)==4, f"got input tensor shape {sh}. Should be length 4: (batch, channels, width, height)"
        if len(sh) == 3: X = X.reshape(1, *sh)
        X = F.leaky_relu(self.conv1(X))
        X = F.leaky_relu(self.conv2(X))
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.lin1(X))
        X = F.relu(self.lin2(X))
        X = F.softmax(self.lin3(X), dim=1)
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, dists:torch.tensor, actions, rewards:torch.tensor):
        #print(green, dists, endc)
        #print(blue, actions, endc)
        #print(yellow, rewards, endc)
        probs = torch.sum(dists*actions, axis=-1)
        logprobs = torch.log(probs)
        #print(underline, probs, endc)
        #print(bold, logprobs.shape, rewards.shape, endc)
        eploss = torch.sum(logprobs*rewards, axis=1)
        loss = -torch.mean(eploss)
        return loss

    def train(self, states, actions, rewards):
        out = self.forward(states)
        dists = out.reshape(actions.shape)
        los = self.loss(dists, actions, rewards)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return los

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")

    def load(self, path):
        self.load_state_dict(torch.load(path))


##########################################################################


class spoAgent(agent):
    def __init__(self, env, stepCost=0, actions=4):
        self.env = env
        self.score = 0
        self.numActions = actions
        self.stepCost = stepCost
        self.policy = model(self.env.size, 4)
        self.states = [[]]
        self.actions = [[]]
        self.rewards = [[]]

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        dist = torch.flatten(self.policy(state))
        if greedy: np.argmax(dist.detach().numpy())
        else: action = sampleDist(dist.detach().numpy())
        return action, dist

    def remember(self, state, action, reward):
        self.states[-1].append(state)
        self.actions[-1].append(action)
        self.rewards[-1] = [e + reward for e in self.rewards[-1]]
        self.rewards[-1].append(reward)

    def train(self):
        sh = (3, self.env.size[1], self.env.size[0])
        states = torch.Tensor(self.states).reshape(-1, *sh)
        actions = torch.Tensor(self.actions)
        rewards = torch.Tensor(self.rewards)
        return self.policy.train(states, actions, rewards)

    def save(self, path, name):
        self.policy.save(path, name)
    def load(self, path):
        sd = torch.load(path)
        self.policy.load_state_dict(sd)

    def reset(self):
        s = self.score
        self.score = 0
        self.states.append([])
        self.actions.append([])
        self.rewards.append([])
        return s

    def forget(self):
        self.states = []
        self.actions = []
        self.rewards = []




