import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import deepgrid as dg
from deepgrid.colors import *
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

    def loss(self, dists:torch.tensor, actions, weights:torch.tensor):
        #print(green, dists, endc)
        #print(blue, actions, endc)
        #print(yellow, weights, endc)
        probs = torch.sum(dists*actions, axis=-1)
        logprobs = torch.log(probs)
        #print(underline, probs, endc)
        #print(bold, logprobs.shape, weights.shape, endc)
        eploss = torch.sum(logprobs*weights, axis=1)
        loss = -torch.mean(eploss)
        return loss

    def train(self, states, actions, weights):
        out = self.forward(states)
        dists = out.reshape(actions.shape)
        los = self.loss(dists, actions, weights)
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


class vpgAgent(dg.agent):
    def __init__(self, env, stepCost=0, actions=4):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.policy = model(self.env.size, 4)
        self.states = []
        self.actions = []
        self.weights = []

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        dist = torch.flatten(self.policy(state))
        if greedy: action = np.argmax(dist.detach().numpy())
        else: action = dg.sampleDist(dist.detach().numpy())
        return action, dist

    #NOTE: for a policy agent, remember should only be called AFTER and episode, not during.
    def remember(self, states, actions, weights):
        self.states.append(states)
        self.actions.append(actions)
        self.weights.append(weights)

    def train(self):
        sh = (3, self.env.size[1], self.env.size[0])
        states = torch.Tensor(np.float32(self.states)).reshape(-1, *sh)
        actions = torch.Tensor(np.float32(self.actions))
        weights = torch.Tensor(np.float32(self.weights))
        return self.policy.train(states, actions, weights)

    def save(self, path, name):
        self.policy.save(path, name)
    def load(self, path):
        sd = torch.load(path)
        self.policy.load_state_dict(sd)

    def reset(self):
        s = self.score
        self.score = 0
        return s

    def forget(self):
        self.states = []
        self.actions = []
        self.weights = []




