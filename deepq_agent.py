import numpy as np
from tinygrad.nn import Tensor
from tinygrad import nn
import os
from utils import *

class model:
    def __init__(self, gridSize, actions, lr=.001):
        self.gridSize, self.actions, self.lr = gridSize, actions, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.lin1 = nn.Linear(32*height*width, 512)
        self.lin2 = nn.Linear(512, 64)
        self.lin3 = nn.Linear(64, self.actions)
        #self.layers = [self.conv1.weight, self.conv2.weight, self.lin1.weight, self.lin2.weight, self.lin3.weight]
        self.layers = [self.conv1, self.conv2, self.lin1, self.lin2, self.lin3]
        self.names = ["conv1", "conv2", "lin1", "lin2", "lin3"]

        self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)

    def forward(self, X: Tensor):
        sh = X.shape
        assert len(sh)==4, f"got input tensor shape {sh}. Should be length 4: (batch, channels, width, height)"
        #X = self.conv1(X).softmax()
        #X = self.conv2(X).softmax()
        X = self.conv1(X).leakyrelu()
        X = self.conv2(X).leakyrelu()
        X = X.reshape(sh[0], -1)
        X = self.lin1(X).sigmoid()
        X = self.lin2(X).sigmoid()
        X = self.lin3(X)
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, experience, out, discount=1):
        states, actions, rewards, nstates, terminal = experience
        nextout = discount*self.forward(nstates).max(axis=1)
        trueQ = rewards + (terminal-1).abs()*nextout
        predQ = (out*actions).sum(axis=1)
        loss = (predQ - trueQ).pow(2).mean()
        return loss

    def train(self, experience, discount=1.0):
        states, actions, rewards, nstates, terminals = experience
        out = self.forward(states)
        los = self.loss(experience, out, discount=discount)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return out, los

    def copy(self, other):
        for i, layer in enumerate(self.layers):
            layer.weight.assign(other.layers[i].weight.detach())

    def save(self, path):
        for i, name in enumerate(self.names):
            np.save(f"{path}\\{name}.npy", self.layers[i].weight.numpy())

    def load(self, path):
        for name in self.names:
            s = np.load(f"{path}\\{name}.npy")
            layer = self.__getattribute__(name)
            layer.weight.assign(s)
##########################################################################

class agent:
    def __init__(self, env, stepCost=1, actions=4):
        self.env = env
        self.score = 0
        self.gamma = 1
        self.actions = actions
        self.stepCost = stepCost
        self.eps = 1
        self.decayRate = 0.999999
        # states, actions, rewards, and next states stored in separate lists for sampling
        self.memory = [[] for i in range(5)]
        self.main = model(self.env.size, 4)
        self.target = model(self.env.size, 4)
        self.update()

    def donothing(): return

    def reset(self):
        s = self.score
        self.score = 0
        self.update()
        return s

    def doUserAction(self):
        amap = {"w": 0, "a":1, "s":2, "d":3}
        cmd = input("select an action [w, a, s, d]:\n")
        while cmd not in amap:
            cmd = input(f"{red}not a proper action.{endc} select an action [w, a, s, d]:\n")
        reward = self.doAction(amap[cmd])
        print(f"taking action {yellow}{cmd}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{self.score}{endc} on step {self.env.stepsTaken}/{self.env.maxSteps}")
        return reward

    def chooseAction(self, state, eps=None):
        if eps is None:
            eps = self.eps
            self.eps *= self.decayRate
        r = np.random.uniform()
        if r <= eps:
            pred = np.zeros((4))
            return self.randomAction(), pred, True
        else:
            if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
            else: st = state.reshape((1, *state.shape))
            pred = self.main(st).numpy()
            action = np.argmax(pred)
            return action, pred, False

    def doAction(self, action, store=True):
        if store: s = self.env.observe()
        reward = self.env.doAction(action) - self.stepCost
        if store:
            action_hot = np.zeros((self.actions), np.float32)
            action_hot[action] = 1
            experience = (s, action_hot, reward, self.env.observe(), 1*(self.env.stepsTaken==self.env.maxSteps))
            self.remember(experience)
        self.score += reward
        return reward

    def remember(self, experience):
        for i in range(5):
            self.memory[i].append(experience[i])

    def doRandomAction(self):
        return self.doAction(self.randomAction())
    def randomAction(self):
        return np.random.randint(0,self.actions)

    def sampleMemory(self, num, tensor=True):
        assert len(self.memory[1]) > num, f"requested sample size greater than number of recorded experiences"
        samp = np.random.randint(0, len(self.memory[0]), size=(num))
        expSample = [[], [], [], [], []]
        for s in samp:
            for i in range(len(self.memory)):
                expSample[i].append(self.memory[i][s])
        if tensor:
            for i in range(len(expSample)): expSample[i] = Tensor(expSample[i])
        return tuple(expSample)

    def train(self, experience):
        return self.target.train(experience, discount=self.gamma)

    def update(self):
        self.main.copy(self.target)

    def save(self, path):
        self.target.save(path)
    def load(self, path):
        self.target.load(path)
        self.update()







