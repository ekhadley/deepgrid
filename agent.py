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
        self.layers = [self.conv1.weight, self.conv2.weight, self.lin1.weight, self.lin2.weight, self.lin3.weight]
        
        self.opt = nn.optim.SGD(self.layers, lr=self.lr)

    def forward(self, X: Tensor):
        sh = X.shape
        assert len(sh)==4, f"got input tensor shape {sh}. Should be length 4: (batch, channels, width, height)"
        X = self.conv1(X).softmax()
        X = self.conv2(X).softmax()
        X = X.reshape(sh[0], -1)
        X = self.lin1(X)
        X = self.lin2(X)
        X = self.lin3(X)
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, experience, out, discount=0.9):
        states, actions, rewards, nstates, terminal = experience
        trueq = rewards if terminal else rewards + discount*self.forward(nstates).max(axis=1)
        mask = np.zeros(out.shape, np.float32)
        for i, a in enumerate(actions): mask[i, a] = 1
        mask = Tensor(mask)
        loss = ((out*mask).sum(axis=1) - trueq).pow(2).mean()
        return loss

    def train(self, experience, discount=0.9):
        states, actions, rewards, nstates, terminals = experience
        out = self.forward(states)
        los = self.loss(experience, out, discount=discount)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return out, los

    def copy(self, other):
        self.conv1.weight.assign(other.conv1.weight.detach())
        self.conv2.weight.assign(other.conv2.weight.detach())
        self.lin1.weight.assign(other.lin1.weight.detach())
        self.lin2.weight.assign(other.lin2.weight.detach())
        self.lin3.weight.assign(other.lin3.weight.detach())
        #self.layers = [self.conv1, self.conv2, self.lin1, self.lin2]

    def save(self, path):
        np.save(f"{path}\\conv1.npy", self.conv1.weight.numpy())
        np.save(f"{path}\\conv2.npy", self.conv2.weight.numpy())
        np.save(f"{path}\\lin1.npy", self.lin1.weight.numpy())
        np.save(f"{path}\\lin2.npy", self.lin2.weight.numpy())
        np.save(f"{path}\\lin3.npy", self.lin3.weight.numpy())

    def load(self, path):
        self.conv1.weight.assign(Tensor(np.load(f"{path}\\conv1.npy")))
        self.conv2.weight.assign(Tensor(np.load(f"{path}\\conv2.npy")))
        self.lin1.weight.assign(Tensor(np.load(f"{path}\\lin1.npy")))
        self.lin2.weight.assign(Tensor(np.load(f"{path}\\lin2.npy")))
        self.lin3.weight.assign(Tensor(np.load(f"{path}\\lin3.npy")))

##########################################################################

class agent:
    def __init__(self, env, stepCost=1, actions=4):
        self.env = env
        self.score = 0
        self.gam = 0.95
        self.actions = actions
        self.stepCost = stepCost
        self.eps = 1
        self.decayRate = 0.99995
        # states, actions, rewards, and next states stored in separate lists for sampling
        self.memory = [[] for i in range(5)]
        self.main = model(self.env.size, 4)
        self.target = model(self.env.size, 4)
        self.update()

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

    def chooseAction(self, state, eps=None, givePred=False):
        if eps is None:
            eps = self.eps
            self.eps *= self.decayRate
        r = np.random.uniform()
        if r <= eps: action = self.randomAction()
        else:
            st = Tensor(state).reshape((1, *state.shape))
            pred = self.main(st).numpy()
            action = np.argmax(pred)
            if givePred: return action, pred
        return action

    def doAction(self, action, store=True):
        if store: s = self.env.observe()
        reward = self.env.doAction(action) - self.stepCost
        if store:
            experience = (s, action, reward, self.env.observe(), 1*(self.env.stepsTaken==self.env.maxSteps))
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
            expSample[0] = Tensor(expSample[0])
            expSample[3] = Tensor(expSample[3])
        return tuple(expSample)

    def train(self, experience):
        return self.target.train(experience, discount=self.gam)

    def update(self):
        self.main.copy(self.target)

    def save(self, path):
        self.target.save(path)
    def load(self, path):
        self.target.load(path)
        self.update()







