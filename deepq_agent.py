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
        self.layers = [self.conv1, self.conv2, self.lin1, self.lin2, self.lin3]
        self.names = ["conv1", "conv2", "lin1", "lin2", "lin3"]

        #self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)
        self.opt = nn.optim.AdamW([layer.weight for layer in self.layers], lr=self.lr)

    def forward(self, X: Tensor):
        sh = X.shape
        assert len(sh)==4, f"got input tensor shape {sh}. Should be length 4: (batch, channels, width, height)"
        #X = self.conv1(X).softmax()
        #X = self.conv2(X).softmax()
        X = self.conv1(X).leakyrelu()
        X = self.conv2(X).leakyrelu()
        X = X.reshape(sh[0], -1)
        X = self.lin1(X).relu()
        X = self.lin2(X).relu()
        X = self.lin3(X)
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, experience, out:Tensor, discount=1, debug=False):
        states, actions, rewards, nstates, terminal = experience
        #states, actions, rewards, nstates, terminal, statePred, nextPred_ = experience
        #if np.isnan(nextPred_.numpy()).any():
        #    nextpred = self.forward(nstates)
        #else:
        #    nextpred = nextPred_.reshape(states.shape[0], -1)
        nextpred = self.forward(nstates)
        tmask = 1 - terminal
        trueQ = rewards + discount*tmask*(nextpred.max(axis=1))
        predQ = (out*actions).sum(axis=1)
        diff = predQ - trueQ
        loss = diff.pow(2).sum()
        
        if debug:
            print(f"{blue}{tmask.numpy()=}{endc}")
            print(f"{blue}{out.shape=}{endc}")
            print(f"{underline}{rewards.numpy()=}{endc}")
            print(f"{red}{nextpred.numpy()=}{endc}")
            print(f"{red}{nextpred.max(axis=1).numpy()=}{endc}")
            print(f"{green}{trueQ.numpy()=}{endc}")
            print(f"{yellow}{predQ.numpy()=}{endc}")
            print(f"{purple}{diff.numpy()=}{endc}")
            print(f"{bold}{loss.numpy()=}{endc}\n")
        return loss

    def train(self, experience, discount=1.0):
        states, actions, rewards, nstates, terminals = experience
        #states, actions, rewards, nstates, terminals, statePred, nextPred = experience
        #if np.isnan(statePred[0].numpy()).any():
        #    out = self.forward(states)
        #else:
        #    out = statePred.reshape(states.shape[0], -1)
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
        self.epsilon = 1
        self.decayRate = 0.999999
        self.maxMemory = 10_000
        self.memTypes = 5
        # (states, actions, rewards, next_state, is_terminal)
        self.memory = [[] for i in range(self.memTypes)]
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

    def chooseAction(self, state):
        if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        else: st = state.reshape((1, *state.shape))
        pred = self.main(st).numpy()
        action = np.argmax(pred)
        return action, pred

    def epsRandom(self, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        r = np.random.uniform()
        #return True means make a random choice
        return r <= epsilon

    def doAction(self, action):
        reward = self.env.doAction(action) - self.stepCost
        self.score += reward
        return reward

    def remember(self, experience):
        le = len(experience)
        assert le == self.memTypes, f"got experience tuple of length {le}. number of memory categories is set to {self.memTypes}"
        #state, action, reward, nextState, terminal, statePred, nextPred = experience
        state, action, reward, nextState, terminal = experience
        hot = np.eye(self.actions)[action]
        self.memory[0].append(state)
        self.memory[1].append(hot)
        self.memory[2].append(reward)
        self.memory[3].append(nextState)
        self.memory[4].append(terminal)
        #self.memory[5].append(statePred)
        #self.memory[6].append(nextPred)
        if len(self.memory[0]) > self.maxMemory: self.memory.pop(0)

    def doRandomAction(self):
        return self.doAction(self.randomAction())
    def randomAction(self):
        return np.random.randint(0,self.actions)

    def sampleMemory(self, num, tensor=True):
        assert len(self.memory[1]) > num, f"requested sample size greater than number of recorded experiences"
        samp = np.random.randint(0, len(self.memory[0]), size=(num))
        memlen = len(self.memory)
        expSample = [[] for i in range(memlen)]
        for s in samp:
            for i, mem in enumerate(expSample):
                mem.append(self.memory[i][s])
        if tensor:
            for i, mem in enumerate(expSample):
                t = Tensor(expSample[i])
                expSample[i] = t
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







