import numpy as np
from tinygrad.nn import Tensor
from tinygrad.nn import optim
from funcs import *

class model:
    def __init__(self, gridSize, actions, lr=.01):
        self.gridSize, self.actions, self.lr = gridSize, actions, lr
        width, height = gridSize
        self.conv1 = Tensor.uniform(16, 3, 3, 3)
        self.conv2 = Tensor.uniform(32, 16, 3, 3)
        self.lin1 = Tensor.uniform(32*height*width, 512)
        self.lin2 = Tensor.uniform(512, self.actions)
        self.layers = [self.conv1, self.conv2, self.lin1, self.lin2]
        self.opt = optim.SGD(self.layers, lr=self.lr)

    def forward(self, X: Tensor):
        sh = X.shape
        assert len(sh)==4, f"got input tensor shape {sh}. Should be length 4: (batch, channels, width, height)"
        X = X.conv2d(self.conv1, padding=1).relu()
        X = X.conv2d(self.conv2, padding=1).relu()
        X = X.reshape(sh[0], sh[1]*sh[2]*sh[3]*self.covn1.shape[0])
        X = X.dot(self.lin1).relu()
        X = X.dot(self.lin2).relu()
        return X.log_softmax()
    def __call__(self, X): return self.forward(X)

    def loss(self, out:Tensor, action:int, reward:float):
        return (out[action]-reward).pow(2)

    def train(self, X, action, reward):
        out = self.forward(X)
        los = self.loss(out, action, reward)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return out, los
    
    def copy(self, other):
        Tensor.no_grad = True
        self.conv1 = other.conv1*1
        self.conv2 = other.conv2*1
        self.lin1 = other.lin1*1
        self.lin2 = other.lin2*1
        self.layers = [self.conv1, self.conv2, self.lin1, self.lin2]
        self.opt = optim.SGD(self.layers, lr=self.lr)
        Tensor.no_grad = False


class agent:
    def __init__(self, env, stepCost=1, actions=4):
        self.env = env
        self.score = 0
        self.actions = actions
        self.stepCost = stepCost
        self.eps = 1
        # states, actions, rewards, and next states stored in separate lists for sampling
        self.memory = ([], [], [], [])
        self.main = model(self.env.size, 4)
        self.target = model(self.env.size, 4)
        self.target.copy(self.main)

    def doUserAction(self):
        amap = {"w": 0, "a":1, "s":2, "d":3}
        cmd = input("select an action [w, a, s, d]:\n")
        while cmd not in amap:
            cmd = input(f"{red}not a proper action.{endc} select an action [w, a, s, d]:\n")
        reward = self.doAction(amap[cmd])
        print(f"taking action {yellow}{cmd}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{self.score}{endc} on step {self.env.stepsTaken}/{self.env.maxSteps}")
        return reward

    def chooseAction(self, state, eps=None):
        if eps is None: eps = self.eps
        r = np.random.uniform()
        if r <= eps: return np.random.randint(0, self.actions)
        else:
            sh = np.shape(state)
            st = Tensor(state).reshape((1, sh[0], sh[1], sh[2]))
            pred = self.target(st)
            return np.argmax(pred)

    def doAction(self, action, store=True):
        if store: s = self.env.observe()
        reward = self.env.doAction(action)
        if store:
            self.memory[0].append(s)
            self.memory[1].append(action)
            self.memory[2].append(reward)
            self.memory[3].append(self.env.observe())
        self.score -= self.stepCost
        self.score += reward
        return reward

    def randomAction(self):
        return self.doAction(np.random.randint(0,self.actions))

    def sampleMemory(self, num):
        samp = np.random.randint(0, len(self.memory[0]), size=(num))
        states, actions, rewards, nstates = self.memory
        return states[samp], actions[samp], rewards[samp], nstates[samp]
