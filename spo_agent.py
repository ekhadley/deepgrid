import numpy as np
from tinygrad.nn import Tensor
from tinygrad import nn
from utils import *
from agent import *

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
        X = self.conv1(X).leakyrelu()
        X = self.conv2(X).leakyrelu()
        X = X.reshape(sh[0], -1)
        X = self.lin1(X).relu()
        X = self.lin2(X).relu()
        X = self.lin3(X).softmax()
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

class qAgent(agent):
    def __init__(self, env, stepCost=1, actions=4):
        self.env = env
        self.score = 0
        self.gamma = 1
        self.actions = actions
        self.stepCost = stepCost
        self.epsilon = 1
        self.decayRate = 0.999999
        self.maxMemory = 10_000
        self.memory = [[]]
        self.policy = model(self.env.size, 4)
        self.update()

    def chooseAction(self, state):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if not isinstance(state, Tensor): state = Tensor(state)
        dist = self.main(state).numpy()
        action = sampleDist(dist)
        return action, dist

    def remember(self, reward):


    def sampleMemory(self, num, tensor=True):
        assert len(self.memory[1]) > num, "requested sample size greater than number of recorded experiences"
        samp = np.random.randint(0, len(self.memory[0]), size=(num))
        
        expSample = [[] for i in range(self.memTypes)]
        for s in samp:
            for i, mem in enumerate(expSample):
                mem.append(self.memory[i][s])
        if tensor:
            for i, mem in enumerate(expSample):
                expSample[i] = Tensor(expSample[i])
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

    def reset(self):
        s = self.score
        self.score = 0
        #self.update()
        return s

