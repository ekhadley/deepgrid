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
        X = F.relu(self.conv1(X))
        X = F.leaky_relu(self.conv2(X))
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.lin1(X))
        X = F.relu(self.lin2(X))
        X = self.lin3(X)
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, experience, out, discount=1, debug=False):
        states, actions, rewards, nstates, terminal = experience
        #states, actions, rewards, nstates, terminal, statePred, nextPred_ = experience
        #if np.isnan(nextPred_.numpy()).any():
        #    nextpred = self.forward(nstates)
        #else:
        #    nextpred = nextPred_.reshape(states.shape[0], -1)
        nextpred = self.forward(nstates)
        tmask = (1 - terminal)*discount
        trueQ = rewards + (nextpred.max(axis=1).values)*tmask
        predQ = (out*actions).sum(axis=1)
        diff = predQ - trueQ
        loss = diff.pow(2).sum()
        
        if debug:
            print(f"{red}{states.shape=}{endc}")
            print(f"{green}{out.shape=}{endc}")
            print(f"{bold}{rewards.detach()=}{endc}")
            print(f"{red}{nextpred.detach()=}{endc}")
            print(f"{blue}{tmask.detach()=}{endc}")
            print(f"{green}{trueQ.detach()=}{endc}")
            print(f"{yellow}{predQ.detach()=}{endc}")
            print(f"{purple}{diff.detach()=}{endc}")
            print(f"{bold}{loss.detach()=}{endc}\n")
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
        otherparams = [e for e in other.parameters()]
        with torch.no_grad():
            for i, layer in enumerate(self.parameters()):
                layer.copy_(otherparams[i].detach().clone())

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")

    def load(self, path):
        self.load_state_dict(torch.load(path))


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
        self.memTypes = 5
        # (states, actions, rewards, next_state, is_terminal)
        self.memory = [[] for i in range(self.memTypes)]
        self.main = model(self.env.size, 4)
        self.target = model(self.env.size, 4)
        self.update()

    def chooseAction(self, state):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        pred = self.main(state).detach().numpy()
        action = np.argmax(pred)
        return action, pred

    def remember(self, experience):
        #state, action, reward, nextState, terminal, statePred, nextPred = experience
        state, action, reward, nextState, terminal = experience
        
        if torch.is_tensor(state): state = state.numpy()
        if torch.is_tensor(nextState): nextState = nextState.numpy()
        self.memory[0].append(state)
        hot = np.eye(self.actions)[int(action)]
        self.memory[1].append(hot)
        self.memory[2].append(reward)
        self.memory[3].append(nextState)
        self.memory[4].append(terminal)
        #self.memory[5].append(statePred)
        #self.memory[6].append(nextPred)
        if len(self.memory[0]) > self.maxMemory:
            for i in range(self.memTypes): self.memory[i].pop(0)

    def sampleMemory(self, num, tensor=True):
        assert len(self.memory[1]) > num, "requested sample size greater than number of recorded experiences"
        samp = np.random.randint(0, len(self.memory[0]), size=(num))
        
        expSample = [[] for i in range(self.memTypes)]
        for s in samp:
            for i, mem in enumerate(expSample):
                mem.append(self.memory[i][s])
        if tensor:
            for i, mem in enumerate(expSample):
                expSample[i] = torch.tensor(expSample[i])
        return tuple(expSample)

    def train(self, experience):
        return self.target.train(experience, discount=self.gamma)
    def update(self):
        self.main.copy(self.target)
    def save(self, path, name):
        self.target.save(path, name)
    def load(self, path):
        sd = torch.load(path)
        self.target.load_state_dict(sd)
        self.update()

    def reset(self):
        s = self.score
        self.score = 0
        #self.update()
        return s






