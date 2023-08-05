import cv2, numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils import *

class module(nn.Module):
    def __init__(self): super(module, self).__init__()
    def __call__(self, X): return self.forward(X)

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

class qnet(module):
    def __init__(self, gridSize, actions, lr=.001):
        super(qnet, self).__init__()
        self.gridSize, self.actions, self.lr = gridSize, actions, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.ac1 = nn.LeakyReLU()
        self.lin1 = nn.Linear(32*height*width, 512)
        self.ac2 = nn.ReLU()
        self.lin2 = nn.Linear(512, self.actions)
        self.to("cuda")
        #self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr, fused=True)

    def forward(self, X):
        if X.ndim==3: X = torch.unsqueeze(X, 0)
        if not X.is_cuda: X = X.to("cuda")
        X = self.ac1(self.conv1(X))
        X = X.reshape(X.shape[0], -1)
        X = self.ac2(self.lin1(X))
        X = self.lin2(X)
        return X

class policynet(module):
    def __init__(self, gridSize, actions, lr=.001):
        super(policynet, self).__init__()
        self.gridSize, self.actions, self.lr = gridSize, actions, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.ac1 = nn.LeakyReLU()
        self.lin1 = nn.Linear(32*height*width, 512)
        self.ac2 = nn.ReLU()
        self.lin2 = nn.Linear(512, self.actions)
        self.out = nn.Softmax(dim=1)

        #self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, X:torch.Tensor):
        if X.ndim==3: X = torch.unsqueeze(X, 0)
        X = self.ac1(self.conv1(X))
        X = X.reshape(X.shape[0], -1)
        X = self.ac2(self.lin1(X))
        X = self.out(self.lin2(X))
        return X

class valnet(module):
    def __init__(self, gridSize, lr=.01):
        super(valnet, self).__init__()
        self.gridSize, self.lr = gridSize, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.ac1 = nn.LeakyReLU()
        self.lin1 = nn.Linear(32*height*width, 512)
        self.ac2 = nn.ReLU()
        self.lin2 = nn.Linear(512, 1)
        #self.to("cuda")
        
        #self.opt = torch.optim.SGD(self.parameters(), lr=lr)
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.999, 0.999))

    def forward(self, X):
        if X.ndim==3: X = torch.unsqueeze(X, 0)
        X = self.ac1(self.conv1(X))
        X = X.reshape(X.shape[0], -1)
        X = self.ac2(self.lin1(X))
        X = self.lin2(X)
        return torch.flatten(X)

class agent:
    def __init__(self, *args, **kwargs): raise NotImplementedError

    def doUserAction(self):
        amap = {"w": 0, "a":1, "s":2, "d":3}
        cmd = input("select an action [w, a, s, d]:\n")
        while cmd not in amap:
            cmd = input(f"{red}not a proper action.{endc} select an action [w, a, s, d]:\n")
        reward = self.doAction(amap[cmd])
        print(f"taking action {yellow}{cmd}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{self.score}{endc} on step {self.env.stepsTaken}/{self.env.maxSteps}")
        return reward

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

    def doRandomAction(self):
        return self.doAction(self.randomAction())
    def randomAction(self):
        return np.random.randint(0,self.actions)

def play(agent, grid, show=False, load=None):
    torch.inference_mode(True)
    if load is not None: agent.load(load)

    prnt = False
    agent.epsilon = 0
    epscores = []
    #while 1:
    for i in (t:=trange(1000, ncols=100, desc=purple, unit="ep")):
        while not grid.terminate:
            state = grid.observe()
            action, pred = agent.chooseAction(state)
            reward = agent.doAction(action)
            
            #print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward:.2f}{endc}. The agent now has a score of {cyan}{a.score:.2f}{endc} on step {g.stepsTaken}/{g.maxSteps}")
            #print(f"{green}{pred=}{endc}")
            
            if show:
                im = grid.view()
                cv2.imshow("grid", im)
                cv2.waitKey(50)
            if prnt: print(grid)

        grid.reset()
        epscores.append(agent.reset())
        ascore = np.mean(epscores)
        t.set_description(f"{blue}{ascore=:.2f}{purple}")