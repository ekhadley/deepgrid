import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import os, time
from utils import *
from env import grid
import agent

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
        X = self.conv1(X)
        X = F.leaky_relu(X)
        X = self.conv2(X)
        X = F.leaky_relu(X)
        
        X = X.reshape(X.shape[0], -1)
        
        X = self.lin1(X)
        X = F.relu(X)
        X = self.lin2(X)
        X = F.relu(X)
        X = self.lin3(X)
        X = F.softmax(X, dim=1) #NOTE the softmax output becuase policies should give a probability distribution over actions
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, dists:torch.tensor, actions, weights:torch.tensor):
        probs = torch.sum(dists*actions, axis=-1)
        logprobs = torch.log(probs)
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


class vpoAgent(agent.agent):
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
        else: action = sampleDist(dist.detach().numpy())
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

startVersion = 70000
#loadDir = f"D:\\wgmn\\deepgrid\\deepq_net_new\\net_{startVersion}.pth"
loadDir = f"D:\\wgmn\\deepgrid\\vpo_100k.pth"
saveDir = f"D:\\wgmn\\deepgrid\\vpo_net_new"

def train(show=False,
          save=saveDir,
          load=loadDir,
          saveEvery = 5000,
          trainEvery = 10,
          numEpisodes = 100_001):

    torch.device("cuda")

    g = grid((8, 5), numFood=12, numBomb=12)
    a = vpoAgent(g)
    if load is not None: a.load(loadDir)

    epscores, losses = [], []
    for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
        ep = i + startVersion
        states, rtg, actions = [], [], []
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            
            hot = np.eye(a.numActions)[action]
            states.append(state)
            rtg = [e + reward for e in rtg] # first accumulating rewards to get the reward-to-go
            rtg.append(reward) # then adding the reward for the current step
            actions.append(hot)

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitKey(50)

        a.remember(states, actions, rtg)
        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
        if i != 0 and i%trainEvery==0:
            loss = a.train()
            a.forget()
            
            recents = np.mean(epscores[-100:-1])
            desc = f"{purple}scores:{recents:.2f}, {red}loss:{loss.detach():.3f}{blue}"
            t.set_description(desc)
            if ep%saveEvery == 0:
                name = f"net_{ep}"
                a.save(save, name)

def play(load=loadDir):
    g = grid((8, 5), numFood=12, numBomb=12)
    a = vpoAgent(g)
    agent.play(a, g, load=load)

if __name__ == "__main__":
    play()
    #train(load=loadDir, save=saveDir)