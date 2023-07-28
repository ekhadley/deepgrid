import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import os, time
from utils import *
from env import grid
import agent
import wandb

class model(nn.Module):
    def __init__(self, gridSize, actions, lr=.001):
        super(model, self).__init__()
        self.gridSize, self.actions, self.lr = gridSize, actions, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.ac1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.ac2 = nn.LeakyReLU()
        self.lin1 = nn.Linear(32*height*width, 512)
        self.ac3 = nn.ReLU()
        self.lin2 = nn.Linear(512, 64)
        self.ac4 = nn.ReLU()
        self.lin3 = nn.Linear(64, self.actions)
        self.out = nn.Softmax(dim=1)

        #self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)
        self.opt = torch.optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, X):
        sh = X.shape
        if X.ndim==3: X = X.reshape(1, *sh)
        X = self.ac1(self.conv1(X))
        X = self.ac2(self.conv2(X))
        X = X.reshape(X.shape[0], -1)
        X = self.ac3(self.lin1(X))
        X = self.ac4(self.lin2(X))
        X = self.out(self.lin3(X))
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, dists:torch.tensor, actions:torch.tensor, weights:torch.tensor, debug=False):
        masked = dists*actions
        probs = torch.sum(masked, axis=-1)
        logprobs = torch.log(probs)
        wprobs = logprobs*weights
        loss = -torch.mean(wprobs)

        if debug:
            print(f"\n{red}{dists=}{endc}")
            print(f"{yellow}{actions=}{endc}")
            print(f"{bold}{masked=}{endc}")
            print(f"{blue}{probs=}{endc}")
            print(f"{green}{logprobs=}{endc}")
            print(f"{purple}{weights=}{endc}")
            print(f"{cyan}{wprobs=}{endc}\n\n")
            print(f"{bold}{loss=}{endc}")
        return loss

    def train(self, states, actions, weights, debug=False):
        dists = self.forward(states)
        los = self.loss(dists, actions, weights, debug=debug)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return dists, los

    def save(self, path, name):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}\\{name}.pth")

    def load(self, path):
        self.load_state_dict(torch.load(path))


class vpoAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4, lr=0.001, baseline=False):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.policy = model(self.env.size, 4, lr=lr)
        self.baseline = baseline
        self.states = []
        self.actions = []
        self.weights = []
        self.rewards = []
        self.nstates = []
        self.terminals = []

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        dist = torch.flatten(self.policy(state))
        if greedy: action = np.argmax(dist.detach().numpy())
        else: action = sampleDist(dist.detach().numpy())
        return action, dist

    def remember(self, states, actions, weights):
        self.states += states
        self.actions += actions
        self.weights += weights

    def train(self, debug=False):
        states = torch.Tensor(np.float32(self.states))
        actions = torch.Tensor(np.float32(self.actions))
        weights = torch.Tensor(np.float32(self.weights))
        return self.policy.train(states, actions, weights, debug=debug)

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

startVersion = 100000
#loadDir = f"D:\\wgmn\\deepgrid\\vpo_net_new\\net_{startVersion}.pth"
loadDir = f"D:\\wgmn\\deepgrid\\vpo_100k.pth"
saveDir = f"D:\\wgmn\\deepgrid\\vpo_net_new"

def train(show=False,
          save=saveDir,
          load=loadDir,
          saveEvery = 5000,
          trainEvery = 30,
          lr = 0.001,
          numEpisodes = 100_001):

    torch.device("cuda")

    g = grid((8, 5), numFood=12, numBomb=12)
    a = vpoAgent(g, lr=lr)
    
    wandb.init(project="vpo")
    wandb.watch(a.policy, log="all")
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
            dists, loss = a.train()
            a.forget()
            
            wandb.log({"score": epscore, "loss":loss})
            recents = np.mean(epscores[-200:-1])
            d = np.array_str(dist.detach().numpy(), precision=3, suppress_small=True)
            desc = f"{bold}{blue}scores:{recents:.2f}, {blue}dist={d}, {endc}{blue}"
            t.set_description(desc)
        if save is not None and ep%saveEvery == 0:
            name = f"net_{ep}"
            a.save(save, name)

def play(load=loadDir):
    g = grid((8, 5), numFood=12, numBomb=12)
    a = vpoAgent(g)
    agent.play(a, g, load=load)

if __name__ == "__main__":
    play(load=loadDir)
    #train(load=None, save=saveDir, lr=0.001)