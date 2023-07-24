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
        return X
    def __call__(self, X): return self.forward(X)

    def loss(self, experience, out, discount=1, debug=False):
        states, actions, rewards, nstates, terminal = experience

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

class qAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4):
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
        self.memory[1].append(action)
        self.memory[2].append(reward)
        self.memory[3].append(nextState)
        self.memory[4].append(terminal)
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
                expSample[i] = torch.tensor(np.float32(expSample[i]))
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

startVersion = 0
#loadDir = f"D:\\wgmn\\deepgrid\\deepq_net_new\\net_{startVersion}.pth"
loadDir = f"D:\\wgmn\\deepgrid\\deepq_100k.pth"
saveDir = f"D:\\wgmn\\deepgrid\\deepq_net_new"

def train(show=False,
          load=loadDir,
          save=saveDir,
          epsilon = 1.0,
          decayRate = 0.999997,
          maxMemory = 10_000,
          saveEvery = 5000,
          switchEvery = 5,
          batchSize = 64,
          numEpisodes = 100_001):
    
    torch.device("cuda")

    g = grid((8, 5), numFood=12, numBomb=12)
    a = qAgent(g)

    if load is not None: a.load(loadDir)
    a.epsilon = epsilon
    a.decayRate = decayRate
    a.maxMemory = maxMemory

    epscores, losses = [], []
    trainingStart = 2*batchSize//g.maxSteps
    for i in (t:=trange(numEpisodes, ncols=120, unit="ep")):
        ep = i + startVersion
        while not g.terminate:
            state = g.observe()
            if a.epsRandom() or i < trainingStart:
                action = a.randomAction()
            else:
                action, pred = a.chooseAction(state)

            reward = a.doAction(action)
            a.epsilon *= a.decayRate

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitKey(50)

            hot = np.eye(a.actions)[int(action)]
            exp = (state, hot, reward, g.observe(tensor=False), 1*g.terminate)
            a.remember(exp)

        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
        if i >= trainingStart:
            experience = a.sampleMemory(batchSize)
            out, loss = a.train(experience)
            if i%switchEvery==0: a.update()
            
            recents = np.mean(epscores[-100:-1])
            desc = f"{purple}scores:{recents:.2f}, {cyan}eps:{a.epsilon:.3f}, {red}loss:{loss.detach():.3f}{blue}"
            t.set_description(desc)
            if ep%saveEvery == 0:
                name = f"net_{ep}"
                a.save(save, name)

def play(load=loadDir,):
    g = grid((8, 5), numFood=12, numBomb=12)
    a = qAgent(g)
    agent.play(agent=a, grid=g, load=load)


if __name__ == "__main__":
    play()
    #train(load=None)
