import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import os, time
from utils import *
from env import grid
import agent, vpo
import wandb

class valnet(nn.Module):
    def __init__(self, gridSize, lr=.01):
        super(valnet, self).__init__()
        self.gridSize, self.lr = gridSize, lr
        width, height = gridSize
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.ac1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.ac2 = nn.LeakyReLU()
        self.lin1 = nn.Linear(32*height*width, 512)
        self.ac3 = nn.ReLU()
        self.lin2 = nn.Linear(512, 64)
        self.ac4 = nn.ReLU()
        self.lin3 = nn.Linear(64, 1)
        #self.to("cuda")
        #self.opt = nn.optim.SGD([layer.weight for layer in self.layers], lr=self.lr)
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, X):
        sh = X.shape
        if X.ndim==3: X = X.reshape(1, *sh)
        X = self.ac1(self.conv1(X))
        X = self.ac2(self.conv2(X))
        X = X.reshape(X.shape[0], -1)
        X = self.ac3(self.lin1(X))
        X = self.ac4(self.lin2(X))
        X = self.lin3(X)
        return torch.flatten(X)
    def __call__(self, X): return self.forward(X)

    def loss(self, vals, rewards, nstates, terminal, discount=1, debug=False):
        mask = 1-terminal
        nextval = discount*self.forward(nstates)
        trueval = rewards + mask*nextval
        #loss = F.mse_loss(vals, trueval)
        loss = torch.mean((vals-trueval)**4)
        if debug:
            print(f"\n{blue}{vals=}{endc}")
            print(f"{green}{rewards=}{endc}")
            print(f"{cyan}{mask=}{endc}")
            #print(f"{yellow}{nstates=}{endc}")
            print(f"{red}{nextval=}{endc}")
            print(f"{blue}{trueval=}{endc}")
            print(f"{yellow}{loss=}{endc}\n\n\n")
        return loss

    def train(self, states, rewards, nstates, terminals, discount=1.0, debug=False):
        vals = self.forward(states)
        los = self.loss(vals, rewards, nstates, terminals, discount=discount, debug=debug)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return vals, los

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


class bpoAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4, policyLr=0.001, valnetLr=0.001):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.policy = vpo.model(self.env.size, 4, lr=policyLr) #we borrow the policy net architecture from our vpo implementation
        self.target = valnet(self.env.size, lr=valnetLr)
        self.main = valnet(self.env.size, lr=self.target.lr)
        self.update()
        # policy updates need states, actions, weights from the valnet
        # valnet updates need states, rewards, next states, terminal_state
        self.memory = [[] for i in range(6)]

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        #if state.device != "cuda": state = state.to("cuda")
        dist = torch.flatten(self.policy(state))
        if greedy: action = np.argmax(dist.detach().numpy())
        else: action = sampleDist(dist.detach().numpy())
        return action, dist

    def remember(self, state, action, reward, weight, nextState, terminal):
        self.memory[0] += state
        self.memory[1] += action
        self.memory[2] += reward
        self.memory[3] += weight
        self.memory[4] += nextState
        self.memory[5] += terminal

    def train(self):
        states, actions, rewards, weights, nstates, terminals = self.memory
        states = torch.Tensor(np.float32(states))
        actions = torch.Tensor(np.float32(actions))
        weights = torch.Tensor(np.float32(weights))
        rewards = torch.Tensor(np.float32(rewards))
        nstates = torch.Tensor(np.float32(nstates))
        terminals = torch.Tensor(np.float32(terminals))
        pLoss = self.trainPolicy(states, actions, weights, debug=False)
        vLoss = self.trainValnet(states, rewards, nstates, terminals, debug=False)
        return pLoss, vLoss
    
    def trainPolicy(self, states, actions, weights, debug=False):
        dists, loss = self.policy.train(states, actions, weights, debug=debug)
        return loss
    def trainValnet(self, states, rewards, nstates, terminals, debug=True):
        vals, loss = self.target.train(states, rewards, nstates, terminals, debug=debug)
        return loss
    
    def getVal(self, state):
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        #if state.device != "cuda": state = state.to("cuda")
        #return self.main(state).cpu().detach().numpy()[0]
        return self.main(state).detach().numpy()[0]
    def save(self, path, name):
        self.policy.save(path, f"{name}_policy")
        self.target.save(path, f"{name}_target")
    def load(self, path):
        psd = torch.load(f"{path}_policy.pth")
        tsd = torch.load(f"{path}_target.pth")
        self.policy.load_state_dict(psd)
        self.target.load_state_dict(tsd)
        self.update()
    def update(self):
        self.main.copy(self.target)
    def reset(self):
        s = self.score
        self.score = 0
        return s
    def forget(self):
        self.memory = [[] for i in range(6)]


startVersion = 100000
#loadDir = f"D:\\wgmn\\deepgrid\\bpo_net_new\\net_{startVersion}"
loadDir = f"D:\\wgmn\\deepgrid\\bpo_100k"
saveDir = f"D:\\wgmn\\deepgrid\\bpo_net_new"

def train(show=False,
          save=saveDir,
          load=None,
          saveEvery = 5000,
          trainEvery = 30,
          switchEvery = 3,
          numEpisodes = 15_001,
          policyLr = 0.001,
          valnetLr = 0.001):


    g = grid((8, 5), numFood=12, numBomb=12)
    a = bpoAgent(g, policyLr=policyLr, valnetLr=valnetLr)
    
    wandb.init(project="baseline")
    wandb.watch(a.policy, a.target, log="all")
    if load is not None:
        print(f"{green}attemping load from {loadDir}{endc}")
        a.load(loadDir) 

    epscores, losses = [], []
    for i in (t:=trange(numEpisodes, ncols=120, unit="ep")):
        ep = i + startVersion
        ival = a.getVal(g.observe())
        states, actions, rewards, weights, nstates, terminals = [], [], [], [], [], []
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            nstate = g.observe(tensor=False)

            states.append(state)
            nstates.append(nstate)
            hot = np.eye(a.numActions)[action]
            actions.append(hot)
            
            val = a.getVal(nstate)
            weights = [e+reward for e in weights] # first accumulating rewards to get the reward-to-go
            weights.append(reward - val) # then appending reward-baseline value
            
            rewards.append(reward)
            terminals.append(1*g.terminate)

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitKey(150)
                d = np.array_str(dist.detach().numpy(), precision=3, suppress_small=True)
                #print(f"{bold}{yellow}dist={d}, {green}val={ival:.2f}")

        a.remember(states, actions, rewards, weights, nstates, terminals)
        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
        if i != 0 and i%(switchEvery*trainEvery)==0: a.update()
        if i != 0 and i%trainEvery==0:
            pLoss, vLoss = a.train()
            a.forget()
            
            recents = np.mean(epscores[-200:-1])
            d = np.array_str(dist.detach().numpy(), precision=3, suppress_small=True)
            desc = f"{bold}{blue}scores:{recents:.2f}, {blue}dist={d}, {green}val={ival:.2f} {endc}{blue}"
            t.set_description(desc)
            
            wandb.log({"epscore": epscore, "policy_loss":pLoss, "valnet_loss":vLoss, "val":ival, "score":recents})
        if i%saveEvery == 0:
            name = f"net_{ep}"
            a.save(save, name)

def sweep():
    sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {
        'goal': 'maximize', 
        'name': 'score'
        },
    'parameters': {
        'trainEvery': {'values': [10, 15, 20, 25, 30, 35, 40, 45, 50]},
        'valnetLr': {'max': 0.1, 'min': 0.0001},
        'policyLr': {'max': 0.1, 'min': 0.0001},
        'switchEvery': {'values': [1, 2, 3, 4, 5]},
        }
    }
    swid = wandb.sweep(sweep=sweep_configuration, project='sweep')
    wandb.agent(swid, function=train)

def play(load=loadDir, show=False):
    g = grid(size=(8, 5), numFood=12, numBomb=12)
    a = bpoAgent(g)
    agent.play(a, g, load=load, show=show)


torch.manual_seed(0)
np.random.seed(0)
if __name__ == "__main__":
    play(load=loadDir, show=False)
    #train(load=None, save=saveDir, valnetLr=0.01, policyLr=0.001, trainEvery=30, switchEvery=3, numEpisodes=100_001, show=False)
    #sweep()