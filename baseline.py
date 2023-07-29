import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import os, time
from utils import *
from env import grid
import agent, vpo, vac
import wandb

class bpoAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4, policyLr=0.001, valnetLr=0.001):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.policy = vpo.PolicyNet(self.env.size, 4, lr=policyLr) #we borrow the policy net architecture from our vpo implementation
        self.target = vac.valueNet(self.env.size, lr=valnetLr)
        self.main = vac.valueNet(self.env.size, lr=self.target.lr)
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

def train(show=False,
          save=None,
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

def play(load, show=False):
    g = grid(size=(8, 5), numFood=12, numBomb=12)
    a = bpoAgent(g)
    agent.play(a, g, load=load, show=show)

startVersion = 100000
loadDir = f"D:\\wgmn\\deepgrid\\bpo_net_new\\net_{startVersion}"
#loadDir = f"D:\\wgmn\\deepgrid\\bpo_100k"
saveDir = f"D:\\wgmn\\deepgrid\\bpo_net_new"

if __name__ == "__main__":
    #play(load=loadDir, show=False)
    train(load=None, save=saveDir, valnetLr=0.01, policyLr=0.001, trainEvery=50, switchEvery=3, numEpisodes=100_001, show=False)
    #sweep()