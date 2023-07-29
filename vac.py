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

class valueNet(agent.valnet):
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


class vacAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4, policyLr=0.001, valnetLr=0.001):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.policy = vpo.PolicyNet(self.env.size, 4, lr=policyLr) #we borrow the policy net architecture from our vpo implementation
        self.target = valueNet(self.env.size, lr=valnetLr)
        self.main = valueNet(self.env.size, lr=self.target.lr)
        self.update()
        # policy updates need states, actions, weights from the valnet
        # valnet updates need states, rewards, next states, terminal_state
        self.memory = tuple([[] for i in range(6)])

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        #if state.device != "cuda": state = state.to("cuda")
        dist = torch.flatten(self.policy(state))
        if greedy: action = np.argmax(dist.detach().numpy())
        else: action = sampleDist(dist.detach().numpy())
        return action, dist

    def remember(self, state, action, reward, weight, nextState, terminal):
        self.memory[0].append(state)
        self.memory[1].append(action)
        self.memory[2].append(reward)
        self.memory[3].append(weight)
        self.memory[4].append(nextState)
        self.memory[5].append(terminal)

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
        self.memory = tuple([[] for i in range(6)])

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
    a = vacAgent(g, policyLr=policyLr, valnetLr=valnetLr)
    
    wandb.init(project="vac")
    wandb.watch(a.policy, a.target, log="all")
    if load is not None:
        print(f"{green}attemping load from {loadDir}{endc}")
        a.load(loadDir) 

    epscores, losses = [], []
    for i in (t:=trange(numEpisodes, ncols=120, unit="ep")):
        ep = i + startVersion
        ival = a.getVal(g.observe())
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            nstate = g.observe(tensor=False)

            hot = np.eye(a.numActions)[action]
            
            vnext = a.getVal(nstate)
            weight = reward + vnext
            
            a.remember(state, hot, reward, weight, nstate, 1*g.terminate)

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitKey(150)
                d = np.array_str(dist.detach().numpy(), precision=3, suppress_small=True)
                print(f"{bold}{yellow}dist={d}, {green}val={ival:.2f}, {red}{weight} = {reward} + {vnext}{endc}")

        #try: print(f"\n{blue}{weight} = {green}{reward} + {yellow}{tdiff}{endc}")
        #except: pass
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
    a = vacAgent(g)
    agent.play(a, g, load=load, show=show)

startVersion = 200000
loadDir = f"D:\\wgmn\\deepgrid\\vac_net_new\\net_{startVersion}"
#loadDir = f"D:\\wgmn\\deepgrid\\vac_80k"
saveDir = f"D:\\wgmn\\deepgrid\\vac_net_new"

if __name__ == "__main__":
    play(load=loadDir, show=False)
    #train(load=None, save=saveDir, valnetLr=0.012, policyLr=0.0012, trainEvery=50, switchEvery=5, numEpisodes=100_001, show=False)
    #sweep()