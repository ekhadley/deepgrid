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
import cProfile

class valueNet(agent.valnet):
    def loss(self, vals, rewards, nstates, terminal, discount=1, debug=False):
        mask = 1-terminal
        nextval = discount*self.forward(nstates)
        trueval = rewards + mask*nextval
        loss = F.mse_loss(vals, trueval)
        #loss = torch.mean((vals-trueval)**4)
        if debug:
            print(f"\n{blue}{vals=}{endc}")
            print(f"{green}{rewards=}{endc}")
            print(f"{cyan}{mask=}{endc}")
            #print(f"{yellow}{nstates=}{endc}")
            print(f"{red}{nextval=}{endc}")
            print(f"{blue}{trueval=}{endc}")
            print(f"{yellow}{loss=}{endc}\n\n\n")
        return loss

    def _train(self, states, rewards, nstates, terminals, discount=1.0, debug=False):
        self.train()
        vals = self.forward(states)
        los = self.loss(vals, rewards, nstates, terminals, discount=discount, debug=debug)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        self.eval()
        return vals, los


class vacAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4, policyLr=0.001, valnetLr=0.001, maxMemory=1000):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.maxMemory = maxMemory
        self.policy = vpo.PolicyNet(self.env.size, 4, lr=policyLr) #we borrow the policy net architecture from our vpo implementation
        self.target = valueNet(self.env.size, lr=valnetLr)
        self.main = valueNet(self.env.size, lr=self.target.lr)
        self.target.eval()
        self.main.eval()
        self.update()
        # policy updates need states, actions, weights from the valnet
        # valnet updates need states, rewards, next states, terminal_state
        self.policymem = [[], [], []]
        self.valnetmem = [[], [], [], []]

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        #if state.device != "cuda": state = state.to("cuda")
        dist = torch.flatten(self.policy(state))
        if greedy: action = np.argmax(dist.detach().numpy())
        else: action = sampleDist(dist.detach().numpy())
        return action, dist

    def remember(self, state, action, reward, weight, nextState, terminal):
        self.policymem[0].append(state)
        self.policymem[1].append(action)
        self.policymem[2].append(weight)

        self.valnetmem[0].append(state)
        self.valnetmem[1].append(reward)
        self.valnetmem[2].append(nextState)
        self.valnetmem[3].append(terminal)
        if len(self.valnetmem[0]) > self.maxMemory:
            over = len(self.valnetmem[0]) - self.maxMemory
            del self.valnetmem[0][:over]
            del self.valnetmem[1][:over]
            del self.valnetmem[2][:over]
            del self.valnetmem[3][:over]

    def trainPolicy(self, debug=False):
        states = torch.tensor(np.float32(self.policymem[0]))
        actions = torch.tensor(np.float32(self.policymem[1]))
        weights = torch.tensor(np.float32(self.policymem[2]))
        dists, loss = self.policy.train(states, actions, weights, debug=debug)
        return dists, loss

    def trainValnet(self, experience, debug=False):
        states, rewards, nstates, terminals = experience
        vals, loss = self.target._train(states, rewards, nstates, terminals, debug=debug)
        return vals, loss
    def sampleMemory(self, num, tensor=True):
        assert len(self.valnetmem[1]) > num, f"{red}{bold}requested sample size of {num} but only have {len(self.valnetmem[1])} experiences{endc}"
        samp = np.random.randint(0, len(self.valnetmem[0]), size=(num))
        
        expSample = [[] for i in range(4)]
        for s in samp:
            for i, mem in enumerate(expSample):
                mem.append(self.valnetmem[i][s])
        if tensor:
            for i, mem in enumerate(expSample):
                expSample[i] = torch.tensor(np.float32(expSample[i]))
        return tuple(expSample)
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
        self.policymem = [[], [], []]

def train(show=False,
          save=None,
          load=None,
          saveEvery = 5000,
          trainEvery = 30,
          switchEvery = 3,
          numEpisodes = 15_001,
          batchSize = 32,
          maxMemory=1000,
          policyLr = 0.001,
          valnetLr = 0.001):

    g = grid((8, 5), numFood=12, numBomb=12)
    a = vacAgent(g, policyLr=policyLr, valnetLr=valnetLr, maxMemory=maxMemory)
    
    wandb.init(project="vac")
    wandb.watch(a.policy, a.target, log="all")
    if load is not None:
        print(f"{green}attemping load from {loadDir}{endc}")
        a.load(loadDir) 

    #qqq = vacAgent(g, policyLr=policyLr, valnetLr=valnetLr, maxMemory=maxMemory)
    #qqq.load(f"D:\\wgmn\\deepgrid\\vac_100k")
    #a.main.copy(qqq.main)
    #a.target.copy(qqq.target)

    trainingStart = 2*batchSize//g.maxSteps
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

        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
        if i >= trainingStart and i%trainEvery == 0:
            experience = a.sampleMemory(batchSize)
            vals, vLoss = a.trainValnet(experience)

            if i%switchEvery==0: a.update()
        if i != 0 and i%trainEvery==0:
            dists, pLoss = a.trainPolicy()
            a.forget()
            
            recents = np.mean(epscores[-200:-1])
            d = np.array_str(dist.detach().numpy(), precision=3, suppress_small=True)
            desc = f"{bold}{cyan}scores:{recents:.2f}, {blue}dist={d}, {green}val={ival:.2f} {endc}{blue}"
            t.set_description(desc)
            wandb.log({"epscore": epscore, "policy_loss":pLoss, "valnet_loss":vLoss, "val":ival, "score":recents})
        if save is not None and i%saveEvery == 0:
            name = f"net_{ep}"
            a.save(save, name)

def play(load, show=False):
    g = grid(size=(8, 5), numFood=12, numBomb=12)
    a = vacAgent(g)
    agent.play(a, g, load=load, show=show)

startVersion = 0
#loadDir = f"D:\\wgmn\\deepgrid\\vac_net_new\\net_{startVersion}"
loadDir = f"D:\\wgmn\\deepgrid\\vac_100k"
saveDir = f"D:\\wgmn\\deepgrid\\vac_net_new"

#prof = cProfile.Profile()
#prof.enable()

# TODO: in basically all value net models i trained, the value estimator was perpetually too low.
# TODO: it rose as learning occurred, but fell off early and lagged behind true scores.
# TODO: identify wether this is due to the distribution over true values being unbalanced
# TODO: by sampling from memory with copies or smthn to make it uniform, or try batchnorm or smthn

# NOTE: also why does training a policy from scratch with a high scoring policie's valuenet not really help?
if __name__ == "__main__":
    train(load=None, save=saveDir, valnetLr=0.012, policyLr=0.0012, batchSize=32, trainEvery=50, switchEvery=3, maxMemory=300, numEpisodes=100_001, show=False)
    play(load=saveDir, show=True)
    #sweep()

#prof.disable()
#prof.dump_stats("tmp")