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
import cProfile

class policyNet(agent.policynet):
    def loss(self, dists:torch.tensor, actions:torch.tensor, weights:torch.tensor, prev_probs:torch.tensor, epsilon, debug=False):
        masked = dists*actions
        probs = torch.sum(masked, axis=1)
        ratio = probs/prev_probs
        #L_c(θ) = min(A*ratio, A*clip(ratio, 1 - ϵ, 1 + ϵ))
        losses = torch.min(weights*ratio, weights*torch.clamp(ratio, max=1+epsilon, min=1-epsilon))
        loss = -torch.mean(losses)
        if debug:
            print(f"\n{red}{dists=}{endc}")
            #print(f"{yellow}{actions=}{endc}")
            #print(f"{bold}{masked=}{endc}")
            print(f"{blue}{probs=}{endc}")
            print(f"{orange}{prev_probs=}{endc}")
            print(f"{red}{ratio=}{endc}")
            print(f"{purple}{weights=}{endc}")
            print(f"{pink}{losses=}{endc}\n\n")
            print(f"{bold}{loss=}{endc}")
        return loss

    def train(self, states, actions, weights, probs, epsilon, debug=False):
        dists = self.forward(states)
        los = self.loss(dists, actions, weights, probs, epsilon, debug=debug)
        self.opt.zero_grad()
        los.backward()
        self.opt.step()
        return dists, los

class ppoAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4, epsilon=.2, lr=0.001):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.eps = epsilon
        self.stepCost = stepCost
        self.policy = policyNet(self.env.size, 4, lr=lr)
        self.states = []
        self.actions = []
        self.weights = []
        self.probs = []

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        if state.ndim==3: state = torch.unsqueeze(state, 0)
        dist = torch.flatten(self.policy(state)).detach().numpy()
        if greedy: action = np.argmax(dist)
        else: action = sampleDist(dist)
        return action, dist

    def remember(self, states, actions, weights, probs):
        self.states += states
        self.actions += actions
        self.weights += weights
        self.probs += probs

    def train(self, experience, debug=False):
        states, actions, weights, probs = experience
        states = torch.Tensor(np.float32(states))
        actions = torch.Tensor(np.float32(actions))
        weights = torch.Tensor(np.float32(weights))
        probs = torch.Tensor(np.float32(probs))
        return self.policy.train(states, actions, weights, probs, self.eps, debug=debug)

    def save(self, path, name):
        self.policy.save(path, name)
    def load(self, path):
        sd = torch.load(path)
        self.policy.load_state_dict(sd)

    def sampleMemory(self, num):
        assert len(self.probs) > num, f"{red}{bold}requested sample size of {num} but only have {len(self.probs)} experiences{endc}"
        samp = np.random.randint(0, len(self.probs), size=(num))
        
        expSample = [[] for i in range(4)]
        for s in samp:
            expSample[0].append(self.states[s])
            expSample[1].append(self.actions[s])
            expSample[2].append(self.weights[s])
            expSample[3].append(self.probs[s])
        return tuple([torch.tensor(e) for e in expSample])

    def reset(self):
        s = self.score
        self.score = 0
        return s

    def forget(self):
        self.states = []
        self.actions = []
        self.weights = []
        self.probs = []

def train(show=False,
          save=None,
          load=None,
          saveEvery = 5000,
          trainEvery = 30,
          lr = 0.001,
          epsilon=0.2,
          trainSteps = 3,
          sampleSize=32,
          numEpisodes = 100_001):


    g = grid((8, 5), numFood=12, numBomb=12)
    a = ppoAgent(g, lr=lr, epsilon=epsilon)
    
    wandb.init(project="ppoc")
    wandb.watch(a.policy, log="all", log_freq=10)
    if load is not None: a.load(loadDir)
    
    trainingStart = 2*sampleSize//g.maxSteps
    epscores, losses = [], []
    for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
        ep = i + startVersion
        states, rtg, actions, probs = [], [], [], []
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            
            states.append(state)
            
            hot = np.eye(a.numActions)[action]
            actions.append(hot)
            
            rtg = [e + reward for e in rtg] # first accumulating rewards to get the reward-to-go
            rtg.append(reward) # then adding the reward for the current step
            
            probs.append(dist[action])

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitKey(50)

        a.remember(states, actions, rtg, probs)
        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
        if i != 0 and i%trainEvery==0:
            for zz in range(trainSteps):
                experience = a.sampleMemory(sampleSize)
                dists, loss = a.train(experience, debug=False)
            a.forget()
            
            wandb.log({"score": epscore, "loss":loss})
            recents = np.mean(epscores[-200:-1])
            d = np.array_str(dist, precision=3, suppress_small=True)
            desc = f"{bold}{blue}scores:{recents:.2f}, {blue}dist={d}, {endc}{blue}"
            t.set_description(desc)
        if save is not None and ep%saveEvery == 0:
            name = f"net_{ep}"
            a.save(save, name)

def play(load, show=False):
    g = grid((8, 5), numFood=12, numBomb=12)
    a = ppoAgent(g)
    agent.play(a, g, load=load, show=show)

startVersion = 0
#loadDir = f"D:\\wgmn\\deepgrid\\ppo_net_new\\net_{startVersion}.pth"
loadDir = f"D:\\wgmn\\deepgrid\\ppoc_net_new\\net_40000.pth"
saveDir = f"D:\\wgmn\\deepgrid\\ppoc_net_new"

if __name__ == "__main__":
    play(load=loadDir, show=True)
    #train(load=None, save=saveDir, lr=0.001, trainEvery=30, sampleSize=64, trainSteps=10, epsilon=0.15, numEpisodes=100_001, show=False)