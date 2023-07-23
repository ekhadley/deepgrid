import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import os, time
import rtg, deepq
from utils import *
from env import grid
import agent as agent

class QPolicyAgent(agent.agent):
    def __init__(self, env, stepCost=0, actions=4):
        self.numActions = actions
        self.env = env
        self.score = 0
        self.stepCost = stepCost
        self.policy = rtg.model(self.env.size, 4)
        self.main = deepq.model(self.env.size, 4)
        self.target = deepq.model(self.env.size, 4)
        self.update()
        self.states = []
        self.actions = []
        self.weights = []
        self.memTypes = 5
        self.gamma = 1
        self.qMemory = [[] for i in range(self.memTypes)]

    def chooseAction(self, state, greedy=False):
        #if not isinstance(state, Tensor): st = Tensor(state).reshape((1, *state.shape))
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        dist = torch.flatten(self.policy(state))
        if greedy: action = np.argmax(dist.detach().numpy())
        else: action = sampleDist(dist.detach().numpy())
        print(f"{yellow}{dist.detach().numpy()=}{endc}")
        return action, dist
    def qPredict(self, state):
        if isinstance(state, np.ndarray): state = torch.from_numpy(state)
        pred = torch.flatten(self.main(state)).detach().numpy()
        print(f"{green}{pred=}{endc}")
        return pred

    def rememberEp(self, states, actions, weights):
        self.states.append(states)
        self.actions.append(actions)
        self.weights.append(weights)
    
    def rememberStep(self, experience):
        #state, action, reward, nextState, terminal, statePred, nextPred = experience
        state, action, reward, nextState, terminal = experience
        
        if torch.is_tensor(state): state = state.numpy()
        if torch.is_tensor(nextState): nextState = nextState.numpy()
        self.qMemory[0].append(state)
        self.qMemory[1].append(action)
        self.qMemory[2].append(reward)
        self.qMemory[3].append(nextState)
        self.qMemory[4].append(terminal)
        if len(self.qMemory[0]) > self.maxMemory:
            for i in range(self.memTypes): self.qMemory[i].pop(0)

    def trainPolicy(self):
        sh = (3, self.env.size[1], self.env.size[0])
        states = torch.Tensor(np.float32(self.states)).reshape(-1, *sh)
        actions = torch.Tensor(np.float32(self.actions))
        weights = torch.Tensor(np.float32(self.weights))
        return self.policy.train(states, actions, weights)
    def trainQnet(self, experience):
        return self.target.train(experience, discount=self.gamma)
    
    def sampleMemory(self, num, tensor=True):
        assert len(self.qMemory[1]) > num, "requested sample size greater than number of recorded transitions"
        samp = np.random.randint(0, len(self.qMemory[0]), size=(num))
        
        expSample = [[] for i in range(self.memTypes)]
        for s in samp:
            for i, mem in enumerate(expSample):
                mem.append(self.qMemory[i][s])
        if tensor:
            for i, mem in enumerate(expSample):
                expSample[i] = torch.tensor(np.float32(expSample[i]))
        return tuple(expSample)

    def update(self):
        self.main.copy(self.target)
    def save(self, path, name):
        self.policy.save(path, f"{name}_policy")
        self.target.save(path, f"{name}_value")
    def load(self, path):
        policy_path = torch.load(f"{path}_policy.pth")
        value_path = torch.load(f"{path}_value.pth")
        self.policy.load_state_dict(policy_path)
        self.target.load_state_dict(value_path)
    
    def reset(self):
        s = self.score
        self.score = 0
        return s
    def forget(self):
        self.states = []
        self.actions = []
        self.weights = []


startVersion = 0
#loadDir = f"D:\\wgmn\\deepgrid\\deepq_net_new\\net_{startVersion}.pth"
loadDir = f"D:\\wgmn\\deepgrid\\hybrid_net_new\\net_25000"
saveDir = f"D:\\wgmn\\deepgrid\\hybrid_net_new"

def train(show=False,
          load=None,
          save=saveDir,
          maxMemory = 10_000,
          saveEvery = 5000,
          switchEvery = 5,
          trainEvery = 5,
          batchSize = 64,
          numEpisodes = 1_000_000):
    
    torch.device("cuda")

    g = grid((8, 5), numFood=12, numBomb=12)
    a = QPolicyAgent(g)

    if load is not None: a.load(loadDir)
    a.maxMemory = maxMemory

    epscores, losses = [], []
    trainingStart = 2*batchSize//g.maxSteps
    for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
        ep = i + startVersion
        states, weights, actions = [], [], []
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            
            hot = np.eye(a.numActions)[int(action)]
            exp = (state, hot, reward, g.observe(tensor=False), 1*g.terminate)
            a.rememberStep(exp)

            weight = a.qPredict(state)[action]
            states.append(state)
            actions.append(hot)
            weights.append(weight)
            
            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitkey(50)

        a.rememberEp(states, actions, weights)
        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
        if i != 0 and i%trainEvery==0:
            loss = a.trainPolicy()
            a.forget()
        if i >= trainingStart:
            experience = a.sampleMemory(batchSize)
            out, loss = a.trainQnet(experience)
            if i%switchEvery==0: a.update()
            
            recents = np.mean(epscores[-100:-1])
            desc = f"{purple}scores:{recents:.2f}, {blue}"
            t.set_description(desc)
            if ep%saveEvery == 0:
                name = f"net_{ep}"
                a.save(save, name)

def play(load=loadDir):
    g = grid((8, 5), numFood=12, numBomb=12)
    a = QPolicyAgent(g)
    agent.play(agent=a, grid=g, load=load)

train()