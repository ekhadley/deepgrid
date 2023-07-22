import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2, numpy as np
from tqdm import trange
from deepgrid.colors import *
import deepgrid as dg
from .vpg_agent import vpgAgent

def train(show=False):
    torch.device("cuda")

    g = dg.grid((8, 5), numFood=12, numBomb=12)
    a = vpgAgent(g)

    startVersion = 0
    #loadDir = f"D:\\wgmn\\deepgrid\\vanilla\\rtg_{startVersion}.pth"
    #a.load(loadDir)

    saveDir = f"D:\\wgmn\\deepgrid\\vanilla\\rtg_net_new"
    epscores, losses = [], []
    saveEvery = 5000
    trainEvery = 10
    numEpisodes = 1_000_000
    for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
        ep = i + startVersion
        states, rtg, actions = [], [], []
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            
            hot = np.eye(a.numActions)[action]
            states.apppend(state)
            rtg = [e + reward for e in rtg[-1]] # first accumulating rewards to get the reward-to-go
            rtg.append(reward) # then adding the reward for the current step
            actions.append(hot)

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitkey(50)

        a.remember(states, rtg, actions)
        if i != 0 and i%trainEvery==0:
            loss = a.train()
            a.forget()
            
            recents = np.mean(epscores[-100:-1])
            desc = f"{purple}scores:{recents:.2f}, {red}loss:{loss.detach():.3f}{blue}"
            t.set_description(desc)
            if ep%saveEvery == 0:
                name = f"net_{ep}"
                a.save(saveDir, name)

        g.reset()
        epscore = a.reset()
        epscores.append(epscore)
