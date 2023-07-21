import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2, numpy as np
from tqdm import trange
from deepgrid.colors import *
import deepgrid as dg
from .spo_agent import spoAgent
#from cProfile import Profile
#prof = Profile()
#prof.enable()

def train(show=False):
    torch.device("cuda")

    g = dg.grid((8, 5), numFood=12, numBomb=12)
    a = spoAgent(g)

    startVersion = 0
    #loadDir = f"D:\\wgmn\\deepgrid\\spo_net_new\\net_{startVersion}.pth"
    #a.load(loadDir)

    saveDir = f"D:\\wgmn\\deepgrid\\spo\\spo_net_new"
    epscores, losses = [], []
    saveEvery = 5000
    trainEvery = 10
    numEpisodes = 1_000_000
    for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
        ep = i + startVersion
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state)
            reward = a.doAction(action)
            
            hot = np.eye(a.numActions)[action]
            a.remember(state, hot, reward)

            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitkey(50)

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

#prof.disable()
#prof.dump_stats("D:\\wgmn\\tmp")