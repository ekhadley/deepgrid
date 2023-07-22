import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2, numpy as np
from tqdm import trange
from deepgrid.colors import *
import deepgrid as dg
from .vpg_agent import vpgAgent


def play(show=False):
    torch.device("cuda")

    g = dg.grid((8, 5), numFood=12, numBomb=12)
    a = vpgAgent(g)
    
    startVersion = 0
    #loadDir = f"D:\\wgmn\\deepgrid\\vanilla\\rtg_net\\net_{startVersion}.pth"
    loadDir = f"D:\\wgmn\\deepgrid\\vanilla\\rtg_100k.pth"
    a.load(loadDir)
    
    prnt = False
    a.epsilon = 0
    epscores = []
    #while 1:
    for i in (t:=trange(1000, ncols=120, desc=purple, unit="ep")):
        while not g.terminate:
            state = g.observe()
            action, dist = a.chooseAction(state, greedy=True)
            reward = a.doAction(action)
            
            #print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward:.2f}{endc}. The agent now has a score of {cyan}{a.score:.2f}{endc} on step {g.stepsTaken}/{g.maxSteps}")
            #print(f"{green}{pred=}{endc}")
            
            if show:
                im = g.view()
                cv2.imshow("grid", im)
                cv2.waitkey(50)
            if prnt: print(g)

        g.reset()
        epscores.append(a.reset())
        ascore = np.mean(epscores)
        t.set_description(f"{blue}{ascore=:.2f}{purple}")