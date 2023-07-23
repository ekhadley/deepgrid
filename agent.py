import cv2, numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class agent:
    def __init__(self, *args, **kwargs): raise NotImplementedError

    def doUserAction(self):
        amap = {"w": 0, "a":1, "s":2, "d":3}
        cmd = input("select an action [w, a, s, d]:\n")
        while cmd not in amap:
            cmd = input(f"{red}not a proper action.{endc} select an action [w, a, s, d]:\n")
        reward = self.doAction(amap[cmd])
        print(f"taking action {yellow}{cmd}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{self.score}{endc} on step {self.env.stepsTaken}/{self.env.maxSteps}")
        return reward

    def epsRandom(self, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        r = np.random.uniform()
        #return True means make a random choice
        return r <= epsilon

    def doAction(self, action):
        reward = self.env.doAction(action) - self.stepCost
        self.score += reward
        return reward

    def doRandomAction(self):
        return self.doAction(self.randomAction())
    def randomAction(self):
        return np.random.randint(0,self.actions)

def play(agent, grid, show=False, load=None):
    torch.device("cuda")
    torch.inference_mode(True)
    if load is not None: agent.load(load)

    prnt = False
    agent.epsilon = 0
    epscores = []
    #while 1:
    for i in (t:=trange(1000, ncols=100, desc=purple, unit="ep")):
        while not grid.terminate:
            state = grid.observe()
            action, pred = agent.chooseAction(state)
            reward = agent.doAction(action)
            
            #print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward:.2f}{endc}. The agent now has a score of {cyan}{a.score:.2f}{endc} on step {g.stepsTaken}/{g.maxSteps}")
            #print(f"{green}{pred=}{endc}")
            
            if show:
                im = grid.view()
                cv2.imshow("grid", im)
                cv2.waitKey(50)
            if prnt: print(grid)

        grid.reset()
        epscores.append(agent.reset())
        ascore = np.mean(epscores)
        t.set_description(f"{blue}{ascore=:.2f}{purple}")