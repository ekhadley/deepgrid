import numpy as np
from utils import *

class agent:
    def __init__(self, *args, **kwargs):
        print(red, "dont initialize agent, use a specific subtype", endc)
        raise NotImplementedError

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