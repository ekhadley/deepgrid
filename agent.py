import numpy as np
from tinygrad import nn
from funcs import *

class model:
    def __init__(self, gridSize, outSize=4):
        height, width = gridSize
        self.conv1 = nn.Tensor.uniform(16, 3, 3, 3)
        self.conv2 = nn.Tensor.uniform(32, 16, 3, 3)
        self.lin1 = nn.Tensor.uniform(32*height*width, 512)
        self.lin2 = nn.Tensor.uniform(512, 64)
        self.lin3 = nn.Tensor.uniform(64, 4)

    def forward(self, x):
        
        return x

    def train(self, X, Y):
        pred = self.forward(X)

class agent:
    def __init__(self, world, foodReward=10, bombReward=-10, stepCost=1):
        self.score = 0
        self.world = world
        self.main = model(self.world.size)
        self.target = model(self.world.size)
        
        self.foodReward = foodReward
        self.bombReward = bombReward
        self.stepCost = stepCost

    def userAction(self):
        amap = {"w": 0, "a":1, "s":2, "d":3}
        cmd = input("select an action [w, a, s, d]:\n")
        while cmd not in amap:
            cmd = input(f"{red}not a proper action.{endc} select an action [w, a, s, d]:\n")
        reward = self.takeAction(amap[cmd])
        print(f"taking action {yellow}{cmd}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{self.score}{endc}")
        return reward

    def chooseAction():
        pass

    def takeAction(self, action):
        tileType = self.world.takeAction(action)
        reward = self.rewardOf(tileType)
        self.score -= self.stepCost
        self.score += reward
        return reward

    def randomAction(self):
        return self.takeAction(np.random.randint(0,4))

    def rewardOf(self, val):
        if val==self.world.bombValue: return self.bombReward
        if val==self.world.foodValue: return self.foodReward
        if val==self.world.emptyValue: return 0

