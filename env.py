import cv2, numpy as np
from utils import *
import torch

class grid():
    def __init__(self, size, startPos=None, numBomb=12, numFood=12, windowWidth=1000, maxSteps=16, foodReward=10, bombReward=-10):
        self.stepsTaken = 0
        self.maxSteps = maxSteps
        self.terminate = False
        self.size = size
        width, height = self.size
        
        self.emptyValue, self.foodValue, self.bombValue = 0, 1, -1
        self.foodReward, self.bombReward = foodReward, bombReward
        self.typemap = {self.emptyValue:"empty", self.bombValue:"bomb", self.foodValue:"food"}
        self.colormap = {self.bombValue:(.1, 0, 1), self.foodValue:(0, 1, .25), self.emptyValue:(.3, 0, 0)}
        self.agentColor = (1, 0.7, 0)

        if startPos is None:
            self.startPos = (width//2, height//2)
        else: self.startPos = startPos
        self.agentPos = self.startPos
        self.numBomb = numBomb
        self.numFood = numFood

        self.tiles = np.ones((height, width), np.int32)*self.emptyValue
        self.observation = np.zeros((3, height, width), np.float32)
        posx, posy = self.agentPos
        self.observation[0][posy][posx] = 1

        self.placeBombs()
        self.placeFood()
        
        winwidth, winheight = windowWidth, round(windowWidth*height/width)
        self.windowSize = (winwidth, winheight)
        self.im = np.zeros((winheight, winwidth, 3))
        self.tileSize = winwidth//width

    def placeBombs(self, numBomb=None):
        num = self.numBomb if numBomb==None else numBomb
        width, height = self.size
        for i in range(num):
            rx, ry = np.random.randint(0, width), np.random.randint(0, height)
            while not self.isEmpty((rx, ry)) or (rx, ry)==self.agentPos:
                rx, ry = np.random.randint(0, width), np.random.randint(0, height)
            self.setBomb((rx, ry))

    def placeFood(self, numFood=None):
        num = self.numFood if numFood==None else numFood
        width, height = self.size
        for i in range(num):
            rx, ry = np.random.randint(0, width), np.random.randint(0, height)
            while not self.isEmpty((rx, ry)) or (rx, ry)==self.agentPos:
                rx, ry = np.random.randint(0, width), np.random.randint(0, height)
            self.setFood((rx, ry))

    def observe(self, tensor=False):
        sh = (3, self.size[1], self.size[0])
        if tensor: return torch.from_numpy(self.observation).reshape(sh)
        return np.array(self.observation, copy=True).reshape(sh)

    def getTile(self, pos):
        x, y = pos
        #assert isint(x) and isint(y), f"non-int type in coordinates: ({pos})"
        return self.tiles[y][x]

    def doAction(self, action):
        # actions are 0,1,2,3. They mean move one square up,left,down,right respectively
        ax, ay = self.agentPos
        width, height = self.size
        self.observation[0][ay][ax] = 0
        if action == 0 and ay != 0:
            self.agentPos = (ax, ay-1)
        elif action == 1 and ax != 0:
            self.agentPos = (ax-1, ay)
        elif action == 2 and ay != height-1:
            self.agentPos = (ax, ay+1)
        elif action == 3 and ax != width-1:
            self.agentPos = (ax+1, ay)
        else: assert action in [0, 1, 2, 3], f"unrecognized action: {action}, {type(action)=}"
        val = self.getTile(self.agentPos)
        self.setEmpty(self.agentPos)
        ax, ay = self.agentPos
        self.observation[0][ay][ax] = 1
        self.stepsTaken += 1
        self.terminate = self.stepsTaken == self.maxSteps
        return self.rewardOf(val)
    
    def reset(self, randomPos=True):
        self.stepsTaken, self.terminate, self.agentPos = 0, False, self.startPos

        width, height = self.size
        self.tiles = np.ones((height, width), np.int32)*self.emptyValue
        self.observation = np.zeros((3, height, width), np.float32)
        
        if randomPos:
            self.startPos = (np.random.randint(0, width), np.random.randint(0, height))
        else: self.agentPos = self.startPos
        posx, posy = self.agentPos

        self.observation[0][posy][posx] = 1

        self.placeBombs()
        self.placeFood()

    def setTile(self, pos, val):
        x, y = pos
        #assert isint(x) and isint(y), f"non-int type in coordinates: ({pos})"
        #assert isint(val), f"value to place in grid must be of int type. got: {val} (type={type(val)})"
        self.tiles[y][x] = val
        if val == self.bombValue: self.observation[2][y][x] = 1
        elif val == self.foodValue: self.observation[1][y][x] = 1
        elif val == self.emptyValue:
            self.observation[1][y][x] = 0
            self.observation[2][y][x] = 0
        else: assert 0, f"given value {val} is not a known tile type"

    def setEmpty(self, pos):
        self.setTile(pos, self.emptyValue)
    def setBomb(self, pos):
        self.setTile(pos, self.bombValue)
    def setFood(self, pos):
        self.setTile(pos, self.foodValue)
    def isBomb(self, pos):
        x, y = pos
        return self.observation[2][y][x]
    def isFood(self, pos):
        x, y = pos
        return self.observation[1][y][x]
    def isEmpty(self, pos):
        x, y = pos
        return not (self.observation[1][y][x] or self.observation[2][y][x])

    def rewardOf(self, val):
        if val==self.bombValue: return self.bombReward
        if val==self.foodValue: return self.foodReward
        if val==self.emptyValue: return 0
    
    def view(self, scale=1):
        sx, sy = self.size
        tileSize = self.tileSize
        first = True
        for x in range(sx):
            for y in range(sy):
                tiletype = self.getTile((x, y))
                col = self.colormap[tiletype] if (x, y) != self.agentPos else self.agentColor
                base = self.im if first else im
                im = cv2.rectangle(base, (tileSize*x, tileSize*y), (tileSize*(x+1), tileSize*(y+1)), color=col, thickness=-1)
                first = False
        ax, ay = self.agentPos
        im = cv2.rectangle(im, (tileSize*ax, tileSize*ay), (tileSize*(ax+1), tileSize*(ay+1)), color=self.agentColor, thickness=-1)
        if scale != 1: return imscale(im, scale)
        return im
    
    def __repr__(self):
        width, height = self.size
        rep = "\n" + "#"*(3*width+2) + "\n"
        ax, ay = self.agentPos
        for y in range(height):
            rep += "#"
            for x in range(width):
                if x==ax and y==ay: rep += f"{blue} @ {endc}"
                else:
                    val = self.getTile((x, y))
                    if val==self.bombValue: rep += f"{red} x {endc}"
                    if val==self.foodValue: rep += f"{green} o {endc}"
                    if val==self.emptyValue: rep += f"   "
            rep += "#\n"
        rep += "#"*(3*width+2)
        return rep

    def printObs(self, obs=None):
        obs = self.observation if obs is None else obs
        cols = [cyan, green, red]
        for i in range(3): print(f"{bold}{cols[i]}{self.observation[i]}{endc}")
        print()


