import time
from tqdm import tqdm
from funcs import *
from agent import *
from grid import *

g = grid((8, 5), numFood=6, numBomb=7)
a = agent(g)

for i in tqdm(range(500000)):
#while 1:
    a.randomAction()
    g.printObs()

    #i = g.view()
    #cv2.imshow("g", i)
    #cv2.waitKey(1)
