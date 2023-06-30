import time
from tqdm import trange
from funcs import *
from agent import *
from grid import *

g = grid((8, 5), numFood=6, numBomb=7)
a = agent(g)

for i in trange(10_000):
#while 1:
    a.randomAction()
    print(g)

    if g.terminate: g.reset()

    i = g.view()
    cv2.imshow("g", i)
    cv2.waitKey(1)
