
import cv2, time, numpy as np
from tqdm import tqdm
from funcs import *
from agent import *
from grid import *

g = grid((8, 5), numFood=5, numBomb=5)
a = agent(g)

#for i in tqdm(range(10000)):
i = g.view()
cv2.imshow("g", i)
cv2.waitKey(1)
while 1:
    #g.takeAction(random.randint(0,4))
    a.userAction()
    print(g.tiles)

    i = g.view()
    cv2.imshow("g", i)
    cv2.waitKey(1)
