import time
from tqdm import trange
from funcs import *
from agent import *
from grid import *

g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)
print(f"{yellow}{a.main.layers[2].numpy()}{endc}")
print(f"{blue}{a.target.layers[2].numpy()}{endc}")
a.target.opt.params[2] = a.target.opt.params[2]*0
print(f"{yellow}{a.main.layers[2].numpy()}{endc}")
print(f"{blue}{a.target.layers[2].numpy()}{endc}")

print(g)
#for i in trange(10_000):
while 1:
    reward = a.doUserAction()
    #reward = a.randomAction()
    
    print(g)

    if g.terminate: g.reset()

    i = g.view()
    cv2.imshow("g", i)
    cv2.waitKey(1)
