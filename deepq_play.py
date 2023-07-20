from tqdm import trange
from utils import *
from deepq_agent import *
from env import *
import matplotlib.pyplot as plt

torch.device("cuda")

g = grid((8, 5), numFood=12, numBomb=12)
a = qAgent(g)

startVersion = 100000
#loadDir = f"D:\\wgmn\\deepgrid\\deepq_net_new\\net_{startVersion}.pth"
loadDir = f"D:\\wgmn\\deepgrid\\deepq_100k.pth"
a.load(loadDir)

show = True
prnt = True
a.epsilon = 0
epscores = []
#while 1:
for i in (t:=trange(1000, ncols=100, desc=purple, unit="ep")):
    while not g.terminate:
        state = g.observe()
        action, pred = a.chooseAction(state)
        #reward = a.doAction(action)
        reward = a.doUserAction()
        
        #print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward:.2f}{endc}. The agent now has a score of {cyan}{a.score:.2f}{endc} on step {g.stepsTaken}/{g.maxSteps}")
        #print(f"{green}{pred=}{endc}")
        
        if show:
            im = g.view()
            cv2.imshow("grid", im)
            cv2.waitKey(50)
        if prnt: print(g)

    g.reset()
    epscores.append(a.reset())
    ascore = np.mean(epscores)
    t.set_description(f"{blue}{ascore=:.2f}{purple}")