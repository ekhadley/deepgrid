from tqdm import trange
from utils import *
from agent import *
from env import *

g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)
loadDir = f"D:\\wgmn\\deepq\\net1"
a.load(loadDir)
a.eps = 0

epscores = []
#while 1:
for i in trange(300):
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        action, pred = a.chooseAction(state, givePred=True)
        reward = a.doAction(action, store=False)
        #print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{a.score}{endc} on step {g.stepsTaken}/{g.maxSteps}")
        #print(f"{purple}{pred=}{endc}")

        #print(g)
        #im = g.view()
        #cv2.imshow("grid", im)
        #cv2.waitKey(1)

    g.reset()
    epscores.append(a.reset())

ascore = np.mean(epscores)
print(f"{cyan}{ascore=}{endc}")