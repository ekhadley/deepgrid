from tqdm import trange
from utils import *
from deepgrid.deepq_agent import *
from env import *
from tinygrad.helpers import getenv


g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)

print(f"{yellow}{getenv('GPU')=}{endc}")
print(f"{yellow}{getenv('CUDA')=}{endc}")
print(f"{yellow}{getenv('JIT')=}{endc}")
print(f"{red}{a.main.lin1.weight.device=}{endc}")

loadDir = f"D:\\wgmn\\deepgrid\\deepq_net"
a.load(loadDir)
a.eps = 0

epscores = []
#while 1:
for i in trange(300, ncols=100, desc=cyan, unit="ep"):
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        out = a.chooseAction(state, givePred=True)
        if isinstance(out, tuple): action, pred = out
        else: action, pred = out, np.zeros((4))
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