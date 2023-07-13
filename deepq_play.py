from tqdm import trange
from utils import *
from deepq_agent import *
from env import *
from tinygrad.helpers import getenv


g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)

print(f"{yellow}{getenv('GPU')=}{endc}")
print(f"{yellow}{getenv('CUDA')=}{endc}")
print(f"{yellow}{getenv('JIT')=}{endc}")
print(f"{red}{a.main.lin1.weight.device=}{endc}")

loadDir = f"D:\\wgmn\\deepgrid\\net\\1000"
a.load(loadDir)
a.eps = 0

epscores = []
#while 1:
for i in trange(300, ncols=100, desc=cyan, unit="ep"):
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        action, pred, wasRandom = a.chooseAction(state)
        reward = a.doAction(action, store=False)
        print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{a.score}{endc} on step {g.stepsTaken}/{g.maxSteps}")
        print(f"{yellow}{pred=}{endc}")
        
        im = g.view()
        cv2.imshow("grid", im)
        cv2.waitKey(150)
        print(g)

    g.reset()
    epscores.append(a.reset())

ascore = np.mean(epscores)
print(f"{cyan}{ascore=}{endc}")