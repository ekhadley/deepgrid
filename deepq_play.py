from tqdm import trange
from utils import *
from deepq_agent import *
from env import *
import matplotlib.pyplot as plt


g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)


startVersion = 800
#loadDir = f"D:\\wgmn\\deepgrid\\deepq_net"
loadDir = f"D:\\wgmn\\deepgrid\\netx\\{startVersion}"
a.load(loadDir)

#l = a.main.lin2.weight.detach().numpy().flatten()
#ws = l
#ax = plt.hist(ws, bins=1000)
#print(f"{yellow}{l.shape=}{endc}")
#print(f"{red}{l.mean()=}{endc}")
#print(f"{purple}{np.var(l)=}{endc}")
#plt.show()


a.epsilon = 0
epscores = []
#while 1:
for i in trange(300, ncols=100, desc=purple, unit="ep"):
    while not g.terminate:
        state = g.observe()
        action, pred = a.chooseAction(state)
        reward = a.doAction(action)
        #print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward:.2f}{endc}. The agent now has a score of {cyan}{a.score:.2f}{endc} on step {g.stepsTaken}/{g.maxSteps}")
        #print(f"{yellow}{pred=}{endc}")
        
        #im = g.view()
        #cv2.imshow("grid", im)
        #cv2.waitKey(50)
        #print(g)

    g.reset()
    epscores.append(a.reset())

ascore = np.mean(epscores)
print(f"{blue}{ascore=}{endc}")