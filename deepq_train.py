from tqdm import trange
import math
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
saveDir = f"D:\\wgmn\\deepgrid\\deepq_netxxx"

Tensor.training = True
a.eps = 1
a.decayRate = 0.999999
saveEvery = 100
trainingStart = 256
numEpisodes = 100_000
episodeScores, losses = [], []
for ep in (t:=trange(numEpisodes, ncols=100, desc=cyan, unit="ep")):
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        action, pred, wasRandom = a.chooseAction(state)
        if np.isnan(pred).any():
            rb = saveEvery*(ep//saveEvery)
            print(f"\n{red}nan'd out on {ep}. rolling back to version {rb}{endc}")
            a.load(f"{saveDir}\\{rb}")
        a.doAction(action)

        #print(g)
        if ep >= trainingStart:
            experience = a.sampleMemory(256)
            out, loss = a.train(experience)

    #print(g)
    g.reset()
    epscore = a.reset()
    if ep >= trainingStart:
        losses.append(loss.numpy()[0])
        episodeScores.append(epscore)
        #print(f"{purple}{epscore=}, {a.eps=:.4f}, {red}loss={loss.numpy()}{endc}")

        t.set_description(f"{purple}{epscore=}, {a.eps=:.4f}, {red}loss={loss.numpy()}{blue}")
        if ep%saveEvery == 0:
            pth = f"{saveDir}\\{ep}"
            os.makedirs(pth, exist_ok=True)
            a.save(pth)


