from tqdm import trange
import math
from utils import *
from agent import *
from env import *
from tinygrad.helpers import getenv

g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)

print(f"{yellow}{getenv('GPU')=}{endc}")
print(f"{yellow}{getenv('CUDA')=}{endc}")
print(f"{yellow}{getenv('JIT')=}{endc}")
print(f"{red}{a.main.lin1.weight.device=}{endc}")

loadDir = f"D:\\wgmn\\deepq\\net2"
a.load(loadDir)
saveDir = f"D:\\wgmn\\deepq\\netxxx"

donothing(a)

Tensor.training = True
a.eps = 0
a.decayRate = 0.999999
saveEvery = 100
trainingStart = 256
numEpisodes = 100_000
episodeScores, losses = [], []
for ep in (t:=trange(numEpisodes, ncols=100, desc=cyan, unit="ep")):
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        output = a.chooseAction(state, givePred=True)
        if isinstance(output, tuple): action, pred = output
        else: action, pred = output, np.zeros((4))
        reward = a.doAction(action)
        if np.isnan(pred).any():
            rb = 100*(ep//100)
            print(f"\n{red}nan'd out on {ep}. rolling back to version {rb}{endc}")
            a.load(f"{saveDir}\\{rb}")

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


