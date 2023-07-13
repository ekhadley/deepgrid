from tqdm import trange
from deepq_agent import *
from utils import *
from env import *
from tinygrad.helpers import getenv

g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)

print(f"{yellow}{getenv('GPU')=}{endc}")
print(f"{yellow}{getenv('CUDA')=}{endc}")
print(f"{yellow}{getenv('JIT')=}{endc}")
print(f"{red}{a.main.lin1.weight.device=}{endc}")

startVersion = 0
#loadDir = f"D:\\wgmn\\deepgrid\\net\\{startVersion}"
#a.load(loadDir)
saveDir = f"D:\\wgmn\\deepgrid\\net"

Tensor.training = True
a.eps = 1.0
a.decayRate = 0.99999
saveEvery = 100
trainingStart = 256
numEpisodes = 100_000
episodeScores, losses = [], []
for i in (t:=trange(numEpisodes, ncols=100, desc=cyan, unit="ep")):
    ep = i + startVersion
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        action, pred, wasRandom = a.chooseAction(state)
        a.doAction(action)

        #print(g)
        if i >= trainingStart:
            experience = a.sampleMemory(128)
            out, loss = a.train(experience)
            if np.isnan(loss.numpy()).any():
                rb = saveEvery*(ep//saveEvery)
                print(f"\n{red}nan'd out on {ep}. rolling back to version {rb}{endc}")
                a.load(f"{saveDir}\\{rb}")

    #print(g)
    g.reset()
    epscore = a.reset()
    if i >= trainingStart:
        losses.append(loss.numpy()[0])
        episodeScores.append(epscore)
        #print(f"{purple}{loss.numpy()=}{endc}")

        t.set_description(f"{purple}{epscore=}, {a.eps=:.4f}, {red}loss={loss.numpy()}{blue}")
        if ep%saveEvery == 0:
            pth = f"{saveDir}\\{ep}"
            os.makedirs(pth, exist_ok=True)
            a.save(pth)


