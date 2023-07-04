from tqdm import trange
from utils import *
from agent import *
from env import *

g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)

Tensor.training = True

loadDir = f"D:\\wgmn\\deepq\\nets\\prime"
saveDir = f"D:\\wgmn\\deepq\\nets\\prime"
#a.load(loadDir)
a.eps = 1
saveEvery = 1000
trainingStart = 100
numEpisodes = 100_000
episodeScores, losses = [], []
for ep in (t:=trange(numEpisodes, ncols=100, desc=cyan, unit="ep")):
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        action = a.chooseAction(state)
        reward = a.doAction(action)
    
        #print(g)
        if ep >= trainingStart:
            experience = a.sampleMemory(16)
            state, action, reward, nstate, terminal = experience
            out, loss = a.train(experience)

    print(g)
    g.reset()
    epscore = a.reset()
    if ep >= trainingStart:
        losses.append(loss.numpy()[0])
        episodeScores.append(epscore)
        print(f"{purple}{epscore=}, {a.eps=:.4f}, {red}loss={loss.numpy()}{endc}")
        if ep%saveEvery == 0:
            a.save(f"{saveDir}")


