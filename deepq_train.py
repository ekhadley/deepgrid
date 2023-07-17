from tqdm import trange
from deepq_agent import *
from utils import *
from env import *
import os
from tinygrad.helpers import getenv

g = grid((8, 5), numFood=12, numBomb=12)
a = qAgent(g)

print(f"{yellow}{getenv('GPU')=}{endc}")
print(f"{yellow}{getenv('CUDA')=}{endc}")
print(f"{yellow}{getenv('JIT')=}{endc}")
print(f"{red}{a.main.lin1.weight.device=}{endc}")

startVersion = 43500
loadDir = f"D:\\wgmn\\deepgrid\\deepq_net_new\\{startVersion}"
#loadDir = f"D:\\wgmn\\deepgrid\\deepq_net"
a.load(loadDir)

saveDir = f"D:\\wgmn\\deepgrid\\deepq_net_new"
Tensor.training = True
epscores, losses = [], []
a.epsilon = 0.05
a.decayRate = 0.99995
a.maxMemmory = 10_000
saveEvery = 500
switchEvery = 5
batchSize = 64
trainingStart = 2*batchSize//g.maxSteps
numEpisodes = 100_000
for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
    ep = i + startVersion
    while not g.terminate:
        state = g.observe()
        if a.epsRandom() or i < trainingStart:
            action = a.randomAction()
        else:
            action, pred = a.chooseAction(state)
            if np.isnan(pred).any():
                rb = saveEvery*(ep//saveEvery)
                print(f"\n{red}nan'd out on {ep}. rolling back to version {rb}{endc}")
                a.load(f"{saveDir}\\{rb}")

        reward = a.doAction(action)
        a.epsilon *= a.decayRate
        
        exp = (state, action, reward, g.observe(tensor=False), 1*g.terminate)
        a.remember(exp)

    g.reset()
    epscore = a.reset()
    epscores.append(epscore)
    if i >= trainingStart:
        experience = a.sampleMemory(batchSize)
        out, loss = a.train(experience)
        if i%switchEvery==0: a.update()
        
        recents = np.mean(epscores[-10:-1])
        desc = f"{purple}recent scores: {recents:.2f}, {cyan}{a.epsilon=:.4f}, {red}loss={loss.numpy()}{blue}"
        t.set_description(desc)
        if ep%saveEvery == 0:
            pth = f"{saveDir}\\{ep}"
            os.makedirs(pth, exist_ok=True)
            a.save(pth)


