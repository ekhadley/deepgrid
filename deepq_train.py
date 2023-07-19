from tqdm import trange
from deepq_agent import *
from utils import *
from env import *
from cProfile import Profile
prof = Profile()
prof.enable()

torch.device("cuda")

g = grid((8, 5), numFood=12, numBomb=12)
a = qAgent(g)

startVersion = 60000
loadDir = f"D:\\wgmn\\deepgrid\\deepq_net_new\\net_{startVersion}.pth"
a.load(loadDir)

saveDir = f"D:\\wgmn\\deepgrid\\deepq_net_new"
#saveDir = f"deepq_net_new"
epscores, losses = [], []
a.epsilon = .05
a.decayRate = 0.999992
a.maxMemmory = 10_000
saveEvery = 1000
switchEvery = 5
batchSize = 64
trainingStart = 2*batchSize//g.maxSteps
numEpisodes = 1_000_000
for i in (t:=trange(numEpisodes, ncols=110, unit="ep")):
    ep = i + startVersion
    while not g.terminate:
        state = g.observe()
        if a.epsRandom() or i < trainingStart:
            action = a.randomAction()
        else:
            action, pred = a.chooseAction(state)

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
        
        recents = np.mean(epscores[-100:-1])
        desc = f"{purple}scores:{recents:.2f}, {cyan}eps:{a.epsilon:.3f}, {red}loss:{loss.detach():.3f}{blue}"
        t.set_description(desc)
        if ep%saveEvery == 0:
            name = f"net_{ep}"
            a.save(saveDir, name)


prof.disable()
prof.dump_stats("D:\\wgmn\\tmp")