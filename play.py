from matplotlib import pyplot as plt
from utils import *
from agent import *
from env import *

g = grid((8, 5), numFood=12, numBomb=12)
a = agent(g)
loadDir = f"D:\\wgmn\\deepq\\nets\\prime"
a.load(loadDir)

while 1:
    while not g.terminate:
        #reward = a.doRandomAction()
        state = g.observe()
        action = a.chooseAction(state)
        reward = a.doAction(action)
        out = a.main(Tensor(state).reshape((1, *state.shape))).numpy()
        print(f"taking action {yellow}{action}{endc} gave a reward of {purple}{reward}{endc}. The agent now has a score of {cyan}{a.score}{endc} on step {g.stepsTaken}/{g.maxSteps}")
        print(f"{purple}{out}{endc}")

        print(g)
        im = g.view()
        cv2.imshow("grid", im)
        cv2.waitKey(0)

    print(g)
    g.reset()
    epscore = a.reset()

