import random, numpy as np, datetime
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class agent:
    def __init__(self, env, updateRate, discount, epsilon, learnRate):
        self.env = env
        self.memories = []
        self.memlen = 100000
        self.memreq = 1000
        self.updateRate = updateRate
        self.episode = 0
        self.numtargets = self.env.food + self.env.bomb
        self.batchSize = 500

        #log_dir = "qlearn/qlearnlogs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"deepq/logs/step{self.episode*self.env.epLen + self.env.step}"
        self.tb = TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.net = self.genModel(discount, epsilon, learnRate)
        self.targetNet = self.genModel(discount, epsilon, learnRate)
        self.updateTarget()

    def genModel(self, discount, epsilon, learnRate):
        self.disc = discount
        self.eps = epsilon
        self.ler = learnRate

        net = keras.models.Sequential()
        net.add(keras.layers.Conv2D(64, (3, 3), input_shape=(self.numtargets, 3, 1), activation = "relu"))
#        self.net.add(keras.layers.Conv2D(64, (3,3), input_shape=(self.size, self.size, 3, 1), activation = "relu"))
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(128))
        net.add(keras.layers.Dropout(.2))
        net.add(keras.layers.Dense(len(self.env.moves), activation="linear"))
        
        net.compile(loss="mse", optimizer = keras.optimizers.Adam(lr=self.ler), metrics = ['accuracy'])
        return net

    def train(self):
        if len(self.memories) < self.memreq:
            return
        batch = random.sample(self.memories, self.batchSize)
        states = np.array([s[0][:] for s in batch])[:]*.1
        nextStates = np.array([s[3][:] for s in batch])[:]*.1
        Qpredictions = self.net.predict(states.reshape(self.batchSize, self.numtargets, 3, 1))
        futureQPredictions = self.targetNet.predict(nextStates.reshape(self.batchSize, self.numtargets, 3, 1))

        for i, (obs, action, reward, nextObs) in enumerate(batch):
            if self.env.step < self.env.epLen:
                Qpredictions[i][action] = reward + np.max(futureQPredictions[i])*self.eps
            else:
                Qpredictions[i][action] = reward


        if self.env.step ==  self.env.epLen:
            self.episode = 1
            if self.episode%self.updateRate == 0:
                self.updateTarget()

        #print(Qpredictions[:])
        self.net.fit(np.array(states).reshape(self.batchSize, self.numtargets, 3, 1), np.array(Qpredictions),
                    batch_size = self.batchSize, verbose=1, callbacks = [self.tb])

    def predict(self, obs, target = True):
        npObs = np.array(obs)[:]*.1
        return self.targetNet.predict(npObs.reshape(-1, self.numtargets, 3, 1))

    def genAction(self, show=False):
        npObs = np.array(self.env.getObs())[:]*.1

        if np.random.uniform() < self.eps:
            prediction = self.targetNet.predict(npObs.reshape(-1, self.numtargets, 3, 1))
            if show:
                print(prediction)
            return np.argmax(prediction), 'exploit'
        else:
            return random.choice([0, 1, 2, 3]), 'explore'

    def updateTarget(self):
        self.targetNet.set_weights(self.net.get_weights())

    def remember(self, memory):
        self.memories.append(memory)
        if len(self.memories) > self.memlen:
            self.memories.pop(0)

    def sampleAction(self, action):
        obs = self.env.getObs()
        reward, newEnv = self.env.simAction(action)
        return [obs, action, reward, newEnv]

