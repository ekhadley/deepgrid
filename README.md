# Deep Q Learner
This is a small project, my practice run at deep Q learning. The agent and environment are from scratch,
The deep learning is done with [tinygrad](https://github.com/geohot/tinygrad).

### The environment
The task is a simple grid world. You make a grid instance (I used 8x5 in my testing). The agent occupies
one position, and each episode reward and punishment tiles are placed randomly around it. The observation
for this environment is a binary 3 channel Tensor which has ones in each channel corresponding to  the agent's
position, food, and bomb positions. Default scores are -1 step cost, 10 for rewards, -10 for bombs. simple as it gets.

### The agent
The agent is similarly a "baby's first Q learner" type of guy. It can move in one of 4 directions per turn, each
turn nets it a reward value. Every step it records that step as an "experience" which is sampled later to train the
target model. It always starts in the center of the grid. It has a pair of neural nets in the typical deep Q fashion.
This was the main architecture I played with:

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.lin1 = nn.Linear(32*height*width, 512)
    self.lin2 = nn.Linear(512, 64)
    self.lin3 = nn.Linear(64, self.actions)

(height and width are the dimensions of the grid which is passed to the agent on initialization)
On the forward pass, the conv layers are softmax'd, but the linear layers have no activation functions.

I've included the weights of my 2 trained target models as numpy arrays. net1 was the result of maybe 3 hours of training,
where the agent always began in the center of the grid. net2 was trained for longer, around 5 hours, and for this one the
agent's starting position is randomized each episode. Differences are discussed in the next section.

## Performance
I did not keep track of exactly how much training was done for each of the 2 models. net1 was probably around 3 hours,
net2 was around 5. net1 averages around 10 points per episode, (with the same starting position), and net 2 averages about
13, meaning they're both grabbing about 3 rewards per episode. Not bad, but I wouldve liked to see more long distance 
thinking because you should be able to get like 40-50 every episode. The main issue is that it doesn't go very far for rewards.
It gets what's nearby then runs into a wall or oscillates. I think it never reinforced the idea that moving towards a distant 
reward should yield almost the same score as moving toward a close one, you just have to wait a bit longer to grab it.
    The real problem is that its right. If it never learns the ability to move towards distant rewards, then even if it makes
that correct first step towards a distant reward, it still doesn't see the value of the previous step from where it is right now.
This seems like exactly what epsilon greedy selection is supposed to address: forcing the model to make what it believes are
poor actions, making it realize that they are not as bad as it thought, and the net gets updated accordingly. I suppose the issue
is that to randomly make the correct series of random decisions which carrry an agent against its wishes to a distant reward, becomes
exponentially unlikely as the distance increases. If a reward is n tiles away and epsilon is e, then the probability of choosing the
correct action (out of 4 actions) that take it to that square as quickly as possible is (e^n)*(1/4^n): e/4 for n=1, e^2/16, e^3/64
you get the idea. it gets small fast. This could explain why epsilon greedy fails to imprint the proper lessons on the net; simply 
because a series of random choices rarely carry the agent to distant points. This is just my guess. You could probably just train it
for a lot longer and get complex behavior. Probably increasing training efficiency and finding the right decay rate is important.

TODO: Implement prioritized replay, different exploration strategies, different loss function



