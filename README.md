# Deep RL
This is my implementation of a few different deep RL algorithms, all applied to a simple custom
environment. It is a small grid in which you have an agent (@), what I have termed food (o),
giving a reward, and bombs (x), giving negative reward. Default parameters are: 8x5 grid, 16
steps/ep, 0 step cost, +-10 for food/bomb, 12 of each, running into a wall does nothing, random
starting position every ep. Example starting state:
<pre>
##########################
# o  o     x  x  o  x  o #
# o  x  x  x        o    #
# o  o           @  x  o #
#    x  x  x     o  o  x #
#                o     x #
##########################
</pre>
The main goal of agents in this environment is to learn how to choose paths which pick up the greatest
number of rewards per step. Because the default number of steps is only 16, you can only really choose
a portion of the grid to totally exploit before you get reset. Smart agents take effecient paths, 
choosing to exploit the most food-rich regions of the grid. Really smart agents even learn to do things
like tanking a bomb in order to get access to two or more rewards. The ekhadley (human) average is about
70-80 points per episode. Below I'll go over the basics of each method implemented, and the performance
after training.

The structure of the project is annoying, I know. It's just so much code is repeated between different
algorithm implementations, but the differences are hard to inherit around or make cross-compatible. I
just adopted a rewrite everything approach, and shoved it all into one files for one implementation. The 
model, agent class, and training loop are all rewritten from scratch for each one. The playing loop
is identical however, so thats written in agent.py and jsut takes in a grid and agent instance.

## Deep Q Learner.
Deep Q is the simplest drl method to implement so it seemed like a good starting point. The goal of deep
Q is to learn the value of taking every action for a given input state. The policy used once our Q net has
been trained is simply to choose at every point the action for which the net predicts the greatest reward.
During training, a portion of actions are taken uniform-randomly in order to  explore and learn about
actions which the net believed to be bad, which they otherwise wouldnt have explored.  
For the loss function we use this trick that apparently works:
$Loss(state, action, reward, nextState) = forward(state) - (reward + discount*max(forward(nextState)))$
The Q net is supposed to represent the sum of expected rewards from now till forever. We define the "true"
Q value of a state-action pair as the reward we actually got, plus the Q value of the resulting state. We
teach the net to estimate all future rewards by only labelling the reward we got right now. This voodoo
apparently works for updating weights, but means the actual loss number doesnt really measure anything
useful. This is a pretty common property of RL algorithms.  

The trained net I have included (trained for the default parameters I gave at the top) has played 100k
episodes, with a batch size of 64, totalling 6.4 million states seen in training. Its average score is
~69: just under human level performance. I had some nice score gains for doing extra training with low
epsilon (0.05-0.01) after the main training run. Training with higher epsilon at this point actually
started hurting my performance. Probably an estimation bias going on that causes this. I tried tweaking
some small things and retraining, But could not crack 70 without going well beyond 100k episodes. Averaging
a score of 75 here seems to be a randomly very good training run/weight initialization. Worth investigating.

## Vanilla Policy Optimization.
Policy Optimization is the basis of many more capable agents in more complex environments. Instead of
choosing actions greedily based on a value function, we represent the policy explicitly, as a neural net
which maps states to probabilities of taking each possible action. But to do weight updates on a policy,
we need to find the gradient of performance with respect to the weights of the net. But how do we find
what direction to step to increase performance, when it is our environment (which is rarely a differentiable
function), that tells us how well we would score using some policy? The answer is to estimate the performance
gradient through an expected value of reward over a number of episodes. The expected reward for an episode is 
$\large EV = \pi(\alpha|\tau)R(\tau)$
, the probability of choosing the sequence of actions you chose (under policy $\pi$), times the sum of all
rewards received after taking that action during that episode. We can strengthen our estimate by averaging
this value over anumber of episodes.  
But then what's our gradient? Well this estimate of performance makes our objective clear. We want expected
episode reward to go up: if the sum of all rewards following some action were high, we want the probability
of taking that action in the future to rise. If the accumulated rewards were negative, we want our prob
to go down. You can find the derivation of the exact form online but our loss should be given by:
$\large Loss(state, actionProbability, weight) = -ln(actionProbability)*weight$
. Note that $\large R(\tau, \alpha)$ is not the only choice that can be made. Several choices, all related to
the wider concept of the "value" of a particular  set of actions or states, can be chosen. 

This algorithm is on-policy, or online, meaning we can only update our policy using experience generated
by the current exact weights. Once we update our weights, we throw all our experience out the window. Note
that this is an undesirable and ineffecient property, and not the case in Q learning. Off policy learning
employs methods that allow us to gain data effeciency by recycling, at the cost of training on potentially
stale experience.

In the files, vanilla and vpo refer to vanilla policy optimization. rtg refers to the fact that the
"weight" of each action in the loss function is the so-called "reward-to-go": the sum of all rewards
received *after* taking an action, during that episode.

The net included has played 100k episodes in training, updating the weights every 10 episodes. It also
acheives an average score of about 72.

### Baselines
Also tested was a version of vanilla policy optimization which uses a learned state-value in the
weight: $\large weight = rtg - V(s_0)$. This version of the weight corresponds to the advantage.
Here it defintely stabilizes early training, but it doesnt acheive a much bettwe score than vpo.

## Vanilla Actor Critic
Actor critic is like vanilla policy gradient methods, but the weight of the policy is given by
another neural net which estimates the value of a certain state. The value net is trained by temporal
difference learning like the Q learner, and the policy is updated based on log-probs of actions times
the weights. The only difference, which I'm not sure is commonplace, is that my early training was 
very volatile. The policy tends to collapse to the same action at evry step and learning never starts.
To combat this, I used, instead of $\large V(s_t)$ as the weight at time t, $\large R_t + V(s_{t+1})$.
This adds a bit of supervision early on, and becuase $\large V(s_t) \rightarrow R_t + V(s_{t+1})$ as training
continues.  
Something I noticed though is that my value function was not usually close to the real values I was
collecting at the later stages of training. The policy learns, the episode scores go up, the estimated
values go up, but the policy just learns faster than the valnet can learn. I played around with different
loss functions, raising to the power of 4, for ex, instead of just mse_loss, to place greater emphasis
on outliers, but they didn't help much.

## PPO-Clip
Proximal Policy Optimization algorithms are offline. They ask: how can I reuse old experience?
The problem with training on experience that was generated with a previous version of the policy net
is that it suggests updates for the policy's weights which may not actually increase the expected rewards
for the current policy. The way PPO (and its parent, TRPO) address this is to remember, for each transition
in experience, the probability that the previous policy version had to take the action that was taken. Then,
during training, we find the ratio between the current policy's action-prob for that state, andthe previous
action prob. Using this ratio, we can approximate the degree of agreement between the two policy versions.
We multiply this ratio by the weights (here I use unweighted rtg, usually advantage is used). If the current
action prob is higher than the previous one, we take a (>1x) correspondingly larger step in the objective's
direction. If the current is less than previous, the step gets scaled down. The clip part of PPO-Clip
refers to the fact that we clip this ratio to a max and min value, (set by epsilon in the code). This
makes our shit more well behaved in the case of extremes, say where one probability was 0 or extremely
small. The whole point of this robustness in the face of outdated experience, is that we can take old
data, and train on it multiple times before discarding it. Here this is implemented by storing experiences
for a number of episodes, then randomly taking a sample, training on that, updating weights, and repeating
for some number of iterations. This results in greater sample effeciency. PPO methods  derive their
motivations from Trust Region Policy Optimization, which uses much more complicated second order measures
to define similarity between policies and safe amounts of difference (regions of trust, if you will) to
update weights with.