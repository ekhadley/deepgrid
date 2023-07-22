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
71: about human level performance. I had some nice score gains for doing extra training with low epsilon
(0.05-0.01) after the main training run. Training with higher epsilon at this point actually started
hurting my performance. Probably an estimation bias going on that causes this.

## Vanilla Policy Optimization.
Policy Optimization is the basis of many more capable agents in more complex environments. Instead of
choosing actions greedily based on a value function, we represent the policy explicitly, as a neural net
which maps states to probabilities of taking each possible action. But to do weight updates on a policy,
we need to find the gradient of performance with respect to the weights of the net. But how do we find
what direction to step to increase performance, when it is our environment (which is rarely a differentiable
function), that tells us how well we would score using some policy? The answer is to estimate the performance
gradient through an expected value of reward over a number of episodes. The expected reward for an episode is 
$\large EV = \pi(\alpha|\tau)R(\tau) $
, the probability of choosing the sequence of actions you chose (under policy $\pi$), times the sum of all
rewards received after taking that action during that episode. We can strengthen our estimate by averaging
this value over anumber of episodes.  
But then what's our gradient? Well this estimate of performance makes our objective clear. We want expected
episode reward to go up: if the sum of all rewards following some action were high, we want the probability
of taking that action in the future to rise. If the accumulated rewards were negative, we want our prob
to go down. You can find the derivation of the exact form online but our loss should be given by:
$\large Loss(state, actionProbability, weight) = -ln(actionProbability)*weight$
. Where in our case weight is chosen to be the cumulative rewards received after the action was taken. Note
that $\large R(\tau, \alpha)$ is not the only choice that can be made. Several choices, all related to the
wider concept of the "value" of a particular  set of actions or states, can be chosen. Often the weight
is chosen to be $Q(s)$ or $V(s)$, approximated with a neural net and learned along side the policy.  

In the files, vanilla and vpg refer to vanilla policy optimization. rtg_train refers to the fact that the
"weight" of each action in the loss function is the so-called "reward-to-go": the sum of all rewards
received `after` taking an action, during that episode.

The net included has played 100k episodes in training, updating the weights every 10 episodes. It also
acheives an average score of about 70.















