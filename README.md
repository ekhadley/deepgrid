# Deep RL
This is my implementation of a few different deep RL approaches, all applied to a simple custom
environment. It is a small grid in which you have an agent (@), what I have termed food (o),
giving a reward, and bombs (x), giving negative reward. Default parameters are: 8x5 grid, 16
steps/ep, -1 step cost, +-10 for food/bomb, 12 of each, running into a wall does nothing, random
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
like tanking a bomb in order to get access to two or more rewards.  
The ekhadley (human) baseline is about 50 points per episode, including the default step cost of -1.  
Below I'll go over the basics of each method implemented, and the performance after training.

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
apparently works for updating weights, but means the actual loss number doesnt really measure anything useful.
This is a pretty common property of RL algorithms.  

The trained net I have included (trained for the default parameters I gave at the top) has played about 45k
episodes, totalling ~700k frames (most training was done on batch size 8, a bit with 16). Average score of 
the included model is 32.  

I noticed that at the end, doing extra training with a high epsilon was making performance go down. Training
with low epsilon, I continued to see gains. I think I couldve been much more sample effecient with a better
decay rate, who knows. Sometimes exploitation is exploration.

## Simple Policy Optimization

