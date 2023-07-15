# Deep RL
This is my implementation of a few different deep RL approaches, all applied to a simple custom
environment. It is a small grid (I used 8x5 to train all the included nets) in which you have
an agent (@), what I have termed food (o), giving a reward, and bombs (x), giving negative reward:  
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

## Deep Q
Deep Q is the simplest drl method to implement so it seemed like a good starting point. The goal of deep
Q is to learn the value of taking each possible action possible for a given input state. The policy used
once our Q net has been trained is simply to choose at every point the action for which the net predicts
the greatest reward. During training, a portion of actions are taken uniform-randomly in order to 
explore and learn about actions which the net believed to be bad, which they otherwise wouldnt have 
explored.  
Fir the loss function we use this trick that apparently works:
$Loss(state, action, reward, next_state) = forward(state) - (reward + discount*max(forward(next_state)))$
The Q net is supposed to represent the sum of expected rewards from now till forever. We define the "true"
Q value of a state-action pair as the reward we actually got, plus the Q value of the resulting state. We
teach the net to estimate all future rewards by only labelling the reward we got right now. This voodoo
apparently works for updating weights, but means the actual loss number doesnt really measure anything useful.
