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
