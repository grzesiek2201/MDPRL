# MDPRL

Required libraries are described in the 'requirements.txt' file.

The world should be in the 'config/world.txt' file.

There are two programs to run, mdpsolver.py (solving the MDP task) and qsolver.py (solving the RL task).

To run either of those:

<code>python mdpsolver.py</code> 

<code>python qsolver.py</code>

Optinal arguments: y(float) e(float) iter(int) in this order. 

To plot the state values with matplotlib:

in mdpsolver.py uncomment line 69

in qsolver.py uncomment line 71