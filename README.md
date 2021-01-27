Project 1 of the Udacity Deep Reinforcement Learning Nanodegree Program

**Project Details**
The envionrment is the banana collection game from Unity ML Agents. An agent is dropped in a world, where it moves around on a plane to collect yellow 
bananas while avoiding blue ones. 
State Space: Continuous. 37 element vector describing states of the game such as position and velocity. 
Action Space: Discrete. 4 possible actions: move forward, backward, left, right. 
Reward: +1 reward for collecting a yellow banana, -1 reward for collecting a blue banana. 
Episodic task, where each episode is 300 steps through the environment. 
The task is considered solved when the agent is able to get an average total reward of at least 13 over 100 consecutive episodes. 

**Repository Layout**
- results/: the latest run from each of the 4 (so far) implemented algorithms, containing the score numpy file, plot of score, agent model, and progress log. 
- src/: source code for the agent, which is separated into code specific to agent itself, banana environment, and utilities
- Navigation.ipynb: this is simply a file copied over from the official repository. Safe to ignore.
- run.py: code run running, evaluating, and logging the agent. 

**Getting Started** 
Create a conda environment from the environment.yml file:
conda env create -f environment.yml

Activate the newly created environment:
conda activate drlnd

**Instructions**
At the time of writing (Nov 22, 2020), 4 algorithms have been implemented: DQN, Double DQN (DDQN), Dueling DDQN (D3QN), and DDQN with Prioritized Experience Replay(PDDQN)

Only DQN has been able to solve the task, with DDQN and D3QN trailing close behind in performance (likely a matter of hyerparam tuning), and PDDQN performance being abysmal. (likely faulty implementation). For review purposes, please use the DQN implementation. 

To run DQN without loading any weights, run the following:
```
python run.py DQN
```
Note: unless the directory path has been changed in run.py, this will overwrite the previous results.

