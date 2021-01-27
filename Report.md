## Learning Algorithm

The most basic **Deep Q Network (DQN)** algorithm as described in the [2015 paper][dqn_paper] with a target network and uniform sampling experience replay buffer is used to solve the task. The deep neural network (DNN) used to approximation the Q function is a simple fully connected feedforward neural network, with |S| input neurons and |A| output neurons that calculate the Q value for each action available at a given state. 

DQN is an off-policy value-based algorithm that learns a Q-function (represented by a DNN) to represent the action-value of each state-action pair that an agent encounters in the world. The Q-function is updated by backpropagating the gradient of the mean squared error between a "target" quantity and a "current" quantity. The former is the one-step temporal difference (TD) prediction that sums the reward obtained at the current timestep and the Q-value prediction at the next state, subject to using an action that maximizes the predicted Q-value at the next state. The latter is the Q-value prediction at the current state and action. The difference between the target and the current quantity is also known as the TD residual. The intuition is that the actual reward obtained at the current timestep is unbiased, so it serves to reduce the bias of the TD prediction at the current timestep, which is a biased estimator of the expected return. The mean squared TD residual is what drives the learning of the DNN used to represent the Q-function. 

DQN is considered value-based and not policy-based because it explicitly represents and learns a Q-value function, yet the policy is simply derived from the learned Q-value function by means of greedy maximization over possible actions. 
DQN is considered off-policy because its learned/target policy is not the same as its behavior policy, due to the Q-learning update using an argmax operator over all possible next actions for next-state Q-value estimation.

_Hyperparameters_
- LR = 5e-4 # learning rate
- BUFFER_SIZE = 1e5 # size of experience replay buffer
- BATCH_SIZE = 64 # the number of (state, action, next_state, reward, done) tuples to use at once to estimate the TD residual error
- GAMMA = 0.99 # discount factor for return
- UPDATE_FREQ = 4 # how often to update the frozen target network with the params of the changing local network

## Reward Plot

![Reward Plot][reward_plot]

The task was solved in 1146 episodes.


## Future Work

Since only the most basic version of DQN as described in the [2015 paper][dqn_paper] was implemented, a number of extensions from as simple as a few lines of code change ([Double DQN][ddqn_paper], [Dueling DQN][d3qn_paper]) to more complicated add-ons involving redesigning replay buffer and update rule([Prioritized Experience Replay][pddqn_paper]) are available. All of which have been implemented and runs without crashing, but further hyperparameter tuning and debugging will be required to improve the result of the DQN baseline. Ultimately, [Rainbow DQN][rainbow], which combines many proposed DQN extensions, will be implemented for coding practice. 

<!-- Links -->
[reward_plot]: https://github.com/yutaizhou/drlnd_p1_navigation/blob/master/results/DQN/result.png
[dqn_paper]: https://www.nature.com/articles/nature14236
[ddqn_paper]: https://arxiv.org/abs/1509.06461
[d3qn_paper]: https://arxiv.org/abs/1511.06581
[pddqn_paper]: https://arxiv.org/abs/1511.05952
[rainbow]: https://arxiv.org/abs/1710.02298
