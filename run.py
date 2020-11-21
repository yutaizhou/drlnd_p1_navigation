from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.agents.agent import DQN, DDQN, D3QN
from src.utils.typing import List


def rollout(agent, env: UnityEnvironment, is_training: bool = True):
    # completes one episode of rollout, is_training functionality not really fully fledged out
    env_info = env.reset(train_mode=True)[brain_name] 
    
    state = env_info.vector_observations[0]
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state, is_training)

        env_info = env.step(action)[brain_name]
        next_state, reward, done = env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0]  

        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    return total_reward


def run(agent, env: UnityEnvironment, num_episodes=10000, is_training=True) -> List[float]:
    scores = []
    max_avg_score = -np.inf
    solved = False

    log_file = open(f'results/{agent}/progress.log', mode='w')
    print(f'Progress for {agent} agent', file=log_file)
    for i_episode in range(1, num_episodes+1):
        total_reward = rollout(agent, env, is_training)
        scores.append(total_reward)

        if is_training:
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                max_avg_score = max(max_avg_score, avg_score)
            
            if i_episode % 100 == 0:
                print(f'Episode {i_episode}/{num_episodes} | Max Average Score: {max_avg_score}', file=log_file)
            if max_avg_score >= 13 and not solved:
                print(f'Task solved in {i_episode} episodes, with average score over the last 100 episode: {max_avg_score}', file=log_file)
                solved = True

    log_file.close()
    torch.save(agent.Q_local.state_dict(), f'results/{agent}/checkpoint.pth')
    with open(f'results/{agent}/scores.npy', 'wb') as f:
        np.save(f, scores)

    return scores


if __name__ == "__main__":
    # set up env
    env = UnityEnvironment(file_name="src/envs/Banana_Linux_NoVis/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    algorithm = DQN
    agent = algorithm(state_size, action_size)

    scores = run(agent, env, num_episodes=2000)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig(f'results/{agent}/result.png')
    plt.close(fig)
