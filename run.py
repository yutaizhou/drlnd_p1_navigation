from os import stat
from unityagents import UnityEnvironment
import numpy as np

from src.agents.agent import DQNAgent


def rollout(agent, env: UnityEnvironment, is_training: bool = True):
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


def run(agent, env: UnityEnvironment, num_episodes=10000, is_training=True):
    scores = []
    max_avg_score = -np.inf

    for i_episode in range(1, num_episodes+1):
        total_reward = rollout(agent, env, is_training)
        scores.append(total_reward)

        if is_training:
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score: max_avg_score = avg_score
            
            if i_episode % 100 == 0:
                print(f'Episode {i_episode}/{num_episodes} | Max Average Score: {max_avg_score}')
    
    return scores


if __name__ == "__main__":
    # set up env
    env = UnityEnvironment(file_name="src/envs/Banana_Linux_NoVis/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = DQNAgent(state_size, action_size)

    scores = run(agent, env)




