#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gym
from dqn import DQN

def main():
    env = gym.make('CartPole-v0')
    agent = DQN(env, window_size=4, initial_eps=0.1, learning_starts=10)
    print('Success to construct')
    history = agent.learn(total_timesteps=20)
    print(len(agent._replay_buffer))

    #fig, ax = plt.subplots(1, 2)
    #ax[0].plot(range(len(history['epi_len'])), history['epi_len'])
    #ax[1].plot(range(len(history['epi_rew'])), history['epi_rew'])
    #plt.show()


if __name__ == '__main__':
    main()
