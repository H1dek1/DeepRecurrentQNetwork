#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gym
from dqn import DQN

def main():
    env = gym.make('CartPole-v0')
    agent = DQN(
            env=env, 
            window_size=4,
            replay_buffer_size=1000,
            learning_starts=1000,
            eps_change_length=500,
            learning_rate=0.01,
            n_replay_epoch=30,
            initial_eps=0.5,
            replay_batch_size=512,
            update_interval=100,
            use_doubleDQN=True,
            target_update_interval=300,
            use_dueling=True
            )
    print('Success to construct')
    history = agent.learn(total_timesteps=50000)
    #print(len(agent._replay_buffer))

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), tight_layout=True)
    ax[0].set_xlabel('total_step', fontsize=15)
    ax[0].set_ylabel('epi_len', fontsize=15)
    ax[0].plot(history['total_step'], history['epi_len'])
    ax[1].set_xlabel('total_step', fontsize=15)
    ax[1].set_ylabel('epi_rew', fontsize=15)
    ax[1].plot(history['total_step'], history['epi_rew'])
    ax[2].set_xlabel('total_step', fontsize=15)
    ax[2].set_ylabel('ave_loss', fontsize=15)
    ax[2].scatter(history['total_step'], history['ave_loss'], marker='.')
    fig.savefig('result.png')
    #plt.show()
    #agent.simulate(visualize=True)


if __name__ == '__main__':
    main()
