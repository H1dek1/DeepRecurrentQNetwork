#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gym
from dqn import DQN

def main():
    env = gym.make('CartPole-v0')
    agent = DQN(env)
    print('Success to construct')


if __name__ == '__main__':
    main()
