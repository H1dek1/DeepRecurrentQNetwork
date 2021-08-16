from collections import deque
import random
import gym
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input, concatenate, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

class DQN:
    """
    Simple Deep Q-Network for time series Reinforcement Learning
    :param env: (Gym.Env) The environment constructed with Gym format.
    :param window_size: (int) The size of time series data input to this model.
    :param replay_buffer_size: (int) The max size of replay buffer.
    :param replay_batch_size: (int) size of a batched sample from replay buffer for training.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning.
    :param learning_rate: (float) learning rate for adam optimizer.
    :param gamma: (float) discount factor
    :param initial_eps: (float) initial value of random action probability.
    :param final_eps: (float) final value of random action probability.
    :param eps_change_length: (int) The number of episodes that is taken to change epsilon.
    """
    
    def __init__(self, env, window_size=12, replay_buffer_size=5000, replay_batch_size=128, learning_starts=1000, learning_rate=0.0025, gamma=0.99, initial_eps=1.0, final_eps=0.1, eps_change_length=1000, load_Qfunc=False, Qfunc_path=None):
        """
        Environment
        """
        self._env = env
        self._window_size = window_size
        self._n_action = self._env.action_space.n
        
        """
        Experience Replay
        """
        self._replay_buffer_size = replay_buffer_size
        self._replay_batch_size = replay_batch_size
        self._replay_buffer = deque(maxlen=self._replay_buffer_size)

        """
        Learning
        """
        self._learning_starts = learning_starts
        self._learning_rate = learning_rate
        self._gamma = gamma
        # For epsilon greedy
        self._initial_eps = initial_eps
        self._final_eps = final_eps
        self._eps_change_length = eps_change_length

        """
        Q-Function
        """
        if not load_Qfunc:
            self._init_Qfunc()
        else:
            self._Qfunc = None

        plot_model(self._Qfunc, show_shapes=True, show_layer_names=True)
        #print(self._Qfunc.summary())

    def _init_Qfunc(self):
        series_input = Input(shape=(self._window_size, 1), name='series_data')
        series_net = LSTM(64, return_sequences=True)(series_input)
        series_net = LSTM(32, return_sequences=False)(series_net)
        series_net = Dense(16, activation='relu')(series_net)
        series_net = Dense(16, activation='relu')(series_net)
        series_output = Dense(self._n_action, activation='linear')(series_net)

        self._Qfunc = Model(inputs=series_input, outputs=series_output)
        self._Qfunc.compile(optimizer=Adam(learning_rate=self._learning_rate), loss='mean_squared_error')

    def learn(self, total_timesteps):
        step_count = 0
        episode_count = 0
        history = {'epi_len': [], 'epi_rew': [], 'total_step': [], 'ave_loss': []}

        while True: # loop episodes
            epi_len = 0
            epi_rew = 0.0
            loss_history =[]
            obs = self._env.reset().reshape(1, -1, 1)
            while True: # loop steps
                # decide action
                action = self._decide_action(obs, episode_count)
                # proceed environment
                next_obs, reward, done, _ = self._env.step(action)
                next_obs = next_obs.reshape(1, -1, 1)
                # store experience
                self._replay_buffer.append( np.array([obs[0], action, reward, next_obs[0]], dtype=object) )
                # update observation
                obs = next_obs
                # increments
                step_count += 1
                epi_len += 1
                epi_rew += reward
                # experience replay
                if step_count > self._learning_starts:
                    loss = self._experience_replay()
                    loss_history.extend(loss.history['loss'])

                # judgement
                if done or step_count == total_timesteps:
                    print(f'Episode {episode_count}:    reward: {epi_rew}, remain step: {total_timesteps-step_count}')
                    break # from inside loop

            # each episode
            episode_count += 1
            history['total_step'].append(step_count)
            history['epi_len'].append(epi_len)
            history['epi_rew'].append(epi_rew)
            if len(loss_history) != 0:
                history['ave_loss'].append( sum(loss_history)/len(loss_history) )
            else:
                history['ave_loss'].append( 0 )
            # End Inside loop
            if step_count >= total_timesteps:
                break # from outside loop

        return history

    def _decide_action(self, obs, episode_count):
        if episode_count < self._eps_change_length:
            eps = self._initial_eps + (self._final_eps - self._initial_eps) * (episode_count/self._eps_change_length)
        else:
            eps = self._final_eps

        if eps < np.random.rand():
            # greedy
            #obs = obs.reshape(1, -1, 1)
            Q_values = self._Qfunc.predict(obs).flatten()
            action = np.argmax(Q_values)
            
        else:
            # random
            action = np.random.randint(0, self._n_action)

        return action
    
    def _experience_replay(self,):
        obs_minibatch = []
        target_minibatch = []
        action_minibatch = []

        # define sampling size
        minibatch_size = min(len(self._replay_buffer), self._replay_batch_size)
        # choose experience batch randomly
        minibatch = np.array(random.sample(self._replay_buffer, minibatch_size), dtype=object)
        obs_batch = np.stack(minibatch[:,0], axis=0)
        action_batch = minibatch[:, 1]
        reward_batch = minibatch[:, 2]
        next_obs_batch = np.stack(minibatch[:,3], axis=0)

        # current output of network
        current_Q_values = self._Qfunc.predict(obs_batch)
        # make target value
        target_Q_values = current_Q_values.copy()

        next_maxQ = np.max(self._Qfunc.predict(next_obs_batch), axis=1)
        for target_Q_value, action, reward, maxQ \
                in zip(target_Q_values, action_batch, reward_batch, next_maxQ):
            target_Q_value[action] = reward + self._gamma*maxQ

        loss = self._Qfunc.fit(
                       obs_batch,
                       target_Q_values,
                       epochs=1,
                       verbose=0
                       )
        return loss

    def simulate(self, visualize=False):
        obs = self._env.reset()
        done = False
        while not done:
            obs, reward, done, info = self._env.step(self._action(obs.reshape(1, -1, 1)))
            if visualize:
                self._env.render()

    def _action(self, obs):
        Q_values = self._Qfunc.predict(obs).flatten()
        return np.argmax(Q_values)



