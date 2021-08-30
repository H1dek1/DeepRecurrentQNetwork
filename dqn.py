from collections import deque
import random
import gym
import numpy as np

import keras
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Dense, LSTM, Lambda, concatenate
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import model_from_config
     
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
    
    def __init__(self, env, window_size=12, replay_buffer_size=1000, replay_batch_size=512, n_replay_epoch=1, learning_starts=1000, learning_rate=0.01, gamma=0.99, initial_eps=1.0, final_eps=0.01, eps_change_length=1000, load_Qfunc=False, Qfunc_path=None, use_doubleDQN=False, update_interval=100, target_update_interval=500, use_dueling=False):
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
        self._n_replay_epoch = n_replay_epoch

        """
        Learning
        """
        self._use_dueling = use_dueling
        self._use_doubleDQN = use_doubleDQN
        self._learning_starts = learning_starts
        self._learning_rate = learning_rate
        self._gamma = gamma
        # For epsilon greedy
        self._initial_eps = initial_eps
        self._final_eps = final_eps
        self._eps_change_length = eps_change_length
        self._update_interval = update_interval
        self._target_update_interval = target_update_interval

        """
        Q-Function
        """
        if not load_Qfunc:
            self._init_Qfunc(self._use_dueling)
        else:
            self._Qfunc = None

        plot_model(self._Qfunc, show_shapes=True, show_layer_names=True)
        print(self._Qfunc.summary())

        if self._use_doubleDQN:
            self._target_Qfunc = self._clone_network(self._Qfunc)

    def _init_Qfunc(self, use_dueling=False):
        if use_dueling:
            series_input = Input(shape=(self._window_size,), name='series_data')
            series_net = Dense(16, activation='relu')(series_input)
            #series_net = Dense(16, activation='relu')(series_net)
            # separate
            #state_value = Dense(16, activation='relu')(series_net)
            state_value = Dense(1, activation='linear', name='state_value')(series_net)
            #advantage = Dense(16, activation='relu')(series_net)
            advantage = Dense(self._n_action, activation='linear', name='advantage')(series_net)
            output = (state_value + (advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True)))

            # concatenate
            #concat = concatenate([state_value, advantage])
            ##output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.stop_gradient(K.mean(a[:, 1:], keepdims=True)), output_shape=(self._n_action,), name='Q_value')(concat)
            #output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(self._n_action,), name='Q_value')(concat)
        else:
            series_input = Input(shape=(self._window_size,), name='series_data')
            #series_input = Input(shape=(self._window_size, 1), name='series_data')
            #series_net = LSTM(64, return_sequences=True)(series_input)
            #series_net = LSTM(32, return_sequences=False)(series_net)
            series_net = Dense(16, activation='relu')(series_input)
            #series_net = Dense(32, activation='relu')(series_net)
            #series_net = Dense(16, activation='relu')(series_net)
            series_net = Dense(16, activation='relu')(series_net)
            output = Dense(self._n_action, activation='linear')(series_net)

        self._Qfunc = Model(inputs=series_input, outputs=output)
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
                if done and epi_len < 195:
                    reward = -1
                elif done and epi_len >= 195:
                    reward = 1
                else:
                    reward = 1
                next_obs = next_obs.reshape(1, -1, 1)
                # store experience
                self._replay_buffer.append( 
                        np.array([obs[0], action, reward, next_obs[0]], dtype=object) 
                        )
                # update observation
                obs = next_obs
                # increments
                step_count += 1
                epi_len += 1
                epi_rew += reward
                # experience replay
                if step_count > self._learning_starts \
                        and step_count%self._update_interval == 0:
                    if step_count%self._target_update_interval == 0:
                        update_targetQ = True
                    else:
                        update_targetQ = False
                    loss = self._experience_replay(update_targetQ)
                    loss_history.extend(loss.history['loss'])

                # judgement
                #if done or step_count == total_timesteps:
                if done:
                    if len(loss_history) != 0:
                        print(f'Episode {episode_count}:  reward: {epi_rew}, remain step: {total_timesteps-step_count}, loss: {np.average(loss_history)}')
                    else:
                        print(f'Episode {episode_count}:  reward: {epi_rew}, remain step: {total_timesteps-step_count}')
                    break # from inside loop

            # each episode
            episode_count += 1
            history['total_step'].append(step_count)
            history['epi_len'].append(epi_len)
            history['epi_rew'].append(epi_rew)
            if len(loss_history) != 0:
                history['ave_loss'].append( sum(loss_history)/len(loss_history) )
            else:
                history['ave_loss'].append( np.nan )
            # End inside loop
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
            Q_values = self._Qfunc.predict(obs).flatten()
            action = np.argmax(Q_values)
            
        else:
            # random
            action = np.random.randint(0, self._n_action)

        return action
    
    def _experience_replay(self, update_targetQ):
        obs_minibatch = []
        target_minibatch = []
        action_minibatch = []

        # define sampling size
        minibatch_size = min(len(self._replay_buffer), self._replay_batch_size)
        # choose experience batch randomly
        minibatch = np.array(random.sample(self._replay_buffer, minibatch_size),
                dtype=object)
        obs_batch = np.stack(minibatch[:,0], axis=0)
        action_batch = minibatch[:, 1]
        reward_batch = minibatch[:, 2]
        next_obs_batch = np.stack(minibatch[:,3], axis=0)

        # make target value
        current_Q_values = self._Qfunc.predict(obs_batch)
        target_Q_values = current_Q_values.copy()

        if self._use_doubleDQN:
            if update_targetQ:
                # parameter value from main Q to target Q
                self._target_Qfunc.set_weights(self._Qfunc.get_weights())

            next_actions = np.argmax(self._Qfunc.predict(next_obs_batch), axis=1)

            for next_obs, next_action, target_Q_value, action, reward \
                    in zip(
                            next_obs_batch, 
                            next_actions, 
                            target_Q_values, 
                            action_batch, 
                            reward_batch):
                next_q = self._target_Qfunc.predict(np.array([next_obs]))[0, next_action]
                target_Q_value[action] = reward + self._gamma*next_q
        else: # not use double DQN
            next_target_Q = np.max(self._Qfunc.predict(next_obs_batch), axis=1)
            for target_Q_value, action, reward, targetQ \
                    in zip(target_Q_values, action_batch, reward_batch, next_target_Q):
                target_Q_value[action] = reward + self._gamma*targetQ


        # update parameters
        loss = self._Qfunc.fit(
                       obs_batch,
                       target_Q_values,
                       epochs=self._n_replay_epoch,
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

    def _clone_network(self, model):
        config = {
                'class_name': model.__class__.__name__,
                'config': model.get_config(),
                }
        clone = model_from_config(config, custom_objects={})
        clone.set_weights(model.get_weights())
        return clone




