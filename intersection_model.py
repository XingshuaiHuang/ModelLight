from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, SimpleRNN, CuDNNLSTM
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import csv
from keras.optimizers import RMSprop
import numpy as np
from episode import BatchEpisodes
import copy
from frapplus_agent import FRAPPlusAgent
from utils import parse, config_all
import pickle
import os
import random


class IntersectionModel(object):
    def __init__(self, model_name='GP', policy=None):
        self.inter_model = None
        self.model_name = model_name
        self._build_model()
        self.policy = policy

    def _build_model(self):
        input_dim = 16
        output_dim = 9
        if self.model_name == 'GP':
            kernel = ConstantKernel(constant_value=1) * RBF(length_scale=0.5)
            self.inter_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)

        elif self.model_name == 'ANN':
            self.inter_model = Sequential()
            self.inter_model.add(Dense(144, input_dim=input_dim, activation='relu'))
            # self.inter_model.add(Dense(72, activation='relu'))
            # self.inter_model.add(Dense(72, activation='relu'))
            # self.inter_model.add(Dense(36, activation='relu'))
            self.inter_model.add(Dense(36, activation='relu'))
            self.inter_model.add(Dense(18, activation='relu'))
            # self.inter_model.add(Dropout(0.5))
            self.inter_model.add(Dense(output_dim))
            self.inter_model.summary()
            self.inter_model.compile(loss='mean_squared_error', optimizer='adam')

        elif self.model_name == 'LSTM':
            self.inter_model = Sequential()
            self.inter_model.add(LSTM(16, input_shape=(1, input_dim), return_sequences=True))
            # self.inter_model.add(LSTM(128, return_sequences=True))
            # self.inter_model.add(LSTM(64, return_sequences=True))
            self.inter_model.add(LSTM(64))
            # self.inter_model.add(Dropout(0.2))
            self.inter_model.add(Dense(output_dim))  # units = 9 denotes the output_size
            self.inter_model.compile(loss='mean_squared_error', optimizer='adam')

    @staticmethod
    def sample_experience(tot_samples, slice_index, dic_action_phase):
        tot_state_action = []
        tot_next_state_reward = []
        for i in slice_index:
            sample = tot_samples[i]  # ith transition
            observation = sample[0]
            lane_num_vehicle = observation[0]['lane_num_vehicle']  # [1,2,3,4,5,6,7,8], without "cur_phase"
            action = sample[1]  # [0]
            new_observation = sample[2]
            new_lane_num_vehicle = new_observation[0]['lane_num_vehicle']
            reward = [sample[3][0] * 4]  # [0.25 * 4] convert to an int for convenience of calculation
            phase = dic_action_phase[str(action)]
            # state_action = lane_num_vehicle + action  # [1,2,3,4,5,6,7,8, 0]
            state_action = lane_num_vehicle + phase  # [1,2,3,4,5,6,7,8, 0,1,0,0,1,0,0,0]
            next_state_reward = new_lane_num_vehicle + reward  # [1,2,3,4,5,6,7,8, 0.0]
            tot_state_action.append(state_action)
            tot_next_state_reward.append(next_state_reward)
            # print(observation, action, new_observation, reward, state_action, next_state_reward)
        return np.array(tot_state_action), np.array(tot_next_state_reward)

    def train(self, tot_samples, slice_index, dic_action_phase,
              epochs):  # len(slice_index) == dic_agent_conf['MODEL_SAMPLE_SIZE']
        tot_state_action, tot_next_state_reward = self.sample_experience(tot_samples, slice_index, dic_action_phase)
        # print('-----------tot_state_action is: {}-----------'.format(tot_state_action))
        # print('-----------tot_next_state_reward is: {}-----------'.format(tot_next_state_reward))
        if self.model_name == 'GP':
            self.inter_model.fit(tot_state_action, tot_next_state_reward)
        elif self.model_name == 'ANN':
            self.inter_model.fit(tot_state_action, tot_next_state_reward, epochs=epochs, batch_size=10, verbose=2)
        elif self.model_name == 'LSTM':
            # reshape input to be [samples, time steps, features(input_size/dimension)]
            tot_state_action = np.reshape(tot_state_action, (tot_state_action.shape[0], 1, tot_state_action.shape[1]))
            self.inter_model.fit(tot_state_action, tot_next_state_reward, epochs=epochs, batch_size=10, verbose=2)

    def predict(self, lane_num_veh=None, action=None):  # lane_num_veh: [1,2,3,4,5,6,7,8], action: [0]
        state_action = np.array(lane_num_veh + action).reshape(1, -1)
        # print('@@@@@@@@@@@@@@@@@@@ state_action is {} @@@@@@@@@@@@@@@@@@'.format(state_action))
        if self.model_name == 'LSTM':
            state_action = np.reshape(state_action, (state_action.shape[0], 1, state_action.shape[1]))
        prediction = self.inter_model.predict(state_action)
        return prediction

    def save_params(self, path_to_env_model, batch_id=None, task_id=None, task_for_params=None):
        if not os.path.exists(path_to_env_model):
            os.makedirs(path_to_env_model)
        if self.model_name == 'GP':
            return
        else:
            if task_for_params:
                self.inter_model.save(os.path.join(path_to_env_model, 'env_model_{0}.h5'.format(task_for_params)))
            else:
                self.inter_model.save(os.path.join(path_to_env_model, 'env_model_{0}_{1}.h5'.format(batch_id, task_id)))

    def load_params(self, path_to_env_model, batch_id=None, task_id=None, task_for_params=None):
        if self.model_name == 'GP':
            return
        else:
            if task_for_params:
                self.inter_model = load_model(
                    os.path.join(path_to_env_model, 'env_model_{0}.h5'.format(task_for_params)))
            else:
                self.inter_model = load_model(
                    os.path.join(path_to_env_model, 'env_model_{0}_{1}.h5'.format(batch_id, task_id)))

    def accu_eval(self, tot_samples, slice_index, dic_action_phase, num_valid=50):
        print('---accu_eval---')
        if len(tot_samples) - len(slice_index) < num_valid:
            return
        left_index = []
        for i in range(len(tot_samples)):
            if i not in slice_index:
                left_index.append(i)

        valid_index = random.sample(left_index, num_valid)  # number of samples in valid set: 50
        num_accu = 0
        pred_list = []
        label_list = []

        for j in valid_index:
            sample = tot_samples[j]  # ith transition
            observation = sample[0]
            lane_num_veh = observation[0]['lane_num_vehicle']  # [1,2,3,4,5,6,7,8], without "cur_phase"
            action = sample[1]  # [0]
            new_observation = sample[2]
            new_lane_num_veh = new_observation[0]['lane_num_vehicle']
            reward = sample[3]  # [0.25]
            next_state_reward = new_lane_num_veh + reward  # [1,2,3,4,5,6,7,8, 0.0]
            print('-----next_state_reward is {}-----'.format(next_state_reward))

            phase = dic_action_phase[str(action)]
            # pred = self.predict(list(lane_num_veh), list(action))  # next_lane_num_veh + reward
            pred = self.predict(list(lane_num_veh), list(phase))  # next_lane_num_veh + reward
            next_lane_num_veh = [int(round(x)) for x in pred[0][0:8]]  # [0, 0, 0, 0, 0, 0, 0, 0]
            pred_reward = [int(pred[0][8]) / 4]  # [0.25]
            pred_next_state_reward = next_lane_num_veh + pred_reward
            print('-----pred_next_state_reward is {}-----'.format(pred_next_state_reward))

            pred_list.append(pred_next_state_reward)
            label_list.append(next_state_reward)

        print('--------------mae is {}; mse is {}----------------'.format(mae(label_list, pred_list),
                                                                          mse(label_list, pred_list)))
        # if next_state_reward == pred_next_state_reward:
        #     num_accu += 1
        # print('---------------validation accuracy is: {}---------------'.format(num_accu / num_valid))

        # write down evaluation metrics to .csv file
        metrics = open('summary/ann_metrics.csv', 'a', newline='')
        csv_write = csv.writer(metrics, dialect='excel')
        csv_write.writerow([mae(label_list, pred_list), mse(label_list, pred_list)])
        print("write over")

    def sample_e(self, start_episode, episodes, params=None, dic_action_phase=None,
                 num_img_trans=36):  # sample imaginary trajectories as Ei or Ei'
        if params:  # load parameters: θ or θi
            self.policy.load_params(params)
        num_img_round = 1
        for img_round in range(num_img_round):
            start_sample_id = np.random.randint(len(start_episode.total_samples))
            observation = start_episode.total_samples[start_sample_id][
                0]  # [{'cur_phase': [0, 1, 0, 0, 0, 0, 0, 1], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]
            observations = np.reshape(observation, (
            -1, 1))  # [[{'cur_phase': [0, 1, 0, 0, 0, 0, 0, 1], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]]
            lane_num_veh = observation[0]['lane_num_vehicle']  # [0, 0, 0, 0, 0, 0, 0, 0], without "cur_phase"
            # action = start_sample[1]  # [0]

            for i in range(num_img_trans):  # number of imaginary transitions
                # print(np.array(state + action).reshape(1, -1))

                # print('@@@@@@@@@@@@@@@@@@@ observations is {} @@@@@@@@@@@@@@@@@@'.format(observations))
                action = self.policy.choose_action(observations)  # [0]
                actions = np.reshape(action, (-1, 1))  # [[0]]

                # print('@@@@@@@@@@@@@@@@@@@ lane_num_veh is {} @@@@@@@@@@@@@@@@@@'.format(lane_num_veh))
                # print('$$$$$$$$$$$$$$$$$$$ action is {} $$$$$$$$$$$$$$$$$$$'.format(action))
                phase = dic_action_phase[str(action)]  # convert action to phase
                # pred = self.predict(list(lane_num_veh), list(action))  # next_lane_num_veh + reward
                pred = self.predict(list(lane_num_veh), list(phase))  # next_lane_num_veh + reward
                # next_lane_num_veh = [abs(int(round(x))) for x in pred[0][0:8]]  # [0, 0, 0, 0, 0, 0, 0, 0]
                next_lane_num_veh = []  # [0, 0, 0, 0, 0, 0, 0, 0]
                for x in pred[0][0:8]:  # limit x to [0, 100]
                    x = int(round(x))
                    if x < 0:
                        next_lane_num_veh.append(0)
                    elif x < 100:
                        next_lane_num_veh.append(x)
                    else:
                        next_lane_num_veh.append(100)

                # print('----------dic_action_phase is: {}'.format(dic_action_phase))
                next_observation = [
                    {'cur_phase': phase, 'lane_num_vehicle': next_lane_num_veh}]
                next_observations = np.reshape(next_observation, (-1, 1))

                reward = [int(pred[0][8]) / 4]  # [0.25]
                rewards = np.reshape(reward, (-1, 1))

                episodes.append(observations, actions, next_observations, rewards,
                                batch_ids=[[0]])  # batch_ids shouldn't be None

                lane_num_veh = next_lane_num_veh
                observations = next_observations

            # if i < 100:
            #     print('-----pred observations | rewards are: {} | {}-----'.format(observations, rewards))

    def gene_transition(self, observation, episodes, params=None,
                        dic_action_phase=None):  # sample imaginary trajectories as Ei or Ei'
        observations = np.reshape(observation, (
        -1, 1))  # [[{'cur_phase': [0, 1, 0, 0, 0, 0, 0, 1], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]]
        lane_num_veh = observation[0]['lane_num_vehicle']  # [0, 0, 0, 0, 0, 0, 0, 0], without "cur_phase"
        # action = start_sample[1]  # [0]

        if params:  # load parameters: θ or θi
            self.policy.load_params(params)

        # print('@@@@@@@@@@@@@@@@@@@ observations is {} @@@@@@@@@@@@@@@@@@'.format(observations))
        action = self.policy.choose_action(observations)  # [0]
        actions = np.reshape(action, (-1, 1))  # [[0]]

        # print('@@@@@@@@@@@@@@@@@@@ lane_num_veh is {} @@@@@@@@@@@@@@@@@@'.format(lane_num_veh))
        # print('$$$$$$$$$$$$$$$$$$$ action is {} $$$$$$$$$$$$$$$$$$$'.format(action))
        phase = dic_action_phase[str(action)]  # convert action to phase
        # pred = self.predict(list(lane_num_veh), list(action))  # next_lane_num_veh + reward
        pred = self.predict(list(lane_num_veh), list(phase))  # next_lane_num_veh + reward
        # next_lane_num_veh = [abs(int(round(x))) for x in pred[0][0:8]]  # [0, 0, 0, 0, 0, 0, 0, 0]
        next_lane_num_veh = []  # [0, 0, 0, 0, 0, 0, 0, 0]
        for x in pred[0][0:8]:  # limit x to [0, 100]
            x = int(round(x))
            if x < 0:
                next_lane_num_veh.append(0)
            elif x < 100:
                next_lane_num_veh.append(x)
            else:
                next_lane_num_veh.append(100)

        # print('----------dic_action_phase is: {}'.format(dic_action_phase))
        next_observation = [{'cur_phase': phase, 'lane_num_vehicle': next_lane_num_veh}]
        next_observations = np.reshape(next_observation, (-1, 1))

        reward = [int(pred[0][8]) / 4]  # [0.25]
        rewards = np.reshape(reward, (-1, 1))

        episodes.append(observations, actions, next_observations, rewards,
                        batch_ids=[[0]])  # batch_ids shouldn't be None

        # if i < 100:
        #     print('-----pred observations | rewards are: {} | {}-----'.format(observations, rewards))
        return next_observation

    def gene_img_traj(self, start_sample, episodes, params=None, dic_action_phase=None):

        observation = start_sample[0]  # [{'cur_phase': [0], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]
        observations = np.reshape(observation,
                                  (-1, 1))  # [[{'cur_phase': [0], 'lane_num_vehicle': [0, 0, 0, 0, 0, 0, 0, 0]}]]
        lane_num_veh = observation[0]['lane_num_vehicle']  # [0, 0, 0, 0, 0, 0, 0, 0], without "cur_phase"
        # action = start_sample[1]  # [0]

        if params:  # load parameters: θ or θi
            self.policy.load_params(params)

        for _ in range(self.num_img_trans):  # number of imaginary transitions
            # print(np.array(state + action).reshape(1, -1))

            action = self.policy.choose_action(observations)  # [0]
            actions = np.reshape(action, (-1, 1))  # [[0]]

            # print('@@@@@@@@@@@@@@@@@@@ lane_num_veh is {} @@@@@@@@@@@@@@@@@@'.format(lane_num_veh))
            # print('$$$$$$$$$$$$$$$$$$$ action is {} $$$$$$$$$$$$$$$$$$$'.format(action))
            pred = self.predict(list(lane_num_veh), list(action))  # next_lane_num_veh + reward
            next_lane_num_veh = [int(round(x)) for x in pred[0][0:8]]  # [0, 0, 0, 0, 0, 0, 0, 0]

            # print('----------dic_action_phase is {}'.format(dic_action_phase))
            cur_phase = dic_action_phase[str(action)]
            next_observation = [
                {'cur_phase': cur_phase, 'lane_num_vehicle': next_lane_num_veh}]
            next_observations = np.reshape(next_observation, (-1, 1))

            reward = [int(pred[0][8]) / 4]  # [0.25]
            rewards = np.reshape(reward, (-1, 1))

            episodes.append(observations, actions, next_observations, rewards,
                            batch_ids=[[0]])  # batch_ids shouldn't be None

            lane_num_veh = next_lane_num_veh
            observations = next_observations

            print('-----pred observations | rewards are: {} | {}-----'.format(observations, rewards))

