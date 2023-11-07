import multiprocessing as mp
from episode import BatchEpisodes, SeperateEpisode
from cityflow_env import CityFlowEnv
import json
import os
import shutil
import random
import copy
import numpy as np
import xml.etree.ElementTree as ET
import pandas as pd
from math import isnan
from subproc_vec_env import SubprocVecEnv
from utils import write_summary, model_write_summary
import pickle
from intersection_model import IntersectionModel


class BatchSampler(object):
    def __init__(self, dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                 dic_path, batch_size, num_workers=2):
        """
            Sample trajectories in one episode by different methods
        """
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.task_path_map = {}
        self.task_traffic_env_map = {}
        if not isinstance(self.dic_traffic_env_conf, list):
            self.list_traffic_env_conf = [self.dic_traffic_env_conf]
            self.list_path = [self.dic_path]  # turn the dic_path into a list
            task = self.dic_path["PATH_TO_DATA"].split("/")[-1] + ".json"
            self.task_path_map[task] = self.dic_path
            self.task_traffic_env_map[task] = self.dic_traffic_env_conf
        else:
            self.list_traffic_env_conf = self.dic_traffic_env_conf
            self.list_path = self.dic_path
            for path in self.dic_path:
                task = path["PATH_TO_DATA"].split("/")[-1] + ".json"
                self.task_path_map[task] = path
            for env in self.dic_traffic_env_conf:
                task = env["TRAFFIC_FILE"]
                self.task_traffic_env_map[task] = env

        # num of episodes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.queue = mp.Queue()
        self.envs = None  # reset_task function: self.envs = SubprocVecEnv()
        self._task_id = 0

        self._path_check()
        self._copy_conf_file()
        # self._copy_cityflow_file() 

        self.path_to_log = self.list_path[0]['PATH_TO_WORK_DIRECTORY']

        self.step = 0
        self.target_step = 0
        self.lr_step = 0

        self.test_step = 0

    def _path_check(self):
        # check path
        if not os.path.exists(self.list_path[0]["PATH_TO_WORK_DIRECTORY"]):
            os.makedirs(self.list_path[0]["PATH_TO_WORK_DIRECTORY"])

        if not os.path.exists(self.list_path[0]["PATH_TO_MODEL"]):
            os.makedirs(self.list_path[0]["PATH_TO_MODEL"])

        if not os.path.exists(self.list_path[0]["PATH_TO_GRADIENT"]):
            os.makedirs(self.list_path[0]["PATH_TO_GRADIENT"])

        if self.dic_exp_conf["PRETRAIN"]:
            if os.path.exists(self.list_path[0]["PATH_TO_PRETRAIN_WORK_DIRECTORY"]):
                pass
            else:
                os.makedirs(self.list_path[0]["PATH_TO_PRETRAIN_WORK_DIRECTORY"])

            if os.path.exists(self.list_path[0]["PATH_TO_PRETRAIN_MODEL"]):
                pass
            else:
                os.makedirs(self.list_path[0]["PATH_TO_PRETRAIN_MODEL"])

    def _copy_conf_file(self, path=None):
        # write conf files
        if path == None:
            path = self.list_path[0]["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
                  indent=4)
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_cityflow_file(self, path=None):
        if path == None:
            path = self.list_path[0]["PATH_TO_WORK_DIRECTORY"]

        for traffic in self.dic_exp_conf["TRAFFIC_IN_TASKS"]:
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], traffic),
                        os.path.join(path, traffic))
            shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["ROADNET_FILE"]),
                        os.path.join(path, self.dic_traffic_env_conf["ROADNET_FILE"]))

    def model_based_sample_traj(self, policy, task=None, batch_id=None, params=None):
        dic_action_phase = dict()

        def action_phase(action, next_cur_phase, act_pha):
            act_pha[str(action)] = next_cur_phase
            return act_pha

        for i in range(self.batch_size):  # 1
            self.queue.put(i)  # q.put()来入列
        for _ in range(self.num_workers):
            self.queue.put(None)
        episodes = BatchEpisodes(dic_agent_conf=self.dic_agent_conf)
        observations, batch_ids = self.envs.reset()  # envs = SubprocVecEnv() after reset_task(); batch_ids != batch_id
        dones = [False]
        if params: 
            policy.load_params(params)
        dones_step = 0
        while (not all(dones)) or (not self.queue.empty()):  # q.empty()判断当前队列中是否还有值，ends until dones==True
            actions = policy.choose_action(observations)
            # for multi_intersection
            actions = np.reshape(actions, (-1, 1))  # matrix with 1 column; actions[1]
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards,
                            batch_ids)  # append is a function defined in BatchEpisodes
            observations, batch_ids = new_observations, new_batch_ids

            # print('---dones_step is: {0} | dones is: {1}---'.format(dones_step, dones))
            dones_step += 1

            if not actions[0][0] in dic_action_phase:
                dic_action_phase = action_phase(actions[0], new_observations[0][0]['cur_phase'], dic_action_phase)
        # self.envs.bulk_log()
        return episodes, dic_action_phase

    def sample_maml(self, policy, task=None, batch_id=None, params=None):
        for i in range(self.batch_size):
            self.queue.put(i)  # q.put()来入列
        for _ in range(self.num_workers):
            self.queue.put(None)
        episodes = BatchEpisodes(
            dic_agent_conf=self.dic_agent_conf)  # different from sample_sotl; self.total_samples = []
        observations, batch_ids = self.envs.reset() 
        dones = [False]
        if params: 
            policy.load_params(params)
        while (not all(dones)) or (not self.queue.empty()):  # different from sample_sotl, q.empty()判断当前队列中是否还有值
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards,
                            batch_ids)  # append is a function defined in BatchEpisodes
            observations, batch_ids = new_observations, new_batch_ids
        # self.envs.bulk_log()
        return episodes

    def sample_sotl(self, policy, task=None, batch_id=None, params=None):
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params:
            policy.load_params(params)
        while (not all(dones)):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            observations, batch_ids = new_observations, new_batch_ids
        write_summary(self.dic_path, task, self.dic_exp_conf["EPISODE_LEN"], 0,  # different from sample_maml
                      self.dic_traffic_env_conf['FLOW_FILE'])
        # self.envs.bulk_log()

    def sample_mbmrl(self, policy, tasks, batch_id, params=None, target_params=None, episodes=None):
        def action_phase(action, next_cur_phase, act_pha):
            act_pha[str(action)] = next_cur_phase
            return act_pha

        # *******************************************sample real trajectory*********************************************
        for i in range(len(tasks)): 
            self.queue.put(i)
        for _ in range(len(tasks)): 
            self.queue.put(None)

        if not episodes:  # episodes: replay buffer
            size = int(len(tasks) / self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"])
            episodes = SeperateEpisode(size=size, group_size=self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"],  # 1
                                       dic_agent_conf=self.dic_agent_conf)
        dic_action_phase_0 = dict()
        dic_action_phase_1 = dict()
        observations, batch_ids = self.envs.reset()
        # print('--------init_observations is {}--------'.format(observations))
        dones = [False]
        if params:  
            policy.load_params(params)
            # print(params[0]['final_conv_w1'], params[1]['final_conv_w1'])

        old_params = None
        meta_update_period = 1
        meta_update = False
        # new_observations = []

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)  
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            # print('--------actions is {}--------'.format(actions))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            # print('-------new_observations is {}--------'.format(new_observations))
            '''
            observations is:
            [[{'cur_phase': [0, 0, 0, 0, 1, 1, 0, 0], 'lane_num_vehicle': [5, 23, 2, 8, 0, 0, 2, 16]}],
            [{'cur_phase': [0, 1, 0, 0, 0, 0, 0, 1], 'lane_num_vehicle': [1, 15, 2, 8, 0, 2, 0, 15]}]]
            '''
            episodes.append(observations, actions, new_observations, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
            if not actions[0][0] in dic_action_phase_0:
                dic_action_phase_0 = action_phase(actions[0], new_observations[0][0]['cur_phase'], dic_action_phase_0)
            if not actions[1][0] in dic_action_phase_1:
                dic_action_phase_1 = action_phase(actions[1], new_observations[1][0]['cur_phase'], dic_action_phase_1)

            # if update
            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf[
                'UPDATE_PERIOD'] == 0:  # 0, 1

                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:  # 2000
                    episodes.forget(self.dic_agent_conf['MAX_MEMORY_LEN'])  # ***

                old_params = params

                # update adapted params
                policy.fit(episodes, params=params, target_params=target_params)
                sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))  # 30
                # slice_index = random.sample(range(len(episodes)), sample_size)
                slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                # print('---------------slice_index is {}------------'.format(slice_index))
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)
                policy.load_params(params)

                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:  # 5
                    target_params = params
                    self.target_step = 0

                # meta update
                if meta_update_period % self.dic_agent_conf["META_UPDATE_PERIOD"] == 0:  # 10
                    policy.fit(episodes, params=params, target_params=target_params)
                    sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                    # new_slice_index = random.sample(range(len(episodes)), sample_size)
                    new_slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                    # print('---------------new_slice_index is {}------------'.format(new_slice_index))
                    params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
                    policy.load_params(params)

                meta_update_period += 1

            self.step += 1
        dic_action_phase = [dic_action_phase_0, dic_action_phase_1]
        # print(episodes.episodes_inter[1].total_samples)
        # **************************************************************************************************************

        # *******************************************train intersection model*******************************************
        # print('----------dic_action_phase is {}----------'.format(dic_action_phase))
        model_training_sample_size = min(self.dic_agent_conf['MODEL_TRAINING_SAMPLE_SIZE'], len(episodes))
        # model_slice_index = random.sample(range(len(episodes)), model_training_sample_size)
        model_slice_index = np.random.choice(len(episodes), model_training_sample_size, replace=False)
        # print('---------------model_slice_index is {}------------'.format(model_slice_index))
        env_models = []
        for j in range(len(tasks)):
            inter_model = IntersectionModel(model_name='ANN', policy=policy.policy_inter[j])  # policy: fθ
            print('-------------------task{0} is: {1}--------------------'.format(j, tasks[j]))

            """muti model"""
            if os.path.exists(os.path.join(self.dic_path[j]["PATH_TO_ENV_MODEL"], 'env_model_{0}.h5'.format(tasks[j]))):
                inter_model.load_params(path_to_env_model=self.dic_path[j]["PATH_TO_ENV_MODEL"],
                                        task_for_params=tasks[j])
            """single model"""
            # if os.path.exists(os.path.join(self.dic_path[j]["PATH_TO_ENV_MODEL"], 'env_model_0.h5')):
            #     inter_model.load_params(path_to_env_model=self.dic_path[j]["PATH_TO_ENV_MODEL"], task_for_params=0)
            # if batch_id > 0:  # 2 different models
            #     inter_model.load_params(path_to_env_model=self.dic_path[j]["PATH_TO_ENV_MODEL"], batch_id=batch_id-1, task_id=j)

            inter_model.train(list(episodes.episodes_inter[j].total_samples),
                              model_slice_index, dic_action_phase[j], self.dic_agent_conf["EPOCHS"])

            """muti model"""
            inter_model.save_params(path_to_env_model=self.dic_path[j]["PATH_TO_ENV_MODEL"], task_for_params=tasks[j])
            """single model"""
            # inter_model.save_params(path_to_env_model=self.dic_path[j]["PATH_TO_ENV_MODEL"], task_for_params=0)
            # inter_model.save_params(path_to_env_model=self.dic_path[j]["PATH_TO_ENV_MODEL"], batch_id=batch_id, task_id=j)

            inter_model.accu_eval(list(episodes.episodes_inter[j].total_samples), model_slice_index,
                                  dic_action_phase[j])
            print('============================inter_model train complete============================')
            env_models.append(inter_model)
        # **************************************************************************************************************

        # *****************************************sample imaginary trajectory******************************************
        # new episode storing imaginary transitions
        num_img_trans = self.dic_agent_conf["NUM_IMG_TRANS"]
        num_img_round = self.dic_agent_conf["NUM_IMG_ROUND"]
        for img_round in range(num_img_round):
            start_sample_id = np.random.randint(len(episodes.episodes_inter[0].total_samples))
            observations_0 = episodes.episodes_inter[0].total_samples[start_sample_id]
            observations_1 = episodes.episodes_inter[1].total_samples[start_sample_id]
            observations = [observations_0[0], observations_1[0]]
            # print('----------------start_random_observations is {}---------------'.format(observations))
            for _ in range(num_img_trans):
                _observations = []
                for k in range(len(tasks)):
                    next_observation = env_models[k].gene_transition(observations[k], episodes.episodes_inter[k],
                                                                     params=params[k],
                                                                     dic_action_phase=dic_action_phase[k])
                    _observations.append(next_observation)
                observations = _observations
                print('-----------len(episodes) is {}-----------'.format(len(episodes.episodes_inter[0].total_samples)))
                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:  # 2000
                    episodes.forget(self.dic_agent_conf['MAX_MEMORY_LEN'])  # ***

                old_params = params

                policy.fit(episodes, params=params, target_params=target_params)
                sample_size = min(self.dic_agent_conf['MODEL_SAMPLE_SIZE'], len(episodes))  # 100
                slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                # print('---------------slice_index is {}------------'.format(slice_index))
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)
                policy.load_params(params)

                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:  # 5
                    target_params = params
                    self.target_step = 0

                # meta update
                if meta_update_period % self.dic_agent_conf["META_UPDATE_PERIOD"] == 0:  # 10
                    policy.fit(episodes, params=params, target_params=target_params)
                    sample_size = min(self.dic_agent_conf['MODEL_SAMPLE_SIZE'], len(episodes))
                    # new_slice_index = random.sample(range(len(episodes)), sample_size)
                    new_slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                    # print('---------------new_slice_index is {}------------'.format(new_slice_index))
                    params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
                    policy.load_params(params)

                meta_update_period += 1

        print('============================sample img_trans complete============================')
        # **************************************************************************************************************

        if not meta_update:  # meta update after a task?
            policy.fit(episodes, params=params, target_params=target_params)
            sample_size = min(self.dic_agent_conf['MODEL_SAMPLE_SIZE'], len(episodes))
            # new_slice_index = random.sample(range(len(episodes)), sample_size)
            new_slice_index = np.random.choice(len(episodes), sample_size, replace=False)
            params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
            policy.load_params(params)

            meta_update_period += 1
        policy.decay_epsilon(batch_id)
        return params[0]

        # self.envs.bulk_log()

    def sample_metalight(self, policy, tasks, batch_id, params=None, target_params=None, episodes=None):
        for i in range(len(tasks)):  
            self.queue.put(i)
        for _ in range(len(tasks)):  
            self.queue.put(None)

        if not episodes:
            size = int(len(tasks) / self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"])
            episodes = SeperateEpisode(size=size, group_size=self.list_traffic_env_conf[0]["FAST_BATCH_SIZE"],
                                       dic_agent_conf=self.dic_agent_conf)

        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params: 
            policy.load_params(params)

        old_params = None
        meta_update_period = 1
        meta_update = True

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

            # if update
            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf[
                'UPDATE_PERIOD'] == 0:  # 0, 1

                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:  # 2000
                    episodes.forget()

                old_params = params

                policy.fit(episodes, params=params, target_params=target_params)
                sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)
                policy.load_params(params)

                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:  # 5
                    target_params = params
                    self.target_step = 0

                # meta update
                if meta_update_period % self.dic_agent_conf["META_UPDATE_PERIOD"] == 0:  # 10
                    policy.fit(episodes, params=params, target_params=target_params)
                    sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
                    new_slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                    params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
                    policy.load_params(params)

                meta_update_period += 1

            self.step += 1

        if meta_update: 
            policy.fit(episodes, params=params, target_params=target_params)
            sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))
            new_slice_index = np.random.choice(len(episodes), sample_size, replace=False)
            params = policy.update_meta_params(episodes, slice_index, new_slice_index, _params=old_params)
            policy.load_params(params)

            meta_update_period += 1
        policy.decay_epsilon(batch_id)
        return params[0]

        # self.envs.bulk_log()

    def model_sample_meta_test(self, policy, task, batch_id, params=None):
        """
        Perform meta-testing (only testing within one step)
        """
        print('sampler.model_sample_meta_test')

        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        if params:
            policy.load_params(params)

        if self.step >= self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf[
            'TEST_PERIOD'] == 0:  # --update_start 0 --test_period 1
            print('--------------------step is {}----------------------'.format(self.step))  # ***
            print('--------------------test_step is {}----------------------'.format(self.test_step))  # ***
            self.model_single_test_sample(policy, task, self.test_step, batch_id, params=params)
            self.test_step += 1
        self.step += 1

    def model_single_test_sample(self, policy, task, test_step, batch_id, params):
        policy.load_params(params)

        dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
        dic_traffic_env_conf['TRAFFIC_FILE'] = task

        dic_path = copy.deepcopy(self.dic_path)
        dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], 'test_round',
                                               task, 'tasks_round_' + str(batch_id) + "_" + str(test_step))

        if not os.path.exists(dic_path['PATH_TO_LOG']):
            os.makedirs(dic_path['PATH_TO_LOG'])

        dic_exp_conf = copy.deepcopy(self.dic_exp_conf)

        env = CityFlowEnv(path_to_log=dic_path["PATH_TO_LOG"],
                          path_to_work_directory=dic_path["PATH_TO_DATA"],
                          dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int(  # purpose: update env ==> env.bulk_log()
                dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):  # 3600 / 10
            action_list = []
            for one_state in state:
                action = policy.choose_action([[one_state]],
                                              test=True)  # one for multi-state, the other for multi-intersection
                action_list.append(action[0])  # for multi-state

            next_state, reward, done, _ = env.step(action_list)
            state = next_state
            step_num += 1
            stop_cnt += 1

        ## output vehicle_inter_{0}.csv
        env.bulk_log()
        ## output test_results.csv
        model_write_summary(dic_path, task, self.dic_exp_conf["EPISODE_LEN"],
                            batch_id, test_step, self.dic_traffic_env_conf['FLOW_FILE'])

    def sample_meta_test(self, policy, task, batch_id, params=None, target_params=None, old_episodes=None):
        '''
        metalearner.py: Perform meta-testing (only testing within one episode)
        '''
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        episodes = BatchEpisodes(dic_agent_conf=self.dic_agent_conf, old_episodes=old_episodes)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        if params:  
            policy.load_params(params)

        while (not all(dones)) or (not self.queue.empty()):
            actions = policy.choose_action(observations)
            ## for multi_intersection
            actions = np.reshape(actions, (-1, 1))
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, new_observations, rewards,
                            batch_ids)  # append is a function defined in BatchEpisodes
            observations, batch_ids = new_observations, new_batch_ids

            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf[
                'UPDATE_PERIOD'] == 0:  # 0, 1
                if len(episodes) > self.dic_agent_conf['MAX_MEMORY_LEN']:  # 2000
                    episodes.forget()

                policy.fit(episodes, params=params, target_params=target_params)
                sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(episodes))  # 30
                slice_index = np.random.choice(len(episodes), sample_size, replace=False)
                params = policy.update_params(episodes, params=copy.deepcopy(params),
                                              lr_step=self.lr_step, slice_index=slice_index)

                policy.load_params(params)

                self.lr_step += 1
                self.target_step += 1
                if self.target_step == self.dic_agent_conf['UPDATE_Q_BAR_FREQ']:  # 5
                    target_params = params
                    self.target_step = 0

            if self.step > self.dic_agent_conf['UPDATE_START'] and self.step % self.dic_agent_conf[
                'TEST_PERIOD'] == 0:  # --update_start 0 --test_period 1
                print('--------------------step is {}----------------------'.format(self.step))  # ***
                print('--------------------test_step is {}----------------------'.format(self.test_step))  # ***
                self.single_test_sample(policy, task, self.test_step, params=params)
                pickle.dump(params, open(
                    os.path.join(self.dic_path['PATH_TO_MODEL'], 'params' + "_" + str(self.test_step) + ".pkl"),
                    'wb'))
                # write_summary(self.dic_path, task,  # from utils.py
                #               self.dic_traffic_env_conf["EPISODE_LEN"], batch_id)  # round = 0??
                # print('--------------------write summary----------------------')
                self.test_step += 1
            self.step += 1

        policy.decay_epsilon(batch_id)
        self.envs.bulk_log()
        return params, target_params, episodes

    def single_test_sample(self, policy, task, batch_id, params):  # batch_id = self.test_step
        policy.load_params(params)

        dic_traffic_env_conf = copy.deepcopy(self.dic_traffic_env_conf)
        dic_traffic_env_conf['TRAFFIC_FILE'] = task

        dic_path = copy.deepcopy(self.dic_path)
        dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], 'test_round',
                                               task, 'tasks_round_' + str(batch_id))

        if not os.path.exists(dic_path['PATH_TO_LOG']):
            os.makedirs(dic_path['PATH_TO_LOG'])

        dic_exp_conf = copy.deepcopy(self.dic_exp_conf)

        env = CityFlowEnv(path_to_log=dic_path["PATH_TO_LOG"],
                          path_to_work_directory=dic_path["PATH_TO_DATA"],
                          dic_traffic_env_conf=dic_traffic_env_conf)

        done = False
        state = env.reset()
        step_num = 0
        stop_cnt = 0
        while not done and step_num < int( 
                dic_exp_conf["EPISODE_LEN"] / dic_traffic_env_conf["MIN_ACTION_TIME"]):  # 3600 / 10
            action_list = []
            for one_state in state:
                action = policy.choose_action([[one_state]],
                                              test=True)  # one for multi-state, the other for multi-intersection
                action_list.append(action[0])  # for multi-state

            next_state, reward, done, _ = env.step(action_list)
            state = next_state
            step_num += 1
            stop_cnt += 1
        # output vehicle_inter_{0}.csv
        env.bulk_log()
        # output test_results.csv
        write_summary(dic_path, task, self.dic_exp_conf["EPISODE_LEN"], batch_id,
                      self.dic_traffic_env_conf['FLOW_FILE'])

    def reset_task(self, tasks, batch_id, reset_type='learning'):
        # regenerate new envs to avoid the engine stuck bug!
        dic_traffic_env_conf_list = []
        dic_path_list = []
        for task in tasks:
            dic_agent_conf = copy.deepcopy(self.dic_agent_conf)
            dic_agent_conf['TRAFFIC_FILE'] = task

            dic_traffic_env_conf = copy.deepcopy(self.task_traffic_env_map[task])
            dic_traffic_env_conf['TRAFFIC_FILE'] = task
            dic_traffic_env_conf_list.append(dic_traffic_env_conf)

            dic_path = copy.deepcopy(self.task_path_map[task])
            if reset_type == 'test':
                dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], reset_type + '_round',
                                                       task, 'tasks_round_' + str(batch_id))
            else:  # reset_type == 'meta', 'learning' or 'modeling'
                dic_path["PATH_TO_LOG"] = os.path.join(dic_path['PATH_TO_WORK_DIRECTORY'], reset_type + '_round',
                                                       'tasks_round_' + str(batch_id), task)
            dic_path_list.append(dic_path)

            if not os.path.exists(dic_path['PATH_TO_LOG']):
                os.makedirs(dic_path['PATH_TO_LOG'])

        self.envs = SubprocVecEnv(dic_path_list, dic_traffic_env_conf_list, len(tasks), queue=self.queue)
