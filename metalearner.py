import os
#import utils
import pickle
import os
import gc
import copy
import numpy as np
from utils import write_summary
import random
import shutil
from intersection_model import IntersectionModel
from episode import BatchEpisodes
import psutil
import tensorflow as tf
from keras import backend as K
import gc
import sys


class MetaLearner(object):
    def __init__(self, sampler, policy, dic_agent_conf, dic_traffic_env_conf, dic_path):
        """
            Meta-learner incorporates MAML and MetaLight and can update the meta model by
            different learning methods.
            Arguments:
                sampler:    sample trajectories and update model parameters 
                policy:     frapplus_agent or metalight_agent
                ...
        """
        self.sampler = sampler  # BatchSampler(dic_exp_conf=dic_exp_conf,
                           # dic_agent_conf=dic_agent_conf,
                           # dic_traffic_env_conf=dic_traffic_env_conf,
                           # dic_path=dic_path,
                           # batch_size=args.fast_batch_size,
                           # num_workers=args.num_workers)
        self.policy = policy  # frapplus_agent or metalight_agent
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.meta_params = self.policy.save_params()  # params of Q network
        self.meta_target_params = self.meta_params  # params of target Q network
        self.adapted_params = self.meta_params  # ***
        self.adapted_target_params = self.meta_params  # ***
        self.step_cnt = 0
        self.period = self.dic_agent_conf['PERIOD']

    # def model_based_sample_traj(self, task, batch_id, episodes):
    #     """
    #         Use MAML framework to samples trajectories before and after the update of the parameters
    #         for all the tasks. Then, update meta-parameters.
    #     """
    #
    #     tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']  # 1
    #     '''
    #     Generate transitions into D (episodes.total_samples) using fθi'
    #     '''
    #     # todo: reset_type='modeling'
    #     self.sampler.reset_task(tasks, batch_id,
    #                             reset_type='modeling')  # regenerate new envs to avoid the engine stuck bug!
    #     env_episodes, dic_action_phase = self.sampler.model_based_sample_traj(self.policy, tasks, batch_id,
    #                                          params=self.adapted_params, episodes=episodes)  # return episodes
    #     return env_episodes, dic_action_phase

    def model_based_sample_maml(self, task, model_id, batch_id, img_train=False, pre_episodes=None):  # ***
        """
            Use MAML framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']  # 1
        if not os.path.exists(os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'dic_act_pha')):
            os.makedirs(os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'dic_act_pha'))
        if not os.path.exists(os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model')):
            os.makedirs(os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model'))

        if not img_train:
            '''
            Generate transitions into D (episodes.total_samples) using fθi'
            '''
            # todo: reset_type='modeling'
            self.sampler.reset_task(tasks, batch_id,
                                    reset_type='modeling')  # regenerate new envs to avoid the engine stuck bug!
            model_learning_episodes, dic_action_phase = self.sampler.model_based_sample_traj(self.policy, tasks, batch_id,
                                                                                  params=self.adapted_params)  # return episodes
            if not os.path.exists('{0}/dic_act_pha/dic_action_phase_{1}.npy'.format(self.dic_path["PATH_TO_ENV_MODEL"], task)):
                np.save('{0}/dic_act_pha/dic_action_phase_{1}.npy'.format(self.dic_path["PATH_TO_ENV_MODEL"], task), dic_action_phase)

            '''
            Randomly sample subset Di from D
            '''
            model_training_sample_size = min(self.dic_agent_conf['MODEL_TRAINING_SAMPLE_SIZE'], len(model_learning_episodes))  # len(episodes): episodes = BatchEpisodes()
                                                       # __len__(self): len(self.total_samples)
            model_slice_index = np.random.choice(len(model_learning_episodes), model_training_sample_size, replace=False)

            '''
            Train intersection model with Di
            '''
            inter_model = IntersectionModel(model_name='LSTM', policy=self.policy)  # policy: fθ
            if os.path.exists(os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model', 'env_model_{0}.h5'.format(task))):
                inter_model.load_params(path_to_env_model=os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model'), task_for_params=task)
            inter_model.train(list(model_learning_episodes.total_samples), model_slice_index, dic_action_phase, self.dic_agent_conf["EPOCHS"])
            # inter_model.save_params(batch_id, model_id)
            inter_model.save_params(path_to_env_model=os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model'), task_for_params=task)
            inter_model.accu_eval(list(model_learning_episodes.total_samples), model_slice_index, dic_action_phase)
            print('============================inter_model train complete============================')
            self.get_memory_info()
        else:
            inter_model = IntersectionModel(model_name='LSTM', policy=self.policy)  # policy: fθ
            if os.path.exists(os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model', 'env_model_{0}.h5'.format(task))):
                inter_model.load_params(path_to_env_model=os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model'), task_for_params=task)
            model_learning_episodes = BatchEpisodes(dic_agent_conf=self.dic_agent_conf)
            dic_action_phase = np.load('{0}/dic_act_pha/dic_action_phase_{1}.npy'.format(self.dic_path["PATH_TO_ENV_MODEL"], task), allow_pickle=True).item()

        # ===================================================================
        '''
        Generate imaginary trajectories to model_learning_episodes using fθ
        Sample imaginary trajectories from model as Ei using fθ
        '''
        # new episode storing imaginary transitions
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')
        inter_model.sample_e(start_episode=pre_episodes['model_learning_episodes'] if img_train else model_learning_episodes,
                             episodes=model_learning_episodes, params=self.meta_params, dic_action_phase=dic_action_phase,
                             num_img_trans=self.dic_agent_conf["NUM_IMG_TRANS"] if img_train else 0)  # sample Ei
        # print('---img_sample: {}---'.format(model_learning_episodes.total_samples[:100]))
        print('============================sample Ei complete============================')
        print('-----len(model_learning_episodes) is: {}-----'.format(len(model_learning_episodes)))
        self.policy.fit(model_learning_episodes, params=self.meta_params, target_params=self.meta_target_params)  # FRAPPlusAgent(Agent)

        '''
        Update θi'
        '''
        sample_size = min(self.dic_agent_conf['MODEL_SAMPLE_SIZE'], len(model_learning_episodes))
        slice_index = np.random.choice(len(model_learning_episodes), sample_size, replace=False)
        params = self.policy.update_params(model_learning_episodes, params=copy.deepcopy(self.meta_params),
                                           lr_step=0, slice_index=slice_index)
        pickle.dump(params, (open(os.path.join(self.sampler.dic_path['PATH_TO_MODEL'], 'params_{0}_{1}.pkl'
                                               .format(batch_id, model_id)), 'wb')))  # model_id == task_id
        print('============================update θi\' complete============================')
        self.get_memory_info()

        # ===================================================================
        '''
        Sample real and imaginary transitions from model as Ei' using fθi'
        '''
        self.sampler.reset_task(tasks, batch_id, reset_type='meta')
        if not img_train:
            model_meta_episodes = self.sampler.sample_maml(self.policy, tasks, batch_id, params=params)
            # model_meta_sample_size = min(self.dic_agent_conf['MODEL_TRAINING_SAMPLE_SIZE'], len(model_meta_episodes))
            # model_slice_index = np.random.choice(len(model_meta_episodes), model_meta_sample_size, replace=False)
            # inter_model.load_params(path_to_env_model=os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model'),
            #                         task_for_params=task)
            # inter_model.train(list(model_meta_episodes.total_samples), model_slice_index, dic_action_phase,
            #                   self.dic_agent_conf["EPOCHS"])
            # inter_model.save_params(path_to_env_model=os.path.join(self.dic_path["PATH_TO_ENV_MODEL"], 'env_model'),
            #                         task_for_params=task)
        else:
            model_meta_episodes = BatchEpisodes(dic_agent_conf=self.sampler.dic_agent_conf)
        inter_model.sample_e(start_episode=pre_episodes['model_meta_episodes'] if img_train else model_meta_episodes,
                             episodes=model_meta_episodes, params=params, dic_action_phase=dic_action_phase,
                             num_img_trans=self.dic_agent_conf["NUM_IMG_TRANS"] if img_train else 0)  # sample Ei'
        print('============================sample Ei\' complete============================')
        print('-----len(model_meta_episodes) is: {}-----'.format(len(model_meta_episodes)))
        self.policy.fit(model_meta_episodes, params=params, target_params=self.meta_target_params)

        '''
        Calculate gradients for the update of θ0 after all tasks 
        '''
        sample_size = min(self.dic_agent_conf['MODEL_SAMPLE_SIZE'], len(model_meta_episodes))
        slice_index = np.random.choice(len(model_meta_episodes), sample_size, replace=False)
        _grads = self.policy.cal_grads(model_learning_episodes, model_meta_episodes, slice_index=slice_index,
                                       params=self.meta_params)

        if self.dic_agent_conf['GRADIENT_CLIP']:
            for key in _grads.keys():
                _grads[key] = np.clip(_grads[key], -1 * self.dic_agent_conf['CLIP_SIZE'],
                                      self.dic_agent_conf['CLIP_SIZE'])
        with open(os.path.join(self.dic_path['PATH_TO_GRADIENT'], "gradients_%d.pkl")%batch_id,"ab+") as f:
            pickle.dump(_grads, f, -1)

        self.meta_params = params
        self.step_cnt += 1
        if self.step_cnt == self.period:
            self.step_cnt = 0
            self.meta_target_params = self.meta_params
        # pickle.dump(self.meta_params, open(
        #     os.path.join(self.sampler.dic_path['PATH_TO_MODEL'],
        #                  'params' + "_" + str(batch_id) + ".pkl"), 'wb'))
        print('============================calculate gradients complete============================')
        return model_learning_episodes, model_meta_episodes

    @staticmethod
    def get_memory_info(total=False):
        mem = psutil.virtual_memory()
        if total:
            zj = float(mem.total) / 1024 / 1024 / 1024
            print('系统总计内存:%.3fGB' % zj)
        ysy = float(mem.used) / 1024 / 1024 / 1024
        print('系统已经使用内存:%.3fGB' % ysy)
        kx = float(mem.free) / 1024 / 1024 / 1024
        print('系统空闲内存:%.3fGB' % kx)

    def sample_maml(self, task, batch_id):
        """
            Use MAML framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']  # 1

        '''
        Generate transitions into D
        '''
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')  # regenerate new envs to avoid the engine stuck bug!
        learning_episodes = self.sampler.sample_maml(self.policy, tasks, batch_id, params=self.meta_params)  # return episodes
        self.policy.fit(learning_episodes, params=self.meta_params, target_params=self.meta_target_params)  # FRAPPlusAgent(Agent)
        sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(learning_episodes))  # len(episodes): episodes = BatchEpisodes()
                                                                              # __len__(self): len(self.total_samples)
        '''
        sample transitions as Di and Update θi
        '''
        slice_index = np.random.choice(len(learning_episodes), sample_size, replace=False)
        params = self.policy.update_params(learning_episodes, params=copy.deepcopy(self.meta_params),
                                           lr_step=0, slice_index=slice_index)

        '''
        Sample new transitions from D as Di'
        '''
        self.sampler.reset_task(tasks, batch_id, reset_type='meta')
        meta_episodes = self.sampler.sample_maml(self.policy, tasks, batch_id, params=params)
        self.policy.fit(meta_episodes, params=params, target_params=self.meta_target_params)

        '''
        Update θ0
        '''
        sample_size = min(self.dic_agent_conf['SAMPLE_SIZE'], len(learning_episodes))
        slice_index = np.random.choice(len(learning_episodes), sample_size, replace=False)
        _grads = self.policy.cal_grads(learning_episodes, meta_episodes, slice_index=slice_index,
                                       params=self.meta_params)

        if self.dic_agent_conf['GRADIENT_CLIP']:
            for key in _grads.keys():
                _grads[key] = np.clip(_grads[key], -1 * self.dic_agent_conf['CLIP_SIZE'],
                                      self.dic_agent_conf['CLIP_SIZE'])
        with open(os.path.join(self.dic_path['PATH_TO_GRADIENT'], "gradients_%d.pkl")%batch_id,"ab+") as f:
            pickle.dump(_grads, f, -1)

        self.meta_params = params
        self.step_cnt += 1
        if self.step_cnt == self.period:
            self.step_cnt = 0
            self.meta_target_params = self.meta_params
        pickle.dump(self.meta_params, open(
            os.path.join(self.sampler.dic_path['PATH_TO_MODEL'],
                         'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

    def sample_mbmrl(self, _tasks, batch_id):
        """
            Use MetaLight framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = []
        for task in _tasks:
            tasks.extend([task] * self.dic_traffic_env_conf[0]['FAST_BATCH_SIZE'])  # 1; dic_traffic_env_conf is a list here
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')
        meta_params = self.sampler.sample_mbmrl(self.policy, tasks, batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params)  # peform meta training
        pickle.dump(meta_params, open(
           os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

        # shutil.copy(os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"),
        #             os.path.join('model', 'initial', 'common', dic_agent_conf['PRE_TRAIN_MODEL_NAME'] + '.pkl'))
        print('-------------------------dump params_%d complete-------------------------'%self.batch_id)  # ***

    def sample_metalight(self, _tasks, batch_id):
        """
            Use MetaLight framework to samples trajectories before and after the update of the parameters
            for all the tasks. Then, update meta-parameters.
        """
        self.batch_id = batch_id
        tasks = []
        for task in _tasks:
            tasks.extend([task] * self.dic_traffic_env_conf[0]['FAST_BATCH_SIZE'])  # 1; dic_traffic_env_conf is a list here
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')
        meta_params = self.sampler.sample_metalight(self.policy, tasks, batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params)  # peform meta training
        pickle.dump(meta_params, open(
           os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))

        # shutil.copy(os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"),
        #             os.path.join('model', 'initial', 'common', dic_agent_conf['PRE_TRAIN_MODEL_NAME'] + '.pkl'))
        print('-------------------------dump params_%d complete-------------------------'%self.batch_id)  # ***

    def sample_meta_test(self, task, batch_id, old_episodes=None):
        """
            Perform meta-testing (only testing within one episode) or offline-training (in multiple episodes to let models well trained and obtrained pre-trained models).
            Arguments:
                old_episodes: episodes generated and kept in former batches, controlled by 'MULTI_EPISODES'
                ...
        """
        self.batch_id = batch_id
        tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        self.meta_params, self.meta_target_params, episodes = \
            self.sampler.sample_meta_test(self.policy, tasks[0], batch_id, params=self.meta_params,
                                       target_params=self.meta_target_params, old_episodes=old_episodes)
        pickle.dump(self.meta_params, open(
            os.path.join(self.sampler.dic_path['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))
        return episodes

    def model_sample_meta_test(self, task, batch_id, old_episodes=None):
        """
            Perform meta-testing (only testing within one episode) or offline-training (in multiple episodes to let models well trained and obtrained pre-trained models).
            Arguments:
                old_episodes: episodes generated and kept in former batches, controlled by 'MULTI_EPISODES'
                ...
        """
        print('metalearner.model_sample_meta_test')

        self.batch_id = batch_id
        tasks = [task] * self.dic_traffic_env_conf['FAST_BATCH_SIZE']
        self.sampler.reset_task(tasks, batch_id, reset_type='learning')

        self.sampler.model_sample_meta_test(self.policy, tasks[0], batch_id, params=self.meta_params)
