from metalearner import MetaLearner
from sampler import BatchSampler
from multiprocessing import Process
import config
import time
import copy
import random
import numpy as np
import tensorflow as tf
import pickle
import shutil
from traffic import *
from utils import parse, config_all, parse_roadnet
import sys
from episode import BatchEpisodes
import psutil


def main(args):
    '''
        Perform meta-training for MAML and MetaLight

        Arguments:
            args: generated in utils.py:parse()
    '''

    # configuration: experiment, agent, traffic_env, path
    dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path = config_all(args)

    _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    # _time = '00_00_00_00_00'
    postfix = args.postfix 
    inner_memo = "_" + _time + postfix
    dic_traffic_env_conf["inner_memo"] = inner_memo
    dic_path.update({
        "PATH_TO_MODEL": os.path.join(dic_path["PATH_TO_MODEL"], inner_memo),
        # os.path.join("model", args.memo, inner_memo)
        "PATH_TO_WORK_DIRECTORY": os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], inner_memo),
        "PATH_TO_GRADIENT": os.path.join(dic_path["PATH_TO_GRADIENT"], inner_memo, "gradient"),
        "PATH_TO_ENV_MODEL": os.path.join(dic_path["PATH_TO_ENV_MODEL"], inner_memo),  # ***
    })

    # traffic_env, traffic_category defined in traffic 
    dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = traffic_category["train_all"]
    dic_traffic_env_conf["traffic_category"] = traffic_category

    random.seed(dic_agent_conf['SEED'])
    np.random.seed(dic_agent_conf['SEED'])
    tf.set_random_seed(dic_agent_conf['SEED'])
    print('------dic_agent_conf_seed is {}-----------'.format(dic_agent_conf['SEED']))

    # load or build initial model
    print('PRE_TRAIN is: ', dic_agent_conf['PRE_TRAIN'])  # ***
    if not dic_agent_conf['PRE_TRAIN']:  # action="store_true"
        p = Process(target=build_init, args=(copy.deepcopy(dic_agent_conf),
                                             copy.deepcopy(dic_traffic_env_conf),
                                             copy.deepcopy(dic_path)))
        p.start()
        p.join()
        print('------------------------build init complete------------------------')
    else:  # load initial model
        if not os.path.exists(dic_path['PATH_TO_MODEL']):
            os.makedirs(dic_path['PATH_TO_MODEL'])
        shutil.copy(os.path.join('model', 'initial', 'common', dic_agent_conf['PRE_TRAIN_MODEL_NAME'] + '.pkl'),
                    # model/initial/common/args.pre_train_model_name.pkl
                    os.path.join(dic_path['PATH_TO_MODEL'],
                                 'params' + "_" + "init.pkl"))  # copy file1 to file2/directory
        # model/args.memo/_%m_%d_%H_%M_%S/params_init.pkl

    get_memory_info(total=True)
    print(
        '---------------len_train_traffic is {}----------------'.format(len(dic_traffic_env_conf['TRAFFIC_IN_TASKS'])))
    for batch_id in range(args.run_round):  # 100 round
        task_num = min(len(dic_traffic_env_conf['TRAFFIC_IN_TASKS']), args.meta_batch_size)
        sample_task_traffic = random.sample(dic_traffic_env_conf['TRAFFIC_IN_TASKS'],
                                            task_num)  # traffic_category["train_all"]
        output_task = dic_traffic_env_conf['TRAFFIC_IN_TASKS'][10]  # ***
        training_tasks = sample_task_traffic  # ***
        print('--------sample_task_traffic is {}------------'.format(sample_task_traffic))
        # training_tasks = dic_traffic_env_conf['TRAFFIC_IN_TASKS'][:dic_agent_conf['NUM_TASKS']]  # ***
        # training_tasks = [output_task] * dic_agent_conf['NUM_TASKS']  # ***

        if dic_exp_conf["TRAINING_OUTPUT_MODE"]:  # output return using pre-trained params
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>batch_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(batch_id))
            print('==================================training output start==================================')
            if dic_traffic_env_conf["MODEL_NAME"] == "MetaLight":
                print('model name is metalight')
                dic_path.update({"PATH_TO_MODEL": os.path.join("model", args.memo, '_12_15_09_46_40' + postfix)})

            elif dic_traffic_env_conf["MODEL_NAME"] == "FRAPPlus":
                dic_path.update({"PATH_TO_MODEL": os.path.join("model", args.memo, '_12_19_04_51_17' + postfix)})

            else:  # "MBMRL"
                dic_path.update({"PATH_TO_MODEL": os.path.join("model", args.memo, '_01_27_03_22_19' + postfix)})

            p = Process(target=output_return,
                        args=(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              output_task, batch_id))
            p.start()
            p.join()
            print('==================================training output complete==================================')
            get_memory_info()

        elif dic_traffic_env_conf["MODEL_NAME"] == "MetaLight":  # args.algorithm
            p = Process(target=metalight_train,
                        args=(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              sample_task_traffic, batch_id)
                        )
            p.start()
            p.join()

            if dic_exp_conf["TRAINING_OUTPUT_RETURN"]:
                output_return(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              output_task, batch_id)
                print('==================================training output complete==================================')
                get_memory_info()

        elif dic_traffic_env_conf["MODEL_NAME"] == "MBMRL":  # *** args.algorithm
            p = Process(target=mbmrl_train,
                        args=(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              sample_task_traffic, batch_id)
                        )
            p.start()
            p.join()

            if dic_exp_conf["TRAINING_OUTPUT_RETURN"]:
                output_return(copy.deepcopy(dic_exp_conf),
                              copy.deepcopy(dic_agent_conf),
                              copy.deepcopy(dic_traffic_env_conf),
                              copy.deepcopy(dic_path),
                              output_task, batch_id)
                print('==================================training output complete==================================')
                get_memory_info()

        elif dic_traffic_env_conf["MODEL_NAME"] == "FRAPPlus":  # maml  --algorithm
            p = Process(target=maml_process,
                        args=(batch_id, sample_task_traffic, dic_agent_conf,
                              dic_exp_conf, dic_traffic_env_conf, dic_path, output_task))
            p.start()
            p.join()
            get_memory_info()

        else:  # *** "MBMRL" --ModelLight
            p = Process(target=mbmrl_process,
                        args=(batch_id, training_tasks, dic_agent_conf,
                              dic_exp_conf, dic_traffic_env_conf, dic_path, output_task))
            p.start()
            p.join()
            get_memory_info()

        ## update the epsilon
        decayed_epsilon = dic_agent_conf["EPSILON"] * pow(dic_agent_conf["EPSILON_DECAY"], batch_id)
        dic_agent_conf["EPSILON"] = max(decayed_epsilon, dic_agent_conf["MIN_EPSILON"])

    # if dic_traffic_env_conf["MODEL_NAME"] == "MBMRL":  # ***
    #     for extra_batch_id in range(args.extra_run_round):
    #         task_num = min(len(dic_traffic_env_conf['TRAFFIC_IN_TASKS']), args.meta_batch_size)
    #         sample_task_traffic = random.sample(dic_traffic_env_conf['TRAFFIC_IN_TASKS'],
    #                                             task_num)  # traffic_category["train_all"]
    #         training_tasks = sample_task_traffic  # ***

    if not os.path.exists(os.path.join('model', 'initial', 'common')):
        os.makedirs(os.path.join('model', 'initial', 'common'))
    shutil.copy(os.path.join(dic_path['PATH_TO_MODEL'], 'params' + "_" + str(args.run_round - 1) + ".pkl"),
                os.path.join('model', 'initial', 'common',
                             dic_agent_conf['PRE_TRAIN_MODEL_NAME'] + str(args.common_model_No) + '.pkl'))  # ***
    # model/initial/common/args.pre_train_model_name.pkl


def maml_process(batch_id, sample_task_traffic, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path,
                 output_task):
    for task in sample_task_traffic:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>batch_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(batch_id))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>task is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(task))
        maml_train(copy.deepcopy(dic_exp_conf),
                   copy.deepcopy(dic_agent_conf),
                   copy.deepcopy(dic_traffic_env_conf),
                   copy.deepcopy(dic_path),
                   task, batch_id)

    if not dic_traffic_env_conf['FIRST_PART']:  # not False
        meta_step(dic_path, dic_agent_conf, dic_traffic_env_conf, batch_id)

    if dic_exp_conf["TRAINING_OUTPUT_RETURN"]:
        output_return(copy.deepcopy(dic_exp_conf),
                      copy.deepcopy(dic_agent_conf),
                      copy.deepcopy(dic_traffic_env_conf),
                      copy.deepcopy(dic_path),
                      output_task, batch_id)
        print('==================================training output complete==================================')
        get_memory_info()
    sys.exit()


def mbmrl_process(batch_id, training_tasks, dic_agent_conf, dic_exp_conf, dic_traffic_env_conf, dic_path, output_task):
    episodes = []
    for task, model_id in zip(training_tasks, range(len(training_tasks))):  # model_id == task_id
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>batch_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(batch_id))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>model_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(model_id))
        print('-------------------MBMRL task is: {}--------------------'.format(task))
        model_learning_episodes, model_meta_episodes = mbmrl_train(copy.deepcopy(dic_exp_conf),
                                                                   copy.deepcopy(dic_agent_conf),
                                                                   copy.deepcopy(dic_traffic_env_conf),
                                                                   copy.deepcopy(dic_path), task, model_id, batch_id,
                                                                   img_train=False, pre_episodes=None)  # ***
        episodes.append(
            {'model_learning_episodes': model_learning_episodes, 'model_meta_episodes': model_meta_episodes})
    print('==================================mbmrl train complete==================================', '\n')

    if not dic_traffic_env_conf['FIRST_PART']:  # not False
        meta_step(dic_path, dic_agent_conf, dic_traffic_env_conf, batch_id, pre_train=True)
        print('==================================update θ0 complete==================================')
        get_memory_info()

    img_tasks = []
    for env_model in os.listdir(os.path.join(dic_path["PATH_TO_ENV_MODEL"], 'env_model')):
        img_task = env_model[10: env_model.find('.h5')]
        img_tasks.append(img_task)
    task_num = min(len(img_tasks), args.meta_batch_size)
    training_tasks = random.sample(img_tasks, task_num)  # traffic_category["train_all"]
    print('--------sample_img_task is {}------------'.format(training_tasks))

    for task, model_id in zip(training_tasks, range(len(training_tasks))):  # model_id == task_id
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>batch_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(batch_id))
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>img_model_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(model_id))
        print('-------------------imagine task is: {}--------------------'.format(task))
        mbmrl_train(copy.deepcopy(dic_exp_conf),
                    copy.deepcopy(dic_agent_conf),
                    copy.deepcopy(dic_traffic_env_conf),
                    copy.deepcopy(dic_path),
                    task, model_id, batch_id, img_train=True, pre_episodes=episodes[model_id])  # ***
    print('==================================mbmrl train complete==================================', '\n')

    if not dic_traffic_env_conf['FIRST_PART']:  # not False
        meta_step(dic_path, dic_agent_conf, dic_traffic_env_conf, batch_id)
        print('==================================update θ0 complete==================================')
        get_memory_info()

    if dic_exp_conf["TRAINING_OUTPUT_RETURN"]:
        output_return(copy.deepcopy(dic_exp_conf),
                      copy.deepcopy(dic_agent_conf),
                      copy.deepcopy(dic_traffic_env_conf),
                      copy.deepcopy(dic_path),
                      output_task, batch_id)
        print('==================================training output complete==================================')
        get_memory_info()

    sys.exit()  # *** release the memory during running codes?


def get_memory_info(total=False):
    mem = psutil.virtual_memory()
    if total:
        zj = float(mem.total) / 1024 / 1024 / 1024
        print('系统总计内存:%.3fGB' % zj)
    ysy = float(mem.used) / 1024 / 1024 / 1024
    print('系统已经使用内存:%.3fGB' % ysy)
    kx = float(mem.free) / 1024 / 1024 / 1024
    print('系统空闲内存:%.3fGB' % kx)


def build_init(dic_agent_conf, dic_traffic_env_conf, dic_path):
    '''
        build initial model for maml and metalight

        Arguments:
            dic_agent_conf:         configuration of agent
            dic_traffic_env_conf:   configuration of traffic environment
            dic_path:               path of source files and output files
    '''

    any_task = dic_traffic_env_conf["traffic_category"]["train_all"][0]
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][any_task][2]
    dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][any_task][3]
    # parse roadnet
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'], any_task.split(".")[0],
                                dic_traffic_env_conf["traffic_category"]["traffic_info"][any_task][
                                    2])  # dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )
    params = policy.init_params()
    if not os.path.exists(dic_path["PATH_TO_MODEL"]):
        os.makedirs(dic_path["PATH_TO_MODEL"])
    pickle.dump(params, open(os.path.join(dic_path['PATH_TO_MODEL'], 'params' + "_" + "init.pkl"), 'wb'))


def output_return(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, output_task, batch_id):
    print('-------------------output task is: {}--------------------'.format(output_task))
    traffic_file = output_task
    traffic_of_tasks = [traffic_file]

    dic_traffic_env_conf['ROADNET_FILE'] = dic_traffic_env_conf["TRAFFIC_CATEGORY"]["traffic_info"][traffic_file][2]
    dic_traffic_env_conf['FLOW_FILE'] = dic_traffic_env_conf["TRAFFIC_CATEGORY"]["traffic_info"][traffic_file][3]

    # # path
    # _time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    # postfix = ""
    dic_path.update({
        # "PATH_TO_MODEL": os.path.join(dic_path["PATH_TO_MODEL"], traffic_file + "_" + _time + postfix),
        # # os.path.join("model", args.memo, hangzhou_baochu_tiyuchang_1h_10_11_2021.json_12_13_13_20_09)
        # "PATH_TO_WORK_DIRECTORY": os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"],
        #                                        traffic_file + "_" + _time + postfix),
        # "PATH_TO_GRADIENT": os.path.join(dic_path["PATH_TO_GRADIENT"], traffic_file + "_" + _time + postfix,
        #                                  "gradient"),
        "PATH_TO_DATA": os.path.join(dic_path["PATH_TO_DATA"], traffic_file.split(".")[0])
    })
    # traffic env
    dic_traffic_env_conf["TRAFFIC_FILE"] = traffic_file
    dic_traffic_env_conf["TRAFFIC_IN_TASKS"] = [traffic_file]
    # parse roadnet
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'], dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    dic_exp_conf.update({
        "TRAFFIC_FILE": traffic_file,
        "TRAFFIC_IN_TASKS": traffic_of_tasks})

    print('before _train')
    _train(copy.deepcopy(dic_exp_conf),
           copy.deepcopy(dic_agent_conf),
           copy.deepcopy(dic_traffic_env_conf),
           copy.deepcopy(dic_path), batch_id)


def _train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, batch_id):
    """
    Perform meta-testing for MAML, Metalight, Random, and Pretrained

    Arguments:
        dic_exp_conf:           dict,   configuration of this experiment
        dic_agent_conf:         dict,   configuration of agent
        dic_traffic_env_conf:   dict,   configuration of traffic environment
        dic_path:               dict,   path of source files and output files
    """
    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=dic_traffic_env_conf,
                           dic_path=dic_path,
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS["FRAPPlus"](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              dic_path=dic_path
                              )

    params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'],  # load the initialization parameters
                                           'params' + "_" + str(batch_id) + ".pkl"), 'rb'))
    metalearner.meta_params = params
    metalearner.meta_target_params = params

    tasks = [dic_exp_conf['TRAFFIC_FILE']]
    metalearner.model_sample_meta_test(tasks[0], batch_id)


def sample_real_traj(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, task, task_id, batch_id,
                     episodes):  # ***
    '''
        maml meta-train function

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            dic_traffic_env_conf:   dict,   configuration of traffic environment
            dic_path:               dict,   path of source files and output files
            task:                   string, traffic files name
            task_id:
            batch_id:               int,    round number
            episodes:
    '''
    dic_path.update({
        "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
    })
    # parse roadnet
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
    dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                dic_traffic_env_conf["traffic_category"]["traffic_info"][task][
                                    2])  # dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    dic_traffic_env_conf["TRAFFIC_FILE"] = task

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=dic_traffic_env_conf,
                           dic_path=dic_path,
                           batch_size=args.fast_batch_size,  # 1
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              dic_path=dic_path
                              )

    # set the adapted params
    if batch_id == 0:
        metalearner.adapted_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))

    else:
        metalearner.adapted_params = pickle.load(  # load the adapted params from last round
            open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_{0}_{1}.pkl'.format((batch_id - 1), task_id)),
                 'rb'))
        # metalearner.adapted_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))

    return metalearner.model_based_sample_traj(task, batch_id, episodes)
    # sys.exit() 


def mbmrl_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf,
                dic_path, task, model_id, batch_id, img_train=False, pre_episodes=None):  # ***
    '''
        maml meta-train function

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            dic_traffic_env_conf:   dict,   configuration of traffic environment
            dic_path:               dict,   path of source files and output files
            task:                   string, traffic files name
            model_id:
            batch_id:               int,    round number
            episodes:
            dic_action_phase:
            img_train:
            pre_episodes:
    '''
    dic_path.update({
        "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
    })
    # parse roadnet
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
    dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                dic_traffic_env_conf["traffic_category"]["traffic_info"][task][
                                    2])  # dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    dic_traffic_env_conf["TRAFFIC_FILE"] = task

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=dic_traffic_env_conf,
                           dic_path=dic_path,
                           batch_size=args.fast_batch_size,  # 1
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              dic_path=dic_path
                              )

    # set the meta_params
    if batch_id == 0:
        if img_train:
            params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'pre_params_%d.pkl' % batch_id), 'rb'))
        else:
            params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        metalearner.meta_params = params
        metalearner.meta_target_params = params
    else:
        if img_train:
            params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'pre_params_%d.pkl' % batch_id), 'rb'))
            metalearner.meta_params = params
            period = dic_agent_conf['PERIOD']  # 5
            target_id = int((batch_id - 1) / period)
            metalearner.meta_target_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'],
                                                                           'pre_params_%d.pkl' % (target_id * period)),
                                                              'rb'))
        else:
            params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)), 'rb'))
            metalearner.meta_params = params
            period = dic_agent_conf['PERIOD']  # 5
            target_id = int((batch_id - 1) / period)
            metalearner.meta_target_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'],
                                                                           'params_%d.pkl' % (target_id * period)),
                                                              'rb'))

    model_learning_episodes, model_meta_episodes = metalearner.model_based_sample_maml(task, model_id, batch_id,
                                                                                       img_train, pre_episodes)
    return model_learning_episodes, model_meta_episodes
    # sys.exit()


def maml_train(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path, task, batch_id):  # ***
    '''
        maml meta-train function 

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            dic_traffic_env_conf:   dict,   configuration of traffic environment
            dic_path:               dict,   path of source files and output files
            task:                   string, traffic files name 
            batch_id:               int,    round number
    '''
    dic_path.update({
        "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
    })
    # parse roadnet
    dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
    dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
    roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                dic_traffic_env_conf["traffic_category"]["traffic_info"][task][
                                    2])  # dic_traffic_env_conf['ROADNET_FILE'])
    lane_phase_info = parse_roadnet(roadnet_path)
    dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
    dic_traffic_env_conf["num_lanes"] = int(
        len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
    dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

    dic_traffic_env_conf["TRAFFIC_FILE"] = task

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=dic_traffic_env_conf,
                           dic_path=dic_path,
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=dic_traffic_env_conf,
        dic_path=dic_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=dic_traffic_env_conf,
                              dic_path=dic_path
                              )

    # θi <-- θ0
    if batch_id == 0:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        metalearner.meta_params = params
        metalearner.meta_target_params = params

    else:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)), 'rb'))
        metalearner.meta_params = params
        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1) / period)
        metalearner.meta_target_params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (target_id * period)), 'rb'))

    metalearner.sample_maml(task, batch_id)
    # sys.exit()


def mbmrl_train(dic_exp_conf, dic_agent_conf, _dic_traffic_env_conf, _dic_path, tasks, batch_id):
    '''
        metalight meta-train function

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            _dic_traffic_env_conf:  dict,   configuration of traffic environment
            _dic_path:              dict,   path of source files and output files
            tasks:                  list,   traffic files name in this round
            batch_id:               int,    round number
    '''
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>batch_id is: {}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(batch_id))
    tot_path = []
    tot_traffic_env = []
    for task in tasks:  # to get tot_path and tot_traffic_env
        dic_traffic_env_conf = copy.deepcopy(_dic_traffic_env_conf)
        dic_path = copy.deepcopy(_dic_path)
        dic_path.update({
            "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
        })
        # parse roadnet
        dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
        dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
        roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                    dic_traffic_env_conf["traffic_category"]["traffic_info"][task][
                                        2])  # dic_traffic_env_conf['ROADNET_FILE'])
        lane_phase_info = parse_roadnet(roadnet_path)
        dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
        dic_traffic_env_conf["num_lanes"] = int(
            len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
        dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

        dic_traffic_env_conf["TRAFFIC_FILE"] = task

        tot_path.append(dic_path)
        tot_traffic_env.append(dic_traffic_env_conf)

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=tot_traffic_env,
                           dic_path=tot_path,
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=tot_traffic_env,
        dic_path=tot_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=tot_traffic_env,
                              dic_path=tot_path
                              )

    if batch_id == 0:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        print(params['final_conv_w1'])
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        metalearner.meta_target_params = params

    else:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)),
                                  'rb'))  # load the prameters of previous batch
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1) / period)
        meta_params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (target_id * period)), 'rb'))
        meta_params = [meta_params] * len(policy.policy_inter)
        metalearner.meta_target_params = meta_params

    metalearner.sample_mbmrl(tasks, batch_id)  # peform meta training
    # pickle.dump(meta_params, open(
    # os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))


def metalight_train(dic_exp_conf, dic_agent_conf, _dic_traffic_env_conf, _dic_path, tasks, batch_id):
    '''
        metalight meta-train function 

        Arguments:
            dic_exp_conf:           dict,   configuration of this experiment
            dic_agent_conf:         dict,   configuration of agent
            _dic_traffic_env_conf:  dict,   configuration of traffic environment
            _dic_path:              dict,   path of source files and output files
            tasks:                  list,   traffic files name in this round 
            batch_id:               int,    round number
    '''
    tot_path = []
    tot_traffic_env = []
    for task in tasks:  # to get tot_path and tot_traffic_env
        dic_traffic_env_conf = copy.deepcopy(_dic_traffic_env_conf)
        dic_path = copy.deepcopy(_dic_path)
        dic_path.update({
            "PATH_TO_DATA": os.path.join(dic_path['PATH_TO_DATA'], task.split(".")[0])
        })
        # parse roadnet
        dic_traffic_env_conf["ROADNET_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][2]
        dic_traffic_env_conf["FLOW_FILE"] = dic_traffic_env_conf["traffic_category"]["traffic_info"][task][3]
        roadnet_path = os.path.join(dic_path['PATH_TO_DATA'],
                                    dic_traffic_env_conf["traffic_category"]["traffic_info"][task][
                                        2])  # dic_traffic_env_conf['ROADNET_FILE'])
        lane_phase_info = parse_roadnet(roadnet_path)
        dic_traffic_env_conf["LANE_PHASE_INFO"] = lane_phase_info["intersection_1_1"]
        dic_traffic_env_conf["num_lanes"] = int(
            len(lane_phase_info["intersection_1_1"]["start_lane"]) / 4)  # num_lanes per direction
        dic_traffic_env_conf["num_phases"] = len(lane_phase_info["intersection_1_1"]["phase"])

        dic_traffic_env_conf["TRAFFIC_FILE"] = task

        tot_path.append(dic_path)
        tot_traffic_env.append(dic_traffic_env_conf)

    sampler = BatchSampler(dic_exp_conf=dic_exp_conf,
                           dic_agent_conf=dic_agent_conf,
                           dic_traffic_env_conf=tot_traffic_env,
                           dic_path=tot_path,
                           batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)

    policy = config.DIC_AGENTS[args.algorithm](
        dic_agent_conf=dic_agent_conf,
        dic_traffic_env_conf=tot_traffic_env,
        dic_path=tot_path
    )

    metalearner = MetaLearner(sampler, policy,
                              dic_agent_conf=dic_agent_conf,
                              dic_traffic_env_conf=tot_traffic_env,
                              dic_path=tot_path
                              )

    if batch_id == 0:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        metalearner.meta_target_params = params

    else:
        params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)),
                                  'rb'))  # load the prameters of previous batch
        params = [params] * len(policy.policy_inter)
        metalearner.meta_params = params
        period = dic_agent_conf['PERIOD']
        target_id = int((batch_id - 1) / period)
        meta_params = pickle.load(
            open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (target_id * period)), 'rb'))
        meta_params = [meta_params] * len(policy.policy_inter)
        metalearner.meta_target_params = meta_params 

    metalearner.sample_metalight(tasks, batch_id)  # peform meta training
    # pickle.dump(meta_params, open(
    # os.path.join(self.sampler.dic_path[0]['PATH_TO_MODEL'], 'params' + "_" + str(self.batch_id) + ".pkl"), 'wb'))


def meta_step(dic_path, dic_agent_conf, dic_traffic_env_conf, batch_id, pre_train=False):
    '''
        update the common model's parameters of metalight using each Di'

        Arguments:
            dic_agent_conf:     dict,   configuration of agent
            dic_traffic_env_conf:   dict,   configuration of traffic environment
            dic_path:           dict,   path of source files and output files
            batch_id:           int,    round number
            img_train:
    '''
    grads = []
    try:
        f = open(os.path.join(dic_path['PATH_TO_GRADIENT'], "gradients_%d.pkl") % batch_id, "rb")
        while True:
            grads.append(pickle.load(f))
    except:
        pass
    if batch_id == 0:
        meta_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_init.pkl'), 'rb'))
    else:
        meta_params = pickle.load(open(os.path.join(dic_path['PATH_TO_MODEL'], 'params_%d.pkl' % (batch_id - 1)), 'rb'))
    tot_grads = dict(zip(meta_params.keys(), [0] * len(meta_params.keys())))
    for key in meta_params.keys():
        for g in grads:
            tot_grads[key] += g[key]
    _beta = dic_agent_conf['BETA']
    meta_params = dict(zip(meta_params.keys(),
                           [meta_params[key] - _beta * tot_grads[key] for key in meta_params.keys()]))

    # save the meta parameters
    if pre_train:
        pickle.dump(meta_params,
                    open(os.path.join(dic_path['PATH_TO_MODEL'], 'pre_params' + "_" + str(batch_id) + ".pkl"),
                         'wb'))  # save parameters to pickle
    else:
        pickle.dump(meta_params,
                    open(os.path.join(dic_path['PATH_TO_MODEL'], 'params' + "_" + str(batch_id) + ".pkl"), 'wb'))


if __name__ == '__main__':
    import os

    args = parse()  # defined in utils.py
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu  # default = ""

    main(args)
