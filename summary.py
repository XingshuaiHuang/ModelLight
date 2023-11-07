import pickle as pkl
import os
import pandas as pd
import numpy as np
import json
import copy
from math import isnan
import matplotlib as mlp
from matplotlib.offsetbox import AnchoredText
from utils import *
import math
import shutil
from traffic_meta_test import *

mlp.use("agg")  # figures are not displayed in PyCharm
import matplotlib.pyplot as plt

font = {'size': 24}
mlp.rc('font', **font)  # set the font of all items to the font specified by the kwargs object, font

NAN_LABEL = -1


def get_metrics(duration_list, min_duration, min_duration_id,
                traffic_name, total_summary, mode_name, save_path, num_rounds, min_duration2=None, min_duration_log=None):
    # total_summary = get_metrics(duration,
    #                                 min_duration, min_duration_ind, traffic_name, total_summary,
    #                                 mode_name="test", save_path=figures_path, num_rounds=duration.size,
    #                                 )
    validation_duration_length = 10
    minimum_round = 50 if num_rounds > 50 else 0
    duration_list = np.array(duration_list)

    nan_count = len(np.where(duration_list == NAN_LABEL)[0])  # count durations equal to 0
    validation_duration = duration_list[-validation_duration_length:]  # the last 10 durations
    final_duration = np.round(np.mean(validation_duration[validation_duration > 0]), decimals=2)  # mean of validation_duration
    final_duration_std = np.round(np.std(validation_duration[validation_duration > 0]), decimals=2)  # standard deviation of validation_duration

    if nan_count == 0:
        convergence = {1.2: len(duration_list) - 1, 1.1: len(duration_list) - 1}
        for j in range(minimum_round, len(duration_list)):
            for level in [1.2, 1.1]:
                if max(duration_list[j:]) <= level * final_duration:
                    if convergence[level] > j:
                        convergence[level] = j  # the index of the duration next to the last duration that is <= level * final_duration
        conv_12 = convergence[1.2]
        conv_11 = convergence[1.1]
    else:
        conv_12, conv_11 = 0, 0

    # --------simple plot for each training instance----------
    f, ax = plt.subplots(1, 1, figsize=(12, 9), dpi=100)
    ax.plot(duration_list, linewidth=2, color='k')
    ax.plot([0, len(duration_list)], [final_duration, final_duration], linewidth=2, color="g")

    ax.plot([conv_12, conv_12], [duration_list[conv_12], duration_list[conv_12] * 3], linewidth=2, color="b")
    ax.plot([conv_11, conv_11], [duration_list[conv_11], duration_list[conv_11] * 3], linewidth=2, color="b")
    ax.plot([0, len(duration_list)], [min_duration, min_duration], linewidth=2, color="r")  # horizontal line implying the min_duration
    ax.plot([min_duration_id, min_duration_id], [min_duration, min_duration * 3], linewidth=2, color="r")  # vertical line indicating the time position of the min_duration

    init_dur = str(int(duration_list[0])) if not math.isnan(duration_list[0]) else 'NaN'
    min_dur = str(int(min_duration)) if not math.isnan(min_duration) else 'NaN'
    print(traffic_name, final_duration)
    final_dur = str(int(final_duration)) if not math.isnan(final_duration) else 'NaN'

    if min_duration == 'NaN':
        min_duration = str(int(min_duration))

    anchored_text = AnchoredText("Initial_duration: %s\nMin_duration: %s\nFinal_duration: %s"%(init_dur, min_dur, final_dur), loc=1)  # legend
    ax.add_artist(anchored_text)

    ax.set_title(traffic_name + "-" + str(final_duration))
    plt.savefig(save_path + "/" + traffic_name + "-" + mode_name + ".png")  # √ summary/memo/figures/traffic_name-test.png
    figure_2 = os.path.join(os.path.dirname(save_path), 'total_figures')  # summary/memo/total_figures
    if not os.path.exists(figure_2):
        os.makedirs(figure_2)

    traffic_file = traffic_name  # 'hangzhou_baochu_tiyuchang_1h_10_11_2021.json_12_13_13_20_09'
    if ".xml" in traffic_file:
        traffic_name, traffic_time = traffic_file.split(".xml")  # use .xml as separator
    elif ".json" in traffic_file:
        traffic_name, traffic_time = traffic_file.split(".json")  # use .json as separator
    plt.savefig(figure_2 + "/" + traffic_name + ".png")  # √ summary/memo/total_figures/traffic_name.png
    plt.close()

    traffic_phase = traffic_name.split(".xml")[0]
    traffic_name = traffic_name + '.json'
    if traffic_name in paper_meta_valid_4a_phase_traffic_list or traffic_name in paper_meta_test_4a_different_cities:
        traffic_phase = '4a'
    elif traffic_name in paper_meta_valid_4b_phase_traffic_list or traffic_name in paper_meta_test_4b_different_cities:
        traffic_phase = '4b'
    elif traffic_name in paper_meta_test_4c_different_cities:
        traffic_phase = '4c'
    elif traffic_name in paper_meta_test_4d_different_cities:
        traffic_phase = '4d'
    elif traffic_name in paper_meta_valid_6a_phase_traffic_list or traffic_name in paper_meta_test_6a_different_cities:
        traffic_phase = '6a'
    elif traffic_name in paper_meta_valid_6c_phase_traffic_list:
        traffic_phase = '6c'
    elif traffic_name in paper_meta_test_6d_different_cities:
        traffic_phase = '6d'
    elif traffic_name in paper_meta_valid_6e_phase_traffic_list:
        traffic_phase = '6e'
    elif traffic_name in paper_meta_test_6f_different_cities:
        traffic_phase = '6f'
    elif traffic_name in paper_meta_valid_8_phase_traffic_list:
        traffic_phase = '8p'
    print('-----traffic_phase is {}-----'.format(traffic_phase))

    # ---------------------------------------------------------

    total_summary["traffic_file"].append(traffic_name)
    # total_summary["traffic"].append(traffic_name.split(".xml")[0])
    total_summary["traffic"].append(traffic_phase)
    total_summary["min_duration"].append(min_duration)
    total_summary["min_duration_round"].append(min_duration_id)
    total_summary["final_duration"].append(final_duration)
    total_summary["final_duration_std"].append(final_duration_std)
    total_summary["convergence_1.2"].append(conv_12)
    total_summary["convergence_1.1"].append(conv_11)
    total_summary["nan_count"].append(nan_count)
    total_summary["min_duration2"].append(min_duration2)  # default = None
    if min_duration_log:  # default = None
        for i in range(len(min_duration_log)):
            total_summary['md_%d'%((i + 1) * 10)].append(min_duration_log[i][0])
            total_summary['md_ind_%d' %((i + 1) * 10)].append(min_duration_log[i][1])

    return total_summary

def get_planed_entering(flowFile, episode_len):
    list_flow = json.load(open(flowFile, "r"))
    dic_traj = {'vehicle_id':[], 'planed_enter_time':[]}
    for flow_id, flow in enumerate(list_flow):
        list_ts_this_flow = []
        for step in range(flow["startTime"], min(flow["endTime"] + 1, episode_len)):
            if step == flow["startTime"]:
                list_ts_this_flow.append(step)
            elif step - list_ts_this_flow[-1] >= flow["interval"]:
                list_ts_this_flow.append(step)

        for vec_id, ts in enumerate(list_ts_this_flow):
            dic_traj['vehicle_id'].append("flow_{0}_{1}".format(flow_id, vec_id))
            dic_traj['planed_enter_time'].append(ts)

    df = pd.DataFrame(dic_traj)
    return df

def cal_travel_time(df_vehicle_actual_enter_leave, df_vehicle_planed_enter, episode_len):
    df_vehicle_planed_enter.set_index('vehicle_id', inplace=True)
    df_vehicle_actual_enter_leave.set_index('vehicle_id', inplace=True)
    df_res = pd.concat([df_vehicle_planed_enter, df_vehicle_actual_enter_leave], axis=1, sort=False)
    assert len(df_res) == len(df_vehicle_planed_enter)

    df_res["leave_time"].fillna(episode_len, inplace=True)
    df_res["travel_time"] = df_res["leave_time"] - df_res["planed_enter_time"]
    travel_time = df_res["travel_time"].mean()
    return travel_time

def summary_meta_test(memo):
    ''' directly copy the write_summary'''
    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }

    path = os.path.join("records", memo)  # records/memo/
    # !!! delete the previous files in path before executing a new summary_meta_test
    for traffic in os.listdir(path):  # each file or folder in records/memo/: 'hangzhou_baochu_tiyuchang_1h_10_11_2021.json_12_13_13_20_09'
        traffic_name = traffic[: traffic.find(".json") + len(".json")]  # 'hangzhou_baochu_tiyuchang_1h_10_11_2021.json'
        task_name = traffic_name
        res_path = os.path.join(path, traffic, "test_round", task_name, "test_results.csv")  # records/memo/traffic/test round/task_name/test_results.csv
                                                                             # utils.py (def write_summary: path_to_log)
        res_summary_path = os.path.join("summary", memo, "total_results")
        fig_summary_path = os.path.join("summary", memo, "total_figures")  # save figures by get_metrics function
        figures_path = os.path.join("summary", memo, "figures")  # save figures by get_metrics function
        if not os.path.exists(res_summary_path):
            os.makedirs(res_summary_path)
        if not os.path.exists(fig_summary_path):
            os.makedirs(fig_summary_path)
        if not os.path.exists(figures_path):
            os.makedirs(figures_path)

        shutil.copy(res_path, os.path.join(res_summary_path, traffic[:traffic.find(".json")] + ".csv"))  # 'hangzhou_baochu_tiyuchang_1h_10_11_2021.csv'
        df = pd.read_csv(res_path)  # round  duration  vec_in  vec_out
        duration = df["duration"]  # <class 'pandas.core.series.Series'>
        min_duration = duration.min()
        min_duration_ind = duration[duration == min_duration].index[0]  # the index of min_duration in the column of duration; todo index[0]
        total_summary = get_metrics(duration,
                                    min_duration, min_duration_ind, traffic_name, total_summary,
                                    mode_name="test", save_path=figures_path, num_rounds=duration.size,
                                    )
    total_result = pd.DataFrame(total_summary)

    total_result.to_csv(os.path.join("summary", memo, "total_test_results.csv"))

def summary_sotl(memo):
    # each_round_train_duration
    total_summary = {
        "traffic": [],
        "min_queue_length": [],
        "min_queue_length_round": [],
        "min_duration": [],
        "min_duration_round": []
    }

    records_dir = os.path.join("records", memo)
    for traffic_file in os.listdir(records_dir):
        if ".xml" not in traffic_file and ".json" not in traffic_file:
            continue
        print(traffic_file)

        # get episode_len to calculate the queue_length each second
        exp_conf = open(os.path.join(records_dir, traffic_file, "exp.conf"), 'r')
        dic_exp_conf = json.load(exp_conf)
        episode_len = dic_exp_conf["EPISODE_LEN"]

        duration_each_round_list = []
        queue_length_each_round_list = []

        # *** same as utils.py(write_summary(record_dir))
        # train_dir = os.path.join(records_dir, traffic_file)  # records/memo/traffic_file
        traffic_name = traffic_file[: traffic_file.find(".json") + len(".json")]
        cnt_round = 0
        train_dir = os.path.join(records_dir, traffic_file, "test_round", traffic_name,
                              "tasks_round_" + str(cnt_round))
        # train_dir: records/memo/traffic_file/test_round/test_traffic/tasks_round_batch_id
        # ***
        # summary items (queue_length) from pickle
        f = open(os.path.join(train_dir, "inter_0.pkl"), "rb") 
        # samples = pkl.load(f)  # ***
        # print('------------samples are {}------------'.format(samples))  # ***
        try:
            samples = pkl.load(f)
        except:
            continue
        for sample in samples:
            queue_length_each_round = sum(sample['state']['lane_queue_length'])
        f.close()

        # summary items (duration) from csv
        df_vehicle_inter_0 = pd.read_csv(os.path.join(train_dir, "vehicle_inter_0.csv"),
                                         sep=',', header=0, dtype={0: str, 1: float, 2: float},
                                         names=["vehicle_id", "enter_time", "leave_time"])
        duration = df_vehicle_inter_0["leave_time"].values - df_vehicle_inter_0["enter_time"].values
        ave_duration = np.mean([time for time in duration if not isnan(time)])
        # print(ave_duration)
        duration_each_round_list.append(ave_duration)
        ql = queue_length_each_round / len(samples)
        print('------------queue_length_each_round is {}------------'.format(queue_length_each_round))  # ***
        print('------------ql is {}------------'.format(ql))  # ***
        print('--------------summary sotl start--------------')  # ***
        queue_length_each_round_list.append(ql)

        # result_dir = os.path.join(records_dir, traffic_file)
        result_dir = os.path.join("summary", memo, traffic_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _res = {
            "duration": duration_each_round_list,
            "queue_length": queue_length_each_round_list
        }
        result = pd.DataFrame(_res)
        result.to_csv(os.path.join(result_dir, "test_results.csv"))
        # print(os.path.join(result_dir, "test_results.csv"))

        total_summary["traffic"].append(traffic_file)
        total_summary["min_queue_length"].append(ql)
        total_summary["min_queue_length_round"].append(0)
        total_summary["min_duration"].append(ave_duration)
        total_summary["min_duration_round"].append(0)

    sotl_sum_path = os.path.join("summary", memo)
    if not os.path.exists(sotl_sum_path):
        os.makedirs(sotl_sum_path)
    total_result = pd.DataFrame(total_summary)
    total_result.to_csv(os.path.join(sotl_sum_path, "total_sotl_test_results.csv"))  


if __name__ == "__main__":
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--memo', type=str, default='default')
    parse.add_argument('--type', type=str, default='train')
    parse.add_argument('--max_round', type=int, default=5)

    args = parse.parse_args()  # New parser: only 3 arguments

    total_summary = {
        "traffic": [],
        "traffic_file": [],
        "min_duration": [],
        "min_duration_round": [],
        "final_duration": [],
        "final_duration_std": [],
        "convergence_1.2": [],
        "convergence_1.1": [],
        "nan_count": [],
        "min_duration2": []
    }
    for i in range(1, args.max_round+1):
        total_summary['md_%d' % (i * 10)] = []
        total_summary['md_ind_%d' % (i * 10)] = []
    memo = args.memo

    if args.type == "meta_test":
        summary_meta_test(memo)
    elif args.type == "sotl":
        summary_sotl(memo)