import os
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import pylab
import torch

def pack_data(type_need_num):
    cur_path = os.getcwd()
    data_path = os.path.join(cur_path)
    dir_list = os.listdir(data_path)
    data_type = ['Acceleration', 'Emg', 'Gyroscope']
    final_target = []
    final_data = []
    type_count = 0
    i = 0
    while (i < type_need_num):
        print('第'+str(i+1)+'种手语--------------------------------------------------------')
        try:
            data_type_os = []
            for type_s in data_type:
                data_type_os.append(os.path.join(data_path, dir_list[i], type_s))
            npy_list = []
            for data_type_os_path in data_type_os:
                npy_list.append(os.listdir(data_type_os_path))
            for npy_num in range(len(npy_list[0])):
                single_final_data = []

                for type_num in range(len(data_type)):
                    raw_data = np.load(os.path.join(data_type_os[type_num], npy_list[type_num][npy_num])).T
                    if type_num == 1:
                        raw_data = raw_data[:-1, :]
                    try:
                        single_final_data.append(raw_data[:, 10:170])
                    except:
                        print(os.path.join(data_type_os[type_num], npy_list[type_num][npy_num]))
                single = np.vstack(k for k in single_final_data)
                if single.shape != (14, 160):
                    print('--------------------------------error-----------------------------------')
                    continue
                final_data.append(single)
                final_target.append(type_count)
                # print(len(final_data))
            type_count += 1
            i += 1
        except:
            pass

    raw_data_dist = {}
    raw_data_dist['data'] = final_data
    raw_data_dist['target'] = final_target
    torch.save(raw_data_dist, os.path.join((cur_path), type_need_num.__str__()+'_raw.mat'))
    print("ok", type_count)

if __name__=='__main__':
    pack_data(50)