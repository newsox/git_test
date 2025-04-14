# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 19:23
# @Author  : xu xinwen
# @FileName: readsif.py
# @Software: PyCharm

import h5py
from contextlib import ExitStack
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

i_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine"
sr = 0.05

# 用以确认选取哪一个波段的所有数据
sr = 0.05
wl_A = [735, 758]
# wl_A_narrow = [743, 758]
wl_B = [682, 692]
wl = np.linspace(640, 850, int((210 / sr) + 1))
n = 0
goal_wave = 740
for i in wl:
    # print(i)
    if i == wl_B[0]:
        b_s = n
    if i == wl_B[1]:
        b_e = n + 1
    if i == wl_A[0]:
        a_s = n
    if i == wl_A[1]:
        a_e = n + 1
    if i == goal_wave:
        goal_num = n
    n += 1
wave = [b_s, b_e, a_s, a_e, goal_num, goal_num-a_s]

sif_scope = np.array(pd.read_csv(os.path.join(i_path, f"sif_scope_{sr}.csv"), header=None, engine="python", comment='#', memory_map=True).T)
sif_goal = sif_scope[:,goal_num]

sif_goal_array = np.empty((60,4608))
for i in range(4608):
    sif_goal_array[:,i] = sif_goal
sif_goal_col = np.tile(sif_goal, 4608).reshape(-1,1)

with ExitStack() as stack:
    f_SIF_hf = stack.enter_context(h5py.File(os.path.join(i_path, f"SIF_re_hf_{sr}.h5"), 'r'))
    SIF_hf = f_SIF_hf[f"SIF_re_{sr}"]

    f_SIF = stack.enter_context(h5py.File(os.path.join(i_path, f"SIF_re_{sr}.h5"), 'r'))
    SIF = f_SIF[f"SIF_re_{sr}"]

    # 读取前600行数据
    data_hf = SIF_hf[:600, :]  # 假设是二维数据
    data = SIF[:600, :]

    data_goal_wave = SIF[:,wave[-1]]
    data_goal = data_goal_wave.reshape((60,4608),order='F')

    # 将求得的结果sr匹配为1nm
    data_origin = []
    for i in range(758-735+1):
        n = 20*i
        a = data[:,n]
        data_origin.append(a)

# 前600行保存为CSV
pd.DataFrame(data_hf).to_csv(os.path.join(i_path, "SIF_hf_top600.csv"), index=False, header=False)
pd.DataFrame(data).to_csv(os.path.join(i_path, "SIF_new_top600.csv"), index=False, header=False)

# 重塑为三维数组 (10, 60, 列数)
data_hf_3d = data_hf.reshape(10, 60, -1)
data_3d = data.reshape(10, 60, -1)
data_origin = np.array(data_origin)
data_origin_3d = data_origin.reshape(10,60,-1)



# 计算z轴（第一个轴）的均值
mean_hf = np.mean(data_hf_3d, axis=0)
mean_data = np.mean(data_3d, axis=0)
mean_data_origin = np.mean(data_origin_3d, axis=0) # sr=1nm



# 保存结果
pd.DataFrame(mean_hf).to_csv(os.path.join(i_path, "SIF_hf_mean.csv"), index=False, header=False)
pd.DataFrame(mean_data).to_csv(os.path.join(i_path, "SIF_new_mean.csv"), index=False, header=False)
pd.DataFrame(mean_data_origin).to_csv(os.path.join(i_path, "SIF_mean_origin.csv"), index=False, header=False)

pd.DataFrame(sif_goal_array).to_csv(os.path.join(i_path, f"SIF_scope_{goal_wave}.csv"), index=False, header=False)
pd.DataFrame(data_goal).to_csv(os.path.join(i_path, f"SIF_{goal_wave}.csv"), index=False, header=False)

pd.DataFrame(data_goal_wave).to_csv(os.path.join(i_path, f"SIF_col_{goal_wave}.csv"), index=False, header=False)
pd.DataFrame(sif_goal_col).to_csv(os.path.join(i_path, f"SIF_scope_col_{goal_wave}.csv"), index=False, header=False)