# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 19:42
# @Author  : xu xinwen
# @FileName: SIF2h5.py
# @Software: PyCharm

import h5py
from contextlib import ExitStack
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

i_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine"
sif_path = "F:\\research\\Processing_program\\SCOPE_dataprocessing\\SCOPE_2.1\\SCOPE-2.1\\output\\MODTRAN_simulation_2024-12-30-1732\\fluorescence.csv"
sr = 0.05
sif_scope = pd.read_csv(sif_path, header=None, engine="python", comment='#', memory_map=True)
sif_scope = sif_scope.to_numpy()

wl_A = [735, 758]
wl_B = [682, 692]
wl = np.linspace(640, 850, int((210 / sr) + 1))
n = 0
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
    n += 1
wave = [b_s, b_e, a_s, a_e]

lambda_wl_a = np.linspace(wl_A[0], wl_A[1], int((wl[a_e] - wl[a_s]) / sr) + 1)
lambda_wl_b = np.linspace(wl_B[0], wl_B[1], int((wl[b_e] - wl[b_s]) / sr) + 1)
wl_sr = [lambda_wl_a, lambda_wl_b]
pd.DataFrame(wl_sr).to_csv(os.path.join(i_path, "wl_sr.csv"), index=False, header=False)


x_original = np.arange(640, 851, 1)
wl = np.linspace(640, 850, int((210 / sr) + 1))
# 假设 Full_SIFsvs 形状为 (211, 1000)，即 1000 条光谱
interp_func = interp1d(
    x_original,
    sif_scope,
    axis=1,  # 沿第一个维度（波长轴）插值
    kind='linear'
)
SIF_interp = interp_func(wl)
SIF_interp_a = SIF_interp[:, a_s:a_e]
pd.DataFrame(SIF_interp_a).to_csv(os.path.join(i_path, "SIF_interp_a.csv"), index=False, header=False)
print("debug")

# with ExitStack() as stack:
#     """SIF_scope"""
