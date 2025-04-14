# -*- coding: utf-8 -*-
# @Time    : 2025/3/22 20:40
# @Author  : xu xinwen
# @FileName: map.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import h5py
from contextlib import ExitStack
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import linregress

print(plt.style.available)
# ======================
# 数据准备（替换为你的真实数据）
# ======================
i_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine"
sr = 0.05
# wl_A = [735, 758]
wl_A_narrow = [743, 758]
wl_B = [682, 692]
wl = np.linspace(640, 850, int((210 / sr) + 1))
n = 0
for i in wl:
    # print(i)
    if i == wl_B[0]:
        b_s = n
    if i == wl_B[1]:
        b_e = n + 1
    if i == wl_A_narrow[0]:
        a_s = n
    if i == wl_A_narrow[1]:
        a_e = n + 1
    n += 1
wave = [b_s, b_e, a_s, a_e]

# lambda_wl_a = np.linspace(wl_A[0], wl_A[1], int((wl[a_e] - wl[a_s]) / sr) + 1)
lambda_wl_a = np.linspace(wl_A_narrow[0], wl_A_narrow[1], int((wl[a_e] - wl[a_s]) / sr) + 1)
lambda_wl_b = np.linspace(wl_B[0], wl_B[1], int((wl[b_e] - wl[b_s]) / sr) + 1)

df_SIF = pd.read_excel(os.path.join(i_path, "compare_hf.xlsx"),
                        sheet_name="SIFinterp",
                        header=None) # 精确匹配工作表名称
df_SIF = df_SIF.to_numpy()
df_SIFsvs = pd.read_excel(os.path.join(i_path, "compare_hf.xlsx"),
                        sheet_name="SIFmean",
                        header=None) # 精确匹配工作表名称
df_SIFsvs = df_SIFsvs.to_numpy()
with ExitStack() as stack:
    f_ems = stack.enter_context(h5py.File(os.path.join(i_path, f'compare_ems.h5'), 'w'))
    ems = f_ems.create_dataset(f'compare_ems', (len(lambda_wl_a),3), dtype=np.float64)
    # 初始化交互模式
    plt.ion()
    plt.style.use('seaborn-v0_8-bright')  # 先设置样式，避免被后续字体覆盖
    plt.rcParams['font.family'] = 'Times New Roman'  # 全局设置字体

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(lambda_wl_a)):
        SIF = df_SIF[i,:]
        SIFsvs = df_SIFsvs[i,:]
        # 清除上一帧
        ax.cla()

        # --- 计算指标 ---
        r2 = r2_score(SIF, SIFsvs)
        rmse = np.sqrt(mean_squared_error(SIF, SIFsvs))
        mae = mean_absolute_error(SIF, SIFsvs)

        # --- 新增：线性拟合及指标计算 ---
        slope, intercept, r_value, _, _ = linregress(SIF, SIFsvs)
        r2_linear = r_value ** 2  # 线性模型的R²
        y_fit = slope * SIF + intercept  # 拟合值
        rmse_fit = np.sqrt(mean_squared_error(SIFsvs, y_fit))  # 拟合后的RMSE

        # 存储指标到h5文件（可选）
        print(lambda_wl_a[i])
        print([r2, r2_linear])
        ems[i, :] = [r2, rmse, mae]

        # --- 绘制散点图 ---
        ax.scatter(SIF, SIFsvs, s=80, facecolors='none',
                   edgecolors='dodgerblue', linewidths=2.4, alpha=0.8)

        # --- 设置字体并加粗 ---
        textstr = '\n'.join((
            f'$R^2$ = {r2:.3f}',
            f'RMSE = {rmse:.3f}',
            f'MAE = {mae:.3f}'))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                fontsize=12, fontweight='bold',  # 文本加粗
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # --- 坐标轴和标题加粗 ---
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.plot([0, 8], [0, 8], '--', color='grey', alpha=0.7)
        ax.set_xlabel('SIFreal', fontsize=14, labelpad=10, fontweight='bold')  # X轴标签加粗
        ax.set_ylabel('SIFpre', fontsize=14, labelpad=10, fontweight='bold')  # Y轴标签加粗
        ax.set_title(f'Wavelength:{lambda_wl_a[i]}nm', fontsize=16, pad=20, fontweight='bold')  # 标题加粗

        # 刻度标签（保持Times New Roman但不加粗）
        ax.tick_params(axis='both', which='major', labelsize=20)

        # 强制刷新并暂停
        plt.draw()
        plt.pause(0.01)

        # 关闭交互模式
    plt.ioff()
    plt.show()