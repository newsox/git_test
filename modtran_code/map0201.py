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


df_SIF = pd.read_excel(os.path.join(i_path, "compare_hf.xlsx"),
                        sheet_name="SIFinterp",
                        header=None) # 精确匹配工作表名称
df_SIF = df_SIF.to_numpy()
df_SIFsvs = pd.read_excel(os.path.join(i_path, "compare_hf.xlsx"),
                        sheet_name="SIFmean",
                        header=None) # 精确匹配工作表名称
df_SIFsvs = df_SIFsvs.to_numpy()
all_SIF = []
all_SIFsvs = []
wavelengths = []

with ExitStack() as stack:
    f_ems = stack.enter_context(h5py.File(os.path.join(i_path, f'compare_ems.h5'), 'w'))
    ems = f_ems.create_dataset(f'compare_ems', (461, 3), dtype=np.float64)

    # 关闭交互模式（重要！）
    plt.ioff()

    # 创建全局绘图画布
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-bright')
    plt.rcParams['font.family'] = 'Times New Roman'

    for i in range(461):
        SIF = df_SIF[i, :]
        SIFsvs = df_SIFsvs[i, :]

        # 收集数据
        all_SIF.extend(SIF)
        all_SIFsvs.extend(SIFsvs)
        wavelengths.extend([lambda_wl_a[i]] * len(SIF))  # 每个数据点关联当前波长

        # 保持原有的指标计算和存储
        r2 = r2_score(SIF, SIFsvs)
        rmse = np.sqrt(mean_squared_error(SIF, SIFsvs))
        mae = mean_absolute_error(SIF, SIFsvs)
        ems[i, :] = [r2, rmse, mae]

    # ======================
    # 全局绘图
    # ======================


    # 转换数据为numpy数组
    all_SIF = np.array(all_SIF)
    all_SIFsvs = np.array(all_SIFsvs)
    wavelengths = np.array(wavelengths)

    # 创建带颜色映射的散点图
    scatter = plt.scatter(
        all_SIF,
        all_SIFsvs,
        c=wavelengths,  # 用波长值作为颜色依据
        cmap='viridis',  # 使用Viridis色谱
        alpha=0.4,  # 半透明效果
        edgecolors='w',  # 白色边缘
        s=60,  # 点大小
        linewidths=0.5  # 边缘线宽
    )

    # 添加颜色条
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label('Wavelength (nm)', fontsize=26, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # 绘制参考线
    max_val = max(all_SIF.max(), all_SIFsvs.max())
    plt.plot([0, max_val + 1], [0, max_val + 1], '--', color='red', linewidth=2)

    # 设置坐标轴
    plt.xlim(0, max_val + 0.5)
    plt.ylim(0, max_val + 0.5)
    plt.xlabel('SIFreal', fontsize=24, fontweight='bold', labelpad=12)
    plt.ylabel('SIFpre', fontsize=24, fontweight='bold', labelpad=12)
    plt.title('SIF',
              fontsize=26, pad=20, fontweight='bold')

    # 设置刻度标签
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 获取当前坐标轴对象并加粗刻度数字
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    # 设置颜色条刻度加粗（可选）
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')

    # 计算全局指标
    global_r2 = r2_score(all_SIF, all_SIFsvs)
    global_rmse = np.sqrt(mean_squared_error(all_SIF, all_SIFsvs))
    global_mae = mean_absolute_error(all_SIF, all_SIFsvs)

    # # 添加统计信息标注
    # textstr = '\n'.join((
    #     f'Global $R^2$ = {global_r2:.3f}',
    #     f'Global RMSE = {global_rmse:.3f}',
    #     f'Global MAE = {global_mae:.3f}'))
    # plt.text(0.05, 0.95, textstr,
    #          transform=plt.gca().transAxes,
    #          fontsize=14,
    #          fontweight='bold',
    #          verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 设置坐标轴
    plt.xlim(0, max_val + 0.5)
    plt.ylim(0, max_val + 0.5)
    plt.xlabel('SIFreal', fontsize=24, fontweight='bold', labelpad=12)
    plt.ylabel('SIFpre', fontsize=24, fontweight='bold', labelpad=12)
    plt.title('',
              fontsize=26, pad=20, fontweight='bold')

    # 设置刻度标签
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 保存和显示
    plt.tight_layout()
    plt.savefig(os.path.join(i_path, 'global_comparison.png'), dpi=300)
    plt.show()