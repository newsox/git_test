# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 22:16
# @Author  : xu xinwen
# @FileName: validation_ltoa_difference(test).py
# @Software: PyCharm

import os
import h5py
from contextlib import ExitStack
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

i_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine"
sr = 0.05
r_scope_path = "F:\\research\\Processing_program\\SCOPE_dataprocessing\\SCOPE_2.1\\SCOPE-2.1\\output\\MODTRAN_simulation_2024-12-30-1732\\reflectance.csv"

"""
筛选拟合窗口 已知所有数据的波段范围均在640-850nm范围，SR取决于拟合的输入值
现有拟合窗口，需要筛选出窗口对应的数据范围
窗口1：O2-B,735-758nm
窗口2：O2-A,682-692nm
"""

# # 定义反射率拟合窗口（及分辨率）
wl = np.linspace(640, 850, int((210 / sr) + 1))

# 读取SIF
sif_scope = pd.read_csv(os.path.join(i_path,"sif_scope_0.05.csv"), header=None, engine="python", memory_map=True).T
sif_scope = np.array(sif_scope)

with ExitStack() as stack:
    f_validation = stack.enter_context(h5py.File(os.path.join(i_path, f"ltoa_test_{sr}.h5"), 'r'))
    validation = f_validation[f"validation_difference_{sr}"]
    sif_validations = f_validation[f'sif_validation_{sr}']
    ltoa = f_validation[f"ltoa_test_{sr}"]
    t_validation= f_validation[f't_validation_{sr}']

    ltoa_validation = f_validation[f"ltoa_test_validation_{sr}"]

    f_t_up = stack.enter_context(h5py.File(os.path.join(i_path, f"t_up_test_{sr}.h5"), 'r'))
    t_up = f_t_up[f"t_up_test_{sr}"]

    f_t_us = stack.enter_context(h5py.File(os.path.join(i_path, f"t_us_test_{sr}.h5"), 'r'))
    t_us = f_t_us[f"t_us_test_{sr}"]

    # 上行透过率  T=τoo+τdo
    f_t = stack.enter_context(h5py.File(os.path.join(i_path, f'T_{sr}.h5'), 'r'))
    t = f_t[f'T_{sr}'][()].T

    f_s = stack.enter_context(h5py.File(os.path.join(i_path, f'S_{sr}.h5'), 'r'))
    s = f_s[f'S_{sr}'][()].T

    r_scope = pd.read_csv(os.path.join(i_path, f"r_scope_{sr}.csv"), header=None, engine="python", memory_map=True).T  # comment='#' 来跳过注释行
    r_scope = np.array(r_scope)

    f_t_model = stack.enter_context(h5py.File(os.path.join(i_path, f"T_model_{sr}.h5"), 'w'))
    t_model = f_t_model.create_dataset(f"T_model_{sr}", (t.shape[0]*r_scope.shape[0], t.shape[1]), dtype=np.float64)

    # 计算T_model
    n = 0
    for i in tqdm(range(t.shape[0]), desc='计算T_model中'):
        for m in range(r_scope.shape[0]):
            a = 1-(s[i,:]*r_scope[m,:])
            t_model[n,:] = t[i,:]/a
            n+=1

    l = validation.shape[0]

    # 在循环前初始化交互模式和图对象
    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(18, 18))  # 创建2个子图
    fig.suptitle('Ltoa/SIF validation')  # 主标题

    # （1）---- 配置Ltoa子图 ----
    ax1.set_xlabel('Wavelength (nm)', labelpad=8)
    ax1.set_ylabel(r'Radiance ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line1, = ax1.plot([], [], 'b--', label='Original Ltoa')
    line2, = ax1.plot([], [], 'r--', label='Ltoa without SIF')
    line3, = ax1.plot([], [], 'g--', label='Difference')
    # LTOA子图的图例和网格
    ax1.legend(loc='upper right', framealpha=0.4)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # （2）---- 配置SIF子图1 ----
    ax2.set_xlabel('Wavelength (nm)', labelpad=8)
    ax2.set_ylabel(r'SIF ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif, = ax2.plot([], [], 'c--', label='SIF_TOC(Ltoa_difference/T_up)')
    line_sif_with_t, = ax2.plot([], [], 'b--', label='SIF_TOA(Ltoa_difference)')
    line_sif_original, = ax2.plot([],[], 'r--', label= 'SIF_TOC(SCOPE_original)')
    # SIF子图的图例和网格
    ax2.legend(loc='upper right', framealpha=0.4)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # （2）---- 配置T_up子图 ----
    ax3.set_xlabel('Wavelength (nm)', labelpad=8)
    ax3.set_ylabel(r'T(100%)', fontsize=10)
    line_t_up, = ax3.plot([], [], 'g--', label='T_up')
    line_t_us, = ax3.plot([], [], 'm--', label='T_us')
    line_t, = ax3.plot([], [], color='orange', linestyle='--', label='T(τoo+τdo)')
    line_t_algorithm, = ax3.plot([], [], 'r--', label='T(t/(1-s*ρ))')
    # T_up子图的图例和网格
    ax3.legend(loc='upper right', framealpha=0.4)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # （4）---- 配置SIF子图2 ----
    ax4.set_xlabel('Wavelength (nm)', labelpad=8)
    ax4.set_ylabel(r'SIF ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif2, = ax4.plot([], [], 'g--', label='SIF_TOA(SIF_SCOPE*T_up)')
    line_sif1, = ax4.plot([], [], 'b--', label='SIF_TOA(Ltoa_difference)')
    line_sif3, = ax4.plot([], [], 'r--', label='SIF_TOC(SCOPE_original)')
    line_sif4, = ax4.plot([], [], 'y--', label="SIF_TOA(SIF_SCOPE*T_model)")
    # SIF子图的图例和网格
    ax4.legend(loc='upper right', framealpha=0.4)
    ax4.grid(True, linestyle=':', alpha=0.6)

    # （5）---- 配置Tdiff子图 ----
    ax5.set_xlabel('Wavelength (nm)', labelpad=8)
    ax5.set_ylabel(r'Value(100%)', fontsize=10)
    line_t_diff, = ax5.plot([],[],'k--', label='T_up-(t/(1-s*ρ))')
    ax5.legend(loc='upper right', framealpha=0.4)
    ax5.grid(True, linestyle=':', alpha=0.6)

    # (6) ---- 配置SIFdiff子图 ----
    ax6.set_xlabel('Wavelength (nm)', labelpad=8)
    ax6.set_ylabel(r'Value($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif_diff, = ax6.plot([],[],'k--', label='(SIF_SCOPE*T_up)-SIF_TOA(Ltoa_difference)')
    ax6.legend(loc='upper right', framealpha=0.4)
    ax6.grid(True, linestyle=':', alpha=0.6)

    # ---- 全局指标文本（显示在图形底部） ----
    metrics_text = fig.text(
        0.5, 0.05,  # x居中，y靠近底部
        '',
        ha='center',
        va='top',
        fontsize=10,
        color='darkred',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
    )

    # ---- 布局优化 ----
    plt.subplots_adjust(
        hspace=0.3,  # 子图垂直间距
        bottom=0.15  # 底部留白给指标文本
    )

    # 计算索引
    r_indices_sif = [divmod(i, sif_scope.shape[0])[1] for i in range(l)]

    for i in range(l):
        ltoa_original = ltoa[i,:]
        ltoa_without_sif = ltoa_validation[i,:]
        sif_toa_difference = validation[i,:]
        t_up_data = t_up[i,:]
        sif_toc_t = sif_toa_difference/t_up_data
        sif_orginal = sif_scope[r_indices_sif[i],:]
        sif_orginal_t_up = sif_orginal*t_up_data
        sif_orginal_t_model = sif_orginal*t_model[i,:]

        # (1)更新Ltoa绘图数据
        line1.set_data(wl, ltoa_original)
        line2.set_data(wl, ltoa_without_sif)
        line3.set_data(wl, sif_toa_difference)
        ax1.set_xlim(min(wl), max(wl))
        ax1.set_ylim(0,
                     max(np.max(ltoa_original), np.max(ltoa_without_sif), np.max(sif_toa_difference)) + 0.1)

        # (2)更新SIF子图
        line_sif.set_data(wl, sif_toc_t)
        line_sif_with_t.set_data(wl, sif_toa_difference)
        line_sif_original.set_data(wl, sif_orginal)
        ax2.set_xlim(min(wl), max(wl))
        ax2.set_ylim(0,
                     max(np.max(sif_toc_t), np.max(sif_toa_difference), np.max(sif_orginal)) + 0.1)

        # (3)更新T子图
        t_us_data = t_us[i, :]
        t_data = t[math.ceil(i / 60), :]
        # 更新T_up子图
        line_t_up.set_data(wl, t_up_data)
        line_t_us.set_data(wl, t_us_data)
        line_t.set_data(wl, t_data)
        line_t_algorithm.set_data(wl, t_validation[i, :])
        ax3.set_xlim(min(wl), max(wl))
        ax3.set_ylim(0,
                     max(np.max(t_up_data), np.max(t_us_data), np.max(t_data)) + 0.1)

        # (4)更新SIF子图
        line_sif2.set_data(wl, sif_orginal_t_up)
        line_sif1.set_data(wl, sif_toa_difference)
        line_sif3.set_data(wl, sif_orginal)
        line_sif4.set_data(wl, sif_orginal_t_model)
        ax4.set_xlim(min(wl), max(wl))
        ax4.set_ylim(0,
                     max(np.max(sif_orginal_t_up), np.max(sif_toa_difference), np.max(sif_orginal),np.max(sif_orginal_t_model)) + 0.1)

        # (5)更新Tdiff
        t_diff = t_up_data - t_validation[i,:]
        line_t_diff.set_data(wl, t_diff)
        ax5.set_xlim(min(wl),max(wl))
        ax5.set_ylim(min(t_diff)-0.1,
                     max(t_diff)+0.1)

        # (6)更新SIFdiff
        sif_diff = sif_orginal_t_up - sif_toa_difference
        line_sif_diff.set_data(wl, sif_diff)
        ax6.set_xlim(min(wl),max(wl))
        ax6.set_ylim(min(sif_diff)-0.1,
                     max(sif_diff)+0.1)

        # 更新指标文本（带颜色标记）
        metrics_text.set_text(
            f'Ltoa/SIF Order {i}/{l}| '
            f'MODTRAN_turn:{divmod(i, 60)[0]+1}| '
            f'SCOPE_turn: {r_indices_sif[i]}'
        )

        # 刷新图像
        plt.draw()
        plt.pause(0.01)

        # 循环结束后关闭交互模式
    plt.ioff()
    plt.show()

    print("debug")
