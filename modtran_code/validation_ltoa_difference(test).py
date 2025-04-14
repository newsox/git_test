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
    t_validation = f_validation[f't_validation_{sr}']

    # 原始计算时用到的r，s，t
    r_validation = f_validation[f'r_validation_{sr}']
    s_validation = f_validation[f's_validation_{sr}']
    t_vali = f_validation[f't_{sr}']

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

    f_art_algorithm = stack.enter_context(h5py.File(os.path.join(i_path, f'art_algorithm.h5'), 'w'))
    art_algorithm_r = f_art_algorithm.create_dataset(f"art_algotithm_r_{sr}", (t.shape[0]*r_scope.shape[0], t.shape[1]), dtype=np.float64)
    art_algorithm_s = f_art_algorithm.create_dataset(f"art_algotithm_s_{sr}", (t.shape[0]*r_scope.shape[0], t.shape[1]), dtype=np.float64)
    art_algorithm_t = f_art_algorithm.create_dataset(f"art_algotithm_t_{sr}", (t.shape[0]*r_scope.shape[0], t.shape[1]), dtype=np.float64)

    # 计算T_model
    n = 0
    for i in tqdm(range(t.shape[0]), desc='计算T_model中'):
        for m in range(r_scope.shape[0]):
            a = 1-(s[i,:]*r_scope[m,:])
            t_model[n,:] = t[i,:]/a
            art_algorithm_r[n,:] = r_scope[m,:]
            art_algorithm_s[n,:] = s[i,:]
            art_algorithm_t[n,:] = t[i,:]
            n+=1

    l = validation.shape[0]

    # 在循环前初始化交互模式和图对象
    plt.ion()
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12),  (ax13, ax14, ax15)) = plt.subplots(5, 3, figsize=(18, 18))  # 创建2个子图
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

    # （4）---- 配置SIF子图1 ----
    ax4.set_xlabel('Wavelength (nm)', labelpad=8)
    ax4.set_ylabel(r'SIF ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif, = ax4.plot([], [], 'c--', label='SIF_TOC(Ltoa_difference/T_up)')
    line_sif_with_t, = ax4.plot([], [], 'b--', label='SIF_TOA(Ltoa_difference)')
    line_sif_original, = ax4.plot([],[], 'r--', label= 'SIF_TOC(SCOPE_original)')
    # SIF子图的图例和网格
    ax4.legend(loc='upper right', framealpha=0.4)
    ax4.grid(True, linestyle=':', alpha=0.6)

    # （7）---- 配置SIF子图3 ----
    ax7.set_xlabel('Wavelength (nm)', labelpad=8)
    ax7.set_ylabel(r'SIF_TOC_Difference ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_toc, = ax7.plot([], [], 'y--', label='SIF_TOC_difference(original-algorithm)')
    # SIF子图的图例和网格
    ax7.legend(loc='upper right', framealpha=0.4)
    ax7.grid(True, linestyle=':', alpha=0.6)

    # （5）---- 配置SIF子图2 ----
    ax5.set_xlabel('Wavelength (nm)', labelpad=8)
    ax5.set_ylabel(r'SIF ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif2, = ax5.plot([], [], 'g--', label='SIF_TOA(SIF_SCOPE*T_up)')
    line_sif1, = ax5.plot([], [], 'b--', label='SIF_TOA(Ltoa_difference)')
    line_sif3, = ax5.plot([], [], 'r--', label='SIF_TOC(SCOPE_original)')
    line_sif4, = ax5.plot([], [], 'y--', label="SIF_TOA(SIF_SCOPE*T_model)")
    # SIF子图的图例和网格
    ax5.legend(loc='upper right', framealpha=0.4)
    ax5.grid(True, linestyle=':', alpha=0.6)

    # （8）---- 配置SIF子图4 ----
    ax8.set_xlabel('Wavelength (nm)', labelpad=8)
    ax8.set_ylabel(r'SIF_TOA_Difference ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_toa, = ax8.plot([], [], 'k--', label='SIF_TOA_difference(original*t-Ltoa_diifference)')
    # SIF子图的图例和网格
    ax8.legend(loc='upper right', framealpha=0.4)
    ax8.grid(True, linestyle=':', alpha=0.6)

    # （2）---- 配置T_up子图 ----
    ax2.set_xlabel('Wavelength (nm)', labelpad=8)
    ax2.set_ylabel(r'T(100%)', fontsize=10)
    line_t_up, = ax2.plot([], [], 'g--', label='T_up')
    line_t_us, = ax2.plot([], [], 'm--', label='T_us')
    line_t, = ax2.plot([], [], color='orange', linestyle='--', label = 'T')
    # T_up子图的图例和网格
    ax2.legend(loc='upper right', framealpha=0.4)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # （10）---- 配置T子图 ----
    ax10.set_xlabel('Wavelength (nm)', labelpad=8)
    ax10.set_ylabel(r'T(100%)', fontsize=10)
    line_t1, = ax10.plot([], [], 'r--', label='T(t/(1-s*ρ)')
    line_t2, = ax10.plot([], [], 'g--', label='T_up')
    line_t3, = ax10.plot([], [], color='orange', linestyle='--', label = 'T')
    line_t_algorithm, = ax10.plot([], [], 'k--', label='T_algorithm')
    # T_up子图的图例和网格
    ax10.legend(loc='upper right', framealpha=0.4)
    ax10.grid(True, linestyle=':', alpha=0.6)

    # （11）---- 配置T子图 ----
    ax11.set_xlabel('Wavelength (nm)', labelpad=8)
    ax11.set_ylabel(r'value(100%)', fontsize=10)
    line_t_dif, = ax11.plot([], [], 'k--', label='Tdif{t_up-(t/(1-s*ρ)}')
    # T_up子图的图例和网格
    ax11.legend(loc='upper right', framealpha=0.4)
    ax11.grid(True, linestyle=':', alpha=0.6)


    # （13）---- 配置sif子图 ----
    ax13.set_xlabel('Wavelength (nm)', labelpad=0.4)
    ax13.set_ylabel(r'SIF ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif_validation, = ax13.plot([], [], 'g--', label="SIF_TOA(test_data)")
    line_sif_t_model, = ax13.plot([], [], 'b--', label="SIF_TOA(SIF_SCOPE*T_model)")
    line_sif_o, = ax13.plot([], [], 'r--', label="SIF_TOC(original)")
    # T_up子图的图例和网格
    ax13.legend(loc='upper right', framealpha=0.4)
    ax13.grid(True, linestyle=':', alpha=0.6)

    # （14）---- 配置sif子图 ----
    ax14.set_xlabel('Wavelength (nm)', labelpad=8)
    ax14.set_ylabel(r'SIF_diff ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_sif_diff, = ax14.plot([], [], 'k--', label='SIFdiff')
    # T_up子图的图例和网格
    ax14.legend(loc='upper right', framealpha=0.4)
    ax14.grid(True, linestyle=':', alpha=0.6)


    # (3) ---- 配置r子图 ----
    ax3.set_xlabel('Wavelength (nm)', labelpad=8)
    ax3.set_ylabel(r'r (%)', fontsize=10)
    line_r_validation, = ax3.plot([],[],'r--', label='vali')
    line_r_algorithm, = ax3.plot([],[],'b--', label='algo')
    ax3.legend(loc='upper right', framealpha=0.4)
    ax3.grid(True, linestyle=':', alpha=0.6)

    #(6) ---- 配置s子图 ----
    ax6.set_xlabel('Wavelength (nm)', labelpad=8)
    ax6.set_ylabel(r'S ($\mathrm{mW \cdot m^{-2} \cdot sr^{-1} \cdot nm^{-1}}$)', fontsize=10)
    line_s_vali, = ax6.plot([],[],'r--', label='vali')
    line_s_algo, = ax6.plot([],[],'b--', label='algo')
    ax6.legend(loc='upper right', framealpha=0.4)
    ax6.grid(True, linestyle=':', alpha=0.6)


    #(9) ---- 配置t子图 ----
    ax9.set_xlabel('Wavelength (nm)', labelpad=8)
    ax9.set_ylabel(r'T (%)', fontsize=10)
    line_t_vali, = ax9.plot([], [], 'r--', label='vali')
    line_t_algo, = ax9.plot([], [], 'b--', label='algo')
    ax9.legend(loc='upper right', framealpha=0.4)
    ax9.grid(True, linestyle=':', alpha=0.6)

    #(12) ---- 配置T_vali & T_algo子图 ----
    ax12.set_xlabel('Wavelength (nm)', labelpad=8)
    ax12.set_ylabel(r'T', fontsize=10)
    line_T_vali, = ax12.plot([],[],'r--', label='vali')
    line_T_algo, = ax12.plot([],[],'b--', label='algo')
    ax12.legend(loc='upper right', framealpha=0.4)
    ax12.grid(True, linestyle=':', alpha=0.6)

    # (15) ---- 配置diff子图 ----
    ax15.set_xlabel('Wavelength (nm)', labelpad=8)
    ax15.set_ylabel(r'T (%)', fontsize=10)
    line_diff, = ax15.plot([], [], 'k--', label='diff')
    ax15.legend(loc='upper right', framealpha=0.4)
    ax15.grid(True, linestyle=':', alpha=0.6)

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

        # 更新Ltoa绘图数据
        line1.set_data(wl, ltoa_original)
        line2.set_data(wl, ltoa_without_sif)
        line3.set_data(wl, sif_toa_difference)
        ax1.set_xlim(min(wl), max(wl))
        ax1.set_ylim(0,
                     max(np.max(ltoa_original), np.max(ltoa_without_sif), np.max(sif_toa_difference)) + 0.1)

        # 更新SIF子图1
        line_sif.set_data(wl, sif_toc_t)
        line_sif_with_t.set_data(wl, sif_toa_difference)
        line_sif_original.set_data(wl, sif_orginal)
        ax4.set_xlim(min(wl), max(wl))
        ax4.set_ylim(0,
                     max(np.max(sif_toc_t), np.max(sif_toa_difference), np.max(sif_orginal)) + 0.1)

        line_toc_data = sif_orginal-sif_toc_t
        # 更新SIF子图3
        line_toc.set_data(wl, line_toc_data)
        ax7.set_xlim(min(wl), max(wl))
        ax7.set_ylim(np.min(line_toc_data) - 0.2,
                     np.max(line_toc_data) + 0.2)

        # 更新SIF子图2
        line_sif2.set_data(wl,sif_orginal_t_up)
        line_sif1.set_data(wl,sif_toa_difference)
        line_sif3.set_data(wl,sif_orginal)
        line_sif4.set_data(wl,sif_orginal_t_model)
        ax5.set_xlim(min(wl), max(wl))
        ax5.set_ylim(0,
                     max(np.max(sif_orginal_t_up), np.max(sif_toa_difference), np.max(sif_orginal), np.max(sif_orginal_t_model)) + 0.1)

        line_toa_data = sif_orginal_t_up - sif_toa_difference
        # 更新SIF子图4
        line_toa.set_data(wl, line_toa_data)
        ax8.set_xlim(min(wl), max(wl))
        ax8.set_ylim(np.min(line_toa_data) - 0.2,
                     np.max(line_toa_data) + 0.2)

        t_us_data = t_us[i,:]
        t_data = t[math.ceil(i/60),:]
        # 更新T_up子图
        line_t_up.set_data(wl, t_up_data)
        line_t_us.set_data(wl, t_us_data)
        line_t.set_data(wl, t_data)
        ax2.set_xlim(min(wl), max(wl))
        ax2.set_ylim(0,
                     max(np.max(t_up_data), np.max(t_us_data), np.max(t_data)) + 0.1)

        # 更新T_up子图
        line_t1.set_data(wl, t_model[i,:])
        line_t2.set_data(wl, t_up_data)
        line_t3.set_data(wl, t_data)
        line_t_algorithm.set_data(wl, t_validation[i, :])
        ax10.set_xlim(min(wl), max(wl))
        ax10.set_ylim(0,
                     max(np.max(t_model[i,:]), np.max(t_us_data), np.max(t_data)) + 0.1)

        line_t_dif_data = t_up_data - t_model[i,:]
        line_t_dif.set_data(wl, line_t_dif_data)
        ax11.set_xlim(min(wl),max(wl))
        ax11.set_ylim(-0.2,
                    0.2)


        line_sif_validation.set_data(wl, sif_validations[i,:])
        line_sif_t_model.set_data(wl,sif_orginal_t_model)
        line_sif_o.set_data(wl,sif_orginal)
        ax13.set_xlim(min(wl), max(wl))
        ax13.set_ylim(0,
                    max(np.max(sif_validations[i,:]), np.max(sif_orginal_t_model), np.max(sif_orginal)) + 0.1)

        sif_dif = sif_validations[i,:] - sif_orginal_t_model
        line_sif_diff.set_data(wl, sif_dif)
        ax14.set_xlim(min(wl), max(wl))
        ax14.set_ylim(np.min(sif_dif) - 0.1,
                      np.max(sif_dif) + 0.1)

        line_r_validation.set_data(wl, r_validation[i,:])
        line_r_algorithm.set_data(wl, art_algorithm_r[i,:])
        ax3.set_xlim(min(wl), max(wl))
        ax3.set_ylim(min(np.min(r_validation[i,:]), np.min(art_algorithm_r[i,:]))-0.1,
                     max(np.max(r_validation[i,:]), np.max(art_algorithm_r[i,:]))+0.1)

        line_s_vali.set_data(wl, s_validation[i,:])
        line_s_algo.set_data(wl, art_algorithm_s[i,:])
        ax6.set_xlim(min(wl),max(wl))
        ax6.set_ylim(min(np.min(s_validation[i,:]), np.min(art_algorithm_s[i,:]))-0.1,
                     max(np.max(s_validation[i,:]), np.max(art_algorithm_s[i,:]))+0.1)

        line_t_vali.set_data(wl, t_vali[i,:])
        line_t_algo.set_data(wl, art_algorithm_t[i,:])
        ax9.set_xlim(min(wl), max(wl))
        ax9.set_ylim(min(np.min(t_vali[i,:]), np.min(art_algorithm_t[i,:]))-0.1,
                     max(np.max(t_vali[i,:]), np.max(art_algorithm_t[i,:]))+0.1)

        T_vali = t_vali[i,:]*(1-(s_validation[i,:]*r_validation[i,:]))
        T_algo = art_algorithm_t[i,:]*(1-(art_algorithm_s[i,:]*art_algorithm_r[i,:]))
        line_T_vali.set_data(wl, T_vali)
        line_T_algo.set_data(wl, T_algo)
        ax12.set_xlim(min(wl),max(wl))
        ax12.set_ylim(min(np.min(T_vali), np.max(T_algo))-0.1,
                      max(np.max(T_vali), np.max(T_algo))+0.1)

        T_diff = T_vali-T_algo
        line_diff.set_data(wl, T_diff)
        ax15.set_xlim(min(wl), max(wl))
        ax15.set_ylim(min(T_diff), max(T_diff))



        # 更新指标文本（带颜色标记）
        metrics_text.set_text(
            f'Ltoa/SIF Order {i}/{l}| '
            f'SCOPE_turn: {r_indices_sif[i]}'
        )

        # 刷新图像
        plt.draw()
        plt.pause(0.01)

        # 循环结束后关闭交互模式
    plt.ioff()
    plt.show()

    print("debug")