# -*- coding: utf-8 -*-
# @Time    : 2025/3/23 1:19
# @Author  : xu xinwen
# @FileName: plot.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据：461组，每组10个点
n_groups = 461
np.random.seed(42)
x = np.random.randn(n_groups, 10)
y = np.random.randn(n_groups, 10)

# --- 定义编码规则 ---
# 1. 颜色：循环使用扩展色板（tab20 + tab20b + tab20c，共60色）
colors = plt.cm.tab20(np.arange(20))
colors = np.vstack([colors, plt.cm.tab20b(np.arange(20)), plt.cm.tab20c(np.arange(20))])

# 2. 形状：定义14种基础形状（覆盖461组需循环33次）
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X']

# 3. 透明度：每组固定alpha=0.6
alpha = 0.6

# --- 绘制散点图 ---
fig, ax = plt.subplots(figsize=(20, 15))

for i in range(n_groups):
    # 计算颜色和形状索引
    color_idx = i % 60
    marker_idx = (i // 60) % 14  # 每60组更换一次形状

    ax.scatter(x[i, :], y[i, :],
               color=colors[color_idx],
               marker=markers[marker_idx],
               s=40,
               alpha=alpha,
               edgecolor='k',
               linewidth=0.5,
               label=f'Group {i + 1}')

# --- 优化图例（仅展示前20组示例）---
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:20], labels[:20],
          loc='upper right',
          ncol=2,
          fontsize=8,
          framealpha=0.8)

# 坐标轴设置
ax.set_xlabel('X Axis', fontsize=14)
ax.set_ylabel('Y Axis', fontsize=14)
ax.set_title('461 Groups Scatter Plot with Multi-Encoding', fontsize=16)
ax.grid(linestyle='--', alpha=0.5)

plt.show()