# -*- coding: utf-8 -*-
# @Time    : 2025/3/21 16:45
# @Author  : xu xinwen
# @FileName: mat2h5.py
# @Software: PyCharm
import numpy as np
import scipy.io
import h5py
from contextlib import ExitStack
import os
i_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine"
# 加载.mat文件
mat_data = scipy.io.loadmat("F:\\research\\Processing_program\\FSM_code\\FSM_code_for_peers_all\\Full_SIFsvs.mat")

# 查看文件中的变量名（字典键）
print("MAT文件中的变量：", mat_data.keys())

# 提取具体变量（例如名为 'data' 的变量）
data = mat_data['Full_SIFsvs']
print("变量形状：", data.shape)

with ExitStack() as stack:
    f_SIFsvs = stack.enter_context(h5py.File(os.path.join(i_path, "SIFsvs.h5"), 'w'))
    SIFsvs = f_SIFsvs.create_dataset('SIFsvs', data=data.T, dtype=np.float64)
    print(SIFsvs)