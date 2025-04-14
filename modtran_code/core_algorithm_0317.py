# -*- coding: utf-8 -*-
# @Time    : 2024/12/16 14:39
# @Author  : xu xinwen
# @FileName: Dataset_Algorithm.py
# @Software: PyCharm
import math
import os
import numpy as np
from fontTools.ttLib.macUtils import SFNTResourceReader
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
import csv
import dask.dataframe as dd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import h5py
from contextlib import ExitStack
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MODTRAN_Interrogation_Technique中计算了L0,S,Ltoc,T，分两个模块，训练集（非荧光地表）和测试集（荧光地表）

# 1.读取文件
def read_paras(i_path, r_scope_path, sif_path, r_usgs_path, sr):

    para_name = ['L0.csv', 'S.csv', 'Ltoc.csv', 'T_up.csv', 'T.csv']
    paths = []
    for i in para_name:
        comb_path = os.path.join(i_path, i)
        paths.append(comb_path)
    # 以下数据均只读取640-850nm

    # MODTRAN计算的大气参数
    l0_data = pd.read_csv(paths[0], header=None, engine="python", memory_map=True)  # MODTRAN模拟的数据是从300nm-2000nm
    s_data = pd.read_csv(paths[1], header=None, engine="python", memory_map=True)  # 筛选640-850nm
    ltoc_data = pd.read_csv(paths[2], header=None, engine="python", memory_map=True)
    t_data = pd.read_csv(paths[4], header=None, engine="python", memory_map=True)  # τoo+τdo

    # SCOPE模拟的数据
    r_scope = pd.read_csv(r_scope_path, header=None, engine="python", comment='#', usecols=range(240, 451),
                          memory_map=True)  # comment='#' 来跳过注释行
    sif_scope = pd.read_csv(sif_path, header=None, engine="python", comment='#', memory_map=True)
    r_scope = r_scope.T
    sif_scope = sif_scope.T

    # USGS 反射率
    r_usgs = pd.read_csv(r_usgs_path, header=None, engine="python", memory_map=True)

    # # dataset_test用
    # ltoc_data = pd.read_csv(paths[2], header=None, engine="python", skiprows=101-1, nrows=1601)  # SCOPE模拟的数据在400-2000+nm
    # r_scope = pd.read_csv(r_scope_path, header=None, engine="python", comment='#', usecols=range(0, 1601))
    # r_scope = r_scope.T
    """
    # 以上数据的光谱分辨率和波段范围需要统一
    # 640-850nm
    # MODTRAN(SR=0.05,0.1,0.3,0.5)
    # sloar(SR=0.01) 不放在这里进行重采样（变化较大）
    # USGS(SR=1nm)
    # SCOPE r,SIF (1nm)
    # 插值方法的选取？？ 根据 USGS反射率,SCOPE 反射率和SIF的形状（较为平滑），可以采取样条插值
    """

    def spectral_interpolate(data, sr):
        wl = np.linspace(640, 850, 211)  # 这里不用np.arange() 可能因步长舍入误差不精确
        new_wl = np.linspace(640, 850, int((210 / sr) + 1))
        new_data = []
        rows, cols = data.shape
        for i in tqdm(range(cols), desc=f"进行插值运算中"):
            intensity = np.array(data.iloc[:, i])
            cs = CubicSpline(wl, intensity)
            new_intensity = cs(new_wl)
            new_data.append(new_intensity)

        new_data = pd.DataFrame(new_data)
        new_data = new_data.T
        new_data.index = new_wl.tolist()
        return new_wl, new_data

    r_usgs_wl, r_usgs = spectral_interpolate(r_usgs, sr)
    r_scope_wl, r_scope = spectral_interpolate(r_scope, sr)
    sif_scope_wl, sif_scope = spectral_interpolate(sif_scope, sr)
    r_usgs.to_csv(os.path.join(i_path, f"r_usgs_{sr}.csv"), header=None, index=False)
    r_scope.to_csv(os.path.join(i_path, f"r_scope_{sr}.csv"), header=None, index=False)
    sif_scope.to_csv(os.path.join(i_path, f"sif_scope_{sr}.csv"), header=None, index=False)

    return l0_data, s_data, ltoc_data, t_data, sif_scope, r_scope, r_usgs


# 测试MODTRAN和SCOPE是否可以耦合仿真数据集
def dataset_test(i_path, ltoc_data, r_scope):
    # 检验MODTRAN模拟的Ltoc数据与SCOPE模型模拟的Rflectance数据的准确性
    lrefs = []
    for i in range(r_scope.shape[1]):
        a = r_scope.iloc[:, i]
        b = ltoc_data
        c = b.mul(a, axis=0)
        lrefs.append(c)
    lref = pd.concat(lrefs, axis=1)
    lref = lref.T  # excel限制为支持最大1048576行和16384列的数据，这里转置存储，读取时候再转置即可
    lref.to_csv(os.path.join(i_path, "lref.csv"), index=False, header=None)


# 计算训练集
def train_set(output_dir, l0_data, s_data, ltoc_data, t_data, r_usgs, sr):
    air_rows, air_cols = l0_data.shape  # 4608
    ref_rows, ref_cols = r_usgs.shape  # 12

    def compute_single(m, n):
        a = t_data.iloc[:, m].values * (10 * ltoc_data.iloc[:, m].values * r_usgs.iloc[:, n].values)
        b = 1 - (s_data.iloc[:, m].values * r_usgs.iloc[:, n].values)
        ltoa_data_single = 10 * l0_data.iloc[:, m].values + a / b
        return ltoa_data_single

    n_rows = air_cols * ref_cols
    n_cols = air_rows
    ## 将test_set保存h5文件
    with h5py.File(os.path.join(i_path, f"ltoa_train_{sr}.h5"), 'w') as file:
        dset = file.create_dataset(f"ltoa_train_{sr}", (n_rows, n_cols), dtype=np.float64)
        i = 0
        for m in tqdm(range(air_cols), desc="计算训练集-仿真星上辐亮度数据集"):
            for n in range(ref_cols):
                # 执行计算
                result = compute_single(m, n)
                dset[i, :] = np.array(result).astype(np.float64)
                i += 1
    with h5py.File(os.path.join(i_path, f"ltoa_train_{sr}.h5"), 'r') as f:
        # 打印文件结构
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")

        f.visititems(print_structure)


# 计算测试集
def test_set(output_dir, l0_data, s_data, ltoc_data, t_data, r_scope, sif_scope, sr):
    air_rows, air_cols = l0_data.shape  # 4608
    sif_rows, sif_cols = sif_scope.shape  # 60

    # 函数：计算并保存每次的结果
    def compute_single(m, n):
        a = t_data.iloc[:, m].values * (
                10 * ltoc_data.iloc[:, m].values * r_scope.iloc[:, n].values + sif_scope.iloc[:, n].values)
        # T*(Ltoc*ρs+SIF) Ltoc是由MODTRAN模拟的数据，单位并非SIF的单位
        b = 1 - (s_data.iloc[:, m].values * r_scope.iloc[:, n].values)
        ltoa_data_single = 10 * l0_data.iloc[:, m].values + a / b
        return ltoa_data_single

    n_rows = air_cols * sif_cols
    n_cols = air_rows

    ## 将test_set保存h5文件
    with h5py.File(os.path.join(i_path, f"ltoa_test_{sr}.h5"), 'w') as file:
        dset = file.create_dataset(f"ltoa_test_{sr}", (n_rows, n_cols), dtype=np.float64)
        i = 0
        for m in tqdm(range(air_cols), desc="计算测试集-仿真星上辐亮度数据集"):
            for n in range(sif_cols):
                # 执行计算
                result = compute_single(m, n)
                dset[i, :] = np.array(result).astype(np.float64)
                i += 1
    with h5py.File(os.path.join(i_path, f"ltoa_test_{sr}.h5"), 'r') as f:
        # 打印文件结构
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name} (Shape: {obj.shape}, Dtype: {obj.dtype})")

        f.visititems(print_structure)


# 算法部分
# 计算行星反射率
"""
计算行星反射率 # rp = ltoa/（Isol*μ0/pi）
其中Isol有两个来源可供选择 1.MODTRAN输出文件中的SOL@OBS 2.sao2010：https://lweb.cfa.harvard.edu/atmosphere/
                      根据对MODTRAN的SLO@OBS数据和sao2010数据分析，选用MODTRAN数据！
                      通过SFWHM控制太阳辐照度的分辨率！！！！
这里的μ0是太阳天顶角的余弦值
MODTRAN中模拟时，数据对应的模拟输入值保存在combs.csv中(太阳天顶角的值)
"""


def reflectance_planet(i_path, mod_comb_path, sr):
    # rp = ltoa/（Isol*μ0/pi）
    # (1) 计算μ0
    print("1、计算μ0")
    comb = pd.read_csv(mod_comb_path, index_col=0)
    miu_training = []  # c_training
    miu_test = []
    cos0 = np.cos(np.radians(comb.iloc[:, 0]))  # 太阳天顶角的余弦值（4608组）
    for m in range(len(cos0)):
        for n in range(12):
            miu_training.append(cos0[m])
    for m in range(len(cos0)):
        for n in range(60):
            miu_test.append(cos0[m])

    def sec(a):
        return 1 / np.cos(a)

    """
    求解 sec(a)/(sec(a)+sec(b))
    """
    angle_s = 180 - comb.iloc[:, 1]  # 这里是观测天顶角
    sec0 = sec(np.radians(comb.iloc[:, 0]))
    secv = sec(np.radians(angle_s))
    result = secv / (secv + sec0)
    result = pd.DataFrame(result)
    result.to_csv(os.path.join(i_path, f'sec_func_{sr}.csv'), index=False, header=None)

    # (2) 计算Rp
    print("2、计算Rp")
    # 行星反射率 Rp = Ltoa/（Isol*μ0/pi）
    """
    1、Ltoa 读取由training_set()/test_set()得到的仿真数据 e.g.:ltoa_data_training_{sr}.csv
    2、Isol 数据由ARTelements_calcilate.py从MODTRAN模拟结果中保存得到的 SOLAR.csv
    """

    def rp_function(a, b, c):
        """
        行星反射率 = Ltoa/（Isol*μ0/pi）
        :param a: Ltoa
        :param b: Isol (这里的太阳辐照度是MODTRAN模拟的数据单位是 μW/cm2/nm,转变成标准单位 10 mW/m2/nm)
        :param c: μ0
        :return: 行星反射率 Rp
        """
        return a / (10 * b * c / np.pi)

    def sun_toa(b, c):
        return 10 * b * c / np.pi

    """
    with h5py.File(os.path.join(i_path, f"ltoa_data_training_{sr}.h5"), 'r') as f:
        ltoa = f[f"ltoa_data_training_{sr}"]
    with h5py.File(os.path.join(i_path, f"ltoa_data_test_{sr}.h5"), 'r') as f:
        ltoa_test = f[f"ltoa_data_test_{sr}"]

    后续计算过程中，需要保证ltoa和ltoa_test都能读取
    也就是需要保持多个句柄都同时打开 需要同时管理多个上下文管理器（如多个文件句柄） 使用 contextlib.ExitStack
    """

    train_h5 = os.path.join(i_path, f"ltoa_train_{sr}.h5")
    test_h5 = os.path.join(i_path, f"ltoa_test_{sr}.h5")
    solar = pd.read_csv(os.path.join(i_path, 'SOLAR.csv'), header=None, memory_map=True)
    with ExitStack() as stack:
        # 读取文件
        f_train = stack.enter_context(h5py.File(train_h5, 'r'))
        f_test = stack.enter_context(h5py.File(test_h5, 'r'))
        ltoa = f_train[f"ltoa_train_{sr}"]
        ltoa_test = f_test[f"ltoa_test_{sr}"]

        # 确定文件的shape，方便创建空set存结果（训练集）
        sample = rp_function(ltoa[0, :], solar.iloc[:, 0], miu_training[0])
        n_rows = ltoa.shape[0]
        n_cols = len(sample)

        # 创建存储h5文件
        # 计算保存Rp_train
        f_Rp_train = stack.enter_context(h5py.File(os.path.join(i_path, f'Rp_train_{sr}.h5'), 'w'))
        dataset = f_Rp_train.create_dataset(f'Rp_train_{sr}', (n_rows, n_cols), dtype=np.float64)
        for i in tqdm(range(n_rows), desc='计算Rp_train'):
            result = rp_function(ltoa[i, :], solar.iloc[:, 0], miu_training[i])
            dataset[i, :] = np.array(result).astype(np.float64)

        # 计算保存sun_toa_train
        f_sun_train = stack.enter_context(h5py.File(os.path.join(i_path, f'sun_toa_train_{sr}.h5'), 'w'))
        dataset = f_sun_train.create_dataset(f'sun_toa_train_{sr}', (n_rows, n_cols), dtype=np.float64)
        for i in tqdm(range(n_rows), desc='计算sun_toa_train'):
            result = sun_toa(solar.iloc[:, 0], miu_training[i])
            dataset[i, :] = np.array(result).astype(np.float64)

        # 确定文件的shape，方便创建空set存结果（测试集）
        sample = rp_function(ltoa_test[0, :], solar.iloc[:, 0], miu_test[0])
        n_rows = ltoa_test.shape[0]
        n_cols = len(sample)

        # 创建存储h5文件
        # 计算保存Rp_test
        f_Rp_test = stack.enter_context(h5py.File(os.path.join(i_path, f'Rp_test_{sr}.h5'), 'w'))
        dataset = f_Rp_test.create_dataset(f'Rp_test_{sr}', (n_rows, n_cols), dtype=np.float64)
        for i in tqdm(range(n_rows), desc='计算Rp_test'):
            result = rp_function(ltoa_test[i, :], solar.iloc[:, 0], miu_test[i])
            dataset[i, :] = np.array(result).astype(np.float64)

        # 计算保存sun_toa_test
        f_sun_test = stack.enter_context(h5py.File(os.path.join(i_path, f'sun_toa_test_{sr}.h5'), 'w'))
        dataset = f_sun_test.create_dataset(f'sun_toa_test_{sr}', (n_rows, n_cols), dtype=np.float64)
        for i in tqdm(range(n_rows), desc='计算sun_toa_test'):
            result = sun_toa(solar.iloc[:, 0], miu_test[i])
            dataset[i, :] = np.array(result).astype(np.float64)

    """
    删除 变量ltoa和solar释放内存
    """
    del ltoa
    del ltoa_test
    del solar
    del result
    del sample
    del n_rows
    del n_cols

    # （3）计算有效上下行透过率
    print("3、计算有效上下行透过率")
    # T↑↓e = Rp/R
    """
    这里严格意义上 T↑↓e = Rp/f(λ) 并且是在选定的拟合波段范围内进行计算
    先进行640-850nm全波段的计算 R就使用read_paras()函数中插值到目标SR的反射率
    """
    r_usgs = pd.read_csv(os.path.join(i_path, f"r_usgs_{sr}.csv"), header=None, memory_map=True)
    r_scope = pd.read_csv(os.path.join(i_path, f"r_scope_{sr}.csv"), header=None, memory_map=True)
    print("3.1 计算有效上下行透过率")
    with ExitStack() as stack:
        """计算t_us_train"""
        # 读取Rp_train
        f_rp = stack.enter_context(h5py.File(os.path.join(i_path, f"Rp_train_{sr}.h5"), 'r'))
        rp_train = f_rp[f'Rp_train_{sr}']

        # 创建h5保存t_us
        f_t_us = stack.enter_context(h5py.File(os.path.join(i_path, f"t_us_train_{sr}.h5"), 'w'))
        dataset = f_t_us.create_dataset(f"t_us_train_{sr}", (rp_train.shape[0], rp_train.shape[1]), dtype=np.float64)

        # 计算t_us
        r_usgs = np.array(r_usgs.T)
        num = int(rp_train.shape[0] / r_usgs.shape[0])
        size = r_usgs.shape[0]
        for i in tqdm(range(num), desc='计算t_us中（训练集）'):
            result = rp_train[i * size:(i + 1) * size, :] / r_usgs
            dataset[i * size:(i + 1) * size, :] = np.array(result).astype(np.float64)

        """计算t_us_test"""
        # 读取Rp_test
        f_rp_test = stack.enter_context(h5py.File(os.path.join(i_path, f"Rp_test_{sr}.h5"), 'r'))
        rp_test = f_rp_test[f'Rp_test_{sr}']

        # 创建h5保存t_us
        f_t_us = stack.enter_context(h5py.File(os.path.join(i_path, f"t_us_test_{sr}.h5"), 'w'))
        dataset = f_t_us.create_dataset(f"t_us_test_{sr}", (rp_test.shape[0], rp_test.shape[1]), dtype=np.float64)

        # 计算t_us
        r_scope = np.array(r_scope.T)
        num = int(rp_test.shape[0] / r_scope.shape[0])
        size = r_scope.shape[0]
        for i in tqdm(range(num), desc='计算t_us中（测试集）'):
            result = rp_test[i * size:(i + 1) * size, :] / r_scope
            dataset[i * size:(i + 1) * size, :] = np.array(result).astype(np.float64)

    """
    已求得测试集的有效上下行透过率，便可以计算测试集的有效上行透过率
    T_up = exp[ln(T_us)*sec_func]
    """
    sec = np.array(pd.read_csv(os.path.join(i_path, f'sec_func_{sr}.csv'), header=None, memory_map=True).iloc[:, 0])

    def t_up_func(t_us, sec):
        df = np.log(t_us)
        df = df * sec
        df = np.exp(df)
        return df

    print("3.2 计算有效上行透过率")
    with ExitStack() as stack:
        # 读取测试集t_us
        f_t_us_test = stack.enter_context(h5py.File(os.path.join(i_path, f't_us_test_{sr}.h5'), 'r'))
        t_us_test = f_t_us_test[f't_us_test_{sr}']

        num = int(t_us_test.shape[0] / len(sec))

        # 创建h5，储存有效上行透过率
        f_t_up_test = stack.enter_context(h5py.File(os.path.join(i_path, f't_up_test_{sr}.h5'), 'w'))
        t_up_test = f_t_up_test.create_dataset(f't_up_test_{sr}', (t_us_test.shape[0], t_us_test.shape[1]),
                                               dtype=np.float64)

        for i in tqdm(range(len(sec)), desc='计算有效上行透过率中（测试集）'):
            cell = t_us_test[num * i:num * (i + 1), :]
            result = t_up_func(cell, sec[i])
            t_up_test[num * i:num * (i + 1), :] = np.array(result).astype(np.float64)


"""
主成分分析
"""


def pca(i_path, sr, wave):
    # 读取训练集上下行透过率
    with ExitStack() as stack:
        f_t_us_train = stack.enter_context(h5py.File(os.path.join(i_path, f't_us_train_{sr}.h5'), 'r'))
        t_us_train = f_t_us_train[f't_us_train_{sr}']

        # 保存前20条数据 用于对比（经过对比，h5计算的数据与csv数据并无差异）
        t_us_train_head = pd.DataFrame(t_us_train[0:20, :])
        t_us_train_head.to_csv(os.path.join(i_path, f't_us_train_head.csv'), index=False, header=None)

        t_us_a = t_us_train[:, a_s:a_e]
        pd.DataFrame(t_us_a[0:20, :]).to_csv(os.path.join(i_path, f't_us_train_a.csv'), index=False, header=None)

        t_us_b = t_us_train[:, b_s:b_e]
        pd.DataFrame(t_us_b[0:20, :]).to_csv(os.path.join(i_path, f't_us_train_b.csv'), index=False, header=None)

        """保存有效上下行透过率和有效上行透过率(前20条，部分波段对比数据)"""
        # # 读取保存有效上下行透过率(测试集)
        # f_t_us_test = stack.enter_context(h5py.File(os.path.join(i_path, f't_us_test_{sr}.h5'), 'r'))
        # t_us_test = f_t_us_test[f't_us_test_{sr}']
        #
        # t_us_test_head_a = pd.DataFrame(t_us_test[0:20,:][:, a_s:a_e])
        # t_us_test_head_a.to_csv(os.path.join(i_path, f't_us_test_a_head.csv'), index=False, header=None)
        #
        # t_us_test_head_b = pd.DataFrame(t_us_test[0:20, :][:, b_s:b_e])
        # t_us_test_head_b.to_csv(os.path.join(i_path, f't_us_test_b_head.csv'), index=False, header=None)
        #
        # # 读取保存有效上行透过率(测试集)
        # f_t_up = stack.enter_context(h5py.File(os.path.join(i_path, f't_up_test_{sr}.h5'), 'r'))
        # t_up = f_t_up[f't_up_test_{sr}']
        #
        # t_up_test_head_a = pd.DataFrame(t_up[0:20,:][:, a_s:a_e])
        # t_up_test_head_a.to_csv(os.path.join(i_path, f't_up_test_a_head.csv'), index=False, header=None)
        #
        # t_up_test_head_b = pd.DataFrame(t_up[0:20,:][:, b_s:b_e])
        # t_up_test_head_b.to_csv(os.path.join(i_path, f't_up_test_b_head.csv'), index=False, header=None)

        """开始主成分分析"""
        # 开始主成分分析
        # （1）标准化
        scaler_a = StandardScaler()
        t_us_a_scaled = scaler_a.fit_transform(t_us_a)
        mean = scaler_a.mean_
        std = scaler_a.scale_
        print("hi")
        # 保存标准化参数和标准化数据
        joblib.dump(scaler_a, os.path.join(i_path, f'scaler_a.pkl'))
        pd.DataFrame(t_us_a_scaled).to_csv(os.path.join(i_path, f't_us_a_scaled.csv'), index=False, header=None)

        # （2）主成分分析
        """标准化数据PCA"""
        pca_a = PCA(n_components=10)
        t_us_pca = pca_a.fit_transform(t_us_a_scaled)
        pd.DataFrame(t_us_pca).to_csv(os.path.join(i_path, f't_us_pca.csv'), index=False, header=None)
        # 保存主成分
        pd.DataFrame(pca_a.components_).to_csv(os.path.join(i_path, f'component_scaled.csv'), index=False, header=None)
        print("\n已经标准化")
        pca_summary_a = pd.DataFrame({"标准差": np.sqrt(pca_a.explained_variance_),
                                      "贡献率": pca_a.explained_variance_ratio_,
                                      "累计贡献率": np.cumsum(pca_a.explained_variance_ratio_)})
        pca_summary_a = pca_summary_a.transpose()
        pca_summary_a.columns = [f'PC{i}' for i in range(1, len(pca_summary_a.columns) + 1)]
        print(pca_summary_a.iloc[:, 0:5].round(5))

        """非标准化数据PCA"""
        pca_a_unscaled = PCA(n_components=10)
        t_us_pca_unscaled = pca_a_unscaled.fit_transform(t_us_a)
        pd.DataFrame(t_us_pca_unscaled).to_csv(os.path.join(i_path, f't_us_pca_unscaled.csv'), index=False, header=None)
        # 保存主成分
        pd.DataFrame(pca_a_unscaled.components_).to_csv(os.path.join(i_path, f'component_unscaled.csv'), index=False,
                                                        header=None)
        print("\n未标准化")
        pca_summary_a_unscaled = pd.DataFrame({"标准差": np.sqrt(pca_a_unscaled.explained_variance_),
                                               "贡献率": pca_a_unscaled.explained_variance_ratio_,
                                               "累计贡献率": np.cumsum(pca_a_unscaled.explained_variance_ratio_)})
        pca_summary_a_unscaled = pca_summary_a_unscaled.transpose()
        pca_summary_a_unscaled.columns = [f'PC{i}' for i in range(1, len(pca_summary_a.columns) + 1)]
        print(pca_summary_a_unscaled.iloc[:, 0:5].round(5))


"""
反演算法
"""
def algorithm(i_path, sr, lambda_wl_a):
    """
    基于PCA的反演算法
    :return:
    """
    print('训练集算法测试:(1) 不含SIF的算法')

    # 开始计算不包含荧光的前向模型
    # (1) 不含SIF的算法
    """
    求解思路： Ltoa=SUN*R*T_us
    对于线性化的前向模型，R不作拟合，未知项为高频变化项（有效上下行透过率）主成分向量的系数
    需要用到的数据：ltoa_data_train， R(usgs)_{sr}，PCA_components, sun_toa_train, scaler_a.pkl
    """
    with ExitStack() as stack:
        """Ltoa"""
        f_ltoa_train = stack.enter_context(h5py.File(os.path.join(i_path, f'ltoa_train_{sr}.h5'), 'r'))
        ltoa_train = f_ltoa_train[f'ltoa_train_{sr}'][:, a_s:a_e]

        """Sun"""
        f_sun_toa_train = stack.enter_context(h5py.File(os.path.join(i_path, f'sun_toa_train_{sr}.h5'), 'r'))
        sun_toa_train = f_sun_toa_train[f'sun_toa_train_{sr}'][:, a_s:a_e]

        """PC"""
        component_a = np.array(pd.read_csv(os.path.join(i_path, f'component_unscaled.csv'), header=None, memory_map=True))

        """R"""
        r_usgs = np.array(pd.read_csv(os.path.join(i_path, f'r_usgs_{sr}.csv'), header=None, memory_map=True)).T[:, a_s:a_e]

        """T_us(用于检验重建效果)"""
        f_t_us = stack.enter_context(h5py.File(os.path.join(i_path, f't_us_train_{sr}.h5'), 'r'))
        t_us = f_t_us[f't_us_train_{sr}'][:, a_s:a_e]

        """利用最小二乘算法解算主成分向量权重系数"""
        # (1) 决定使用主成分分量的数量（根据贡献率决定）
        n_pc = 5
        component_a_algorithm = component_a[0:n_pc,:].T
        cycle_n = ltoa_train.shape[0]

        """核心算法过程"""
        data_n = ltoa_train.shape[1]  # 数据点数

        def residuals(beta, ltoa_obs, sun, r, components, data_n, mean, std):
            """定义的residuals函数仅仅作为least_squares函数中的残差函数，不能返回EM, ltoa_obs[i], ltoa_predicted"""
            residual = np.zeros(data_n)  # 初始化残差向量
            # 计算模型预测值
            for i in range(data_n):
                """Ltoa=SUN*R*T_us"""
                # # 反标准化
                # t_us = sum(beta[j] * components[i, j] for j in range(n_pc))*std[i] + mean[i]
                # 无需反标准化
                t_us = sum(beta[j] * components[i, j] for j in range(n_pc))
                ltoa_predicted = sun[i] * r[i] * t_us
                residual[i] = ltoa_predicted - ltoa_obs[i]
            return residual

        def t_us_residual(beta, components, t_us):
            for i in range(data_n):
                t_us_re = sum(beta[j] * components[i, j] for j in range(n_pc))
            t_us_residual = t_us - t_us_re
            return t_us_residual
        # 读取标准化参数
        scaler_para = joblib.load(os.path.join(i_path, f'scaler_a.pkl'))
        mean = scaler_para.mean_
        std = scaler_para.scale_

        f_lst = stack.enter_context(h5py.File(os.path.join(i_path, f"lst_{sr}.h5"), 'w'))
        lst_cofs = f_lst.create_dataset(f"lst_cofs_{sr}", (cycle_n, n_pc), dtype=np.float64)
        lst_residual = f_lst.create_dataset(f"lst_residual_{sr}", (cycle_n, data_n), dtype=np.float64)

        f_t_us_residuals = stack.enter_context(h5py.File(os.path.join(i_path, f"t_us_residuals_{sr}.h5"), 'w'))
        t_us_residuals = f_t_us_residuals.create_dataset(f"t_us_residuals_{sr}", (cycle_n, data_n), dtype=np.float64)
        ems = f_t_us_residuals.create_dataset(f"t_us_EMs_{sr}", (cycle_n, 3), dtype=np.float64)

        # # 在循环前初始化交互模式和图对象
        # plt.ion()
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_title('LTOA Reconstruction Comparison')
        # ax.set_xlabel('Band Index')
        # ax.set_ylabel('Value')
        # line1, = ax.plot([], [], 'b-', label='Original LTOA')
        # line2, = ax.plot([], [], 'r--', label='Reconstructed LTOA')
        # metrics_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=8)
        # ax.legend()
        # ax.grid(True)

        for i in tqdm(range(cycle_n), desc="计算拟合系数中"):
            # 初始化猜测β系数
            beta = np.ones(n_pc)
            ltoa_obs = ltoa_train[i,:]
            sun = sun_toa_train[i,:]

            i_r = math.floor(float(i / 4608))
            r = r_usgs[i_r,:]

            res = least_squares(residuals, beta,
                args=(ltoa_obs, sun, r, component_a_algorithm, data_n, mean, std))
            lst_cofs[i,:] = res.x
            lst_residual[i,:] = res.fun

            # 使用最优beta重新计算预测值
            ltoa_predicted_all = []
            for j in range(data_n):
                t_us = sum(res.x[k] * component_a_algorithm[j, k] for k in range(n_pc))
                ltoa_predicted = sun[j] * r[j] * t_us
                ltoa_predicted_all.append(ltoa_predicted)
            ltoa_predicted_all = np.array(ltoa_predicted_all)
            t_us_residuals[i,:] = ltoa_predicted_all - ltoa_obs

            mse = mean_squared_error(ltoa_obs, ltoa_predicted_all)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(ltoa_obs, ltoa_predicted_all)
            r2 = r2_score(ltoa_obs, ltoa_predicted_all)
            EM = [rmse, mae, r2]
            ems[i,:] = EM

        #     # 更新绘图数据
        #     x = np.arange(len(ltoa_obs))
        #     line1.set_data(x, ltoa_obs)
        #     line2.set_data(x, ltoa_predicted_all)
        #
        #     # 调整坐标轴范围
        #     ax.set_xlim(0, len(ltoa_obs))
        #     ax.set_ylim(min(np.min(ltoa_obs), np.min(ltoa_predicted_all)) - 0.1,
        #                 max(np.max(ltoa_obs), np.max(ltoa_predicted_all)) + 0.1)
        #
        #     # 更新指标文本
        #     metrics_text.set_text(
        #         f'this is Ltoa-{i} \nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
        #     )
        #
        #     # 刷新图像
        #     plt.draw()
        #     plt.pause(0.01)  # 适当调整暂停时间
        #
        # # 循环结束后关闭交互模式
        # plt.ioff()
        # plt.show()

    # print('训练集算法测试:(2) 含SIF的算法')
    # # 开始计算包含荧光的前向模型
    # # (2) 不含SIF的算法
    # print("2.1 定义Fs分布函数（高斯函数）")
    # """定义Fs分布的高斯函数"""
    #
    # def gaussian_function(lambd, lambda_0, sigma_h):
    #     return np.exp(-(lambd - lambda_0) ** 2 / (2 * sigma_h ** 2))
    #
    # # 根据反演窗口确定高斯函数的参数值 以下针对远红光波段
    # lambda_0_a = 740  # 设置 λ_0 的值
    # sigma_h_a = 21  # 设置 σ_h 的值
    # lamda = lambda_wl_a  # 这里是反演窗口
    # hf = gaussian_function(lamda, lambda_0_a, sigma_h_a)
    #
    # print("2.2 反演SIF")
    # """
    #     求解思路： Ltoa=SUN*R*T_us+SIF*T_up*hf
    #     对于线性化的前向模型，R不作拟合，未知项为高频变化项（有效上下行透过率）主成分向量的系数
    #     需要用到的数据：ltoa_data_test， R(scope)_{sr}，PCA_components, sun_toa_test, scaler_a.pkl, hf
    # """
    # with ExitStack() as stack:
    #     f_ltoa_test = stack.enter_context(h5py.File(os.path.join(i_path, f"ltoa_test_{sr}.h5"), 'r'))
    #     ltoa_test = f_ltoa_test[f"ltoa_test_{sr}"][:, a_s:a_e]
    #
    #     f_sun_toa_test = stack.enter_context(h5py.File(os.path.join(i_path, f"sun_toa_tets_{sr}.h5"), 'r'))
    #     sun_toa_test = f_sun_toa_test[f"sun_toa_test_{sr}"][:, a_s:a_e]
    #
    #     component_a = np.array(pd.read_csv(os.path.join(i_path, f'component_scaled.csv'), header=None, memory_map=True))
    #     r_scope = np.array(pd.read_csv(os.path.join(i_path, f'r_scope_{sr}.csv'), header=None, memory_map=True)).T[:,
    #               a_s:a_e]
    #
    #     f_t_up_test = stack.enter_context(h5py.File(os.path.join(i_path, f"t_up_test_{sr}.h5"), 'r'))
    #     t_up_test = f_t_up_test[f"t_up_test_{sr}"][:, a_s:a_e]
    #
    #     """利用最小二乘算法解算主成分向量权重系数"""
    #     # (1) 决定使用主成分分量的数量（根据贡献率决定）
    #     n_pc = 5
    #     component_a_algorithm = component_a[0:n_pc, :].T
    #     cycle_n = ltoa_test.shape[0]
    #
    #     """核心算法过程"""
    #     data_n = ltoa_test.shape[1]  # 数据点数
    #
    #     def residuals_test(beta, ltoa_obs, sun, r, components, data_n, mean, std, hf, t_up):
    #         residual = np.zeros(data_n)  # 初始化残差向量
    #         # 计算模型预测值
    #         for i in range(data_n):
    #             """Ltoa=SUN*R*T_us"""
    #             # 反标准化
    #             t_us = sum(beta[j] * components[i, j] for j in range(n_pc)) * std[i] + mean[i]
    #             ltoa_predicted = sun[i] * r[i] * t_us + beta[-1] * hf[i] * t_up[i]
    #             residual[i] = ltoa_predicted - ltoa_obs[i]
    #         # print(residual)
    #         return residual
    #
    #     # 读取标准化参数
    #     scaler_para = joblib.load(os.path.join(i_path, f'scaler_a.pkl'))
    #     mean = scaler_para.mean_
    #     std = scaler_para.scale_
    #     lst_cofs = []
    #     lst_residual = []
    #     sif = []
    #     for i in tqdm(range(cycle_n), desc="计算拟合系数中"):
    #         # 初始化猜测β系数
    #         beta = np.ones(n_pc + 1)
    #         ltoa_obs = ltoa_test[i, :]
    #         sun = sun_toa_test[i, :]
    #
    #         i_r = math.floor(float(i / 4608))
    #         r = r_scope[i_r, :]
    #
    #         t_up = t_up_test[i, :]
    #         res = least_squares(residuals_test, beta,
    #                             args=(ltoa_obs, sun, r, component_a_algorithm, data_n, mean, std, hf, t_up))
    #         lst_cofs.append(res.x)
    #         lst_residual.append(res.fun)
    #         sif.append(res.x[-1] * hf)
    #
    #     print("hi")
    #
    # items = [f'ltoa_data_training_{sr}.csv', f'ltoa_data_test_{sr}.csv', f'sun_toa_training_{sr}.csv']
    # pkl_items = ['scaler_a.pkl', 'scaler_b.pkl']
    # components_items = [f'components_a_{sr}.csv', f'components_b_{sr}.csv']
    # components_items_without_scaled = [f'components_a_{sr}_without_scaled.csv', f'components_b_{sr}_without_scaled.csv']
    # r_usgs = [f'r_usgs_{sr}.csv']


if __name__ == '__main__':
    #  这里需要输入计算好的大气参数（L0,S,Ltoc）所在路径
    i_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine"
    #  以及SCOPE模拟的无荧光贡献的反射率曲线,SIF曲线
    r_scope_path = "F:\\research\\Processing_program\\SCOPE_dataprocessing\\SCOPE_2.1\\SCOPE-2.1\\output\\MODTRAN_simulation_2024-12-30-1732\\reflectance.csv"
    sif_path = "F:\\research\\Processing_program\\SCOPE_dataprocessing\\SCOPE_2.1\\SCOPE-2.1\\output\\MODTRAN_simulation_2024-12-30-1732\\fluorescence.csv"
    # 以及USGS 光谱库的非荧光地表反射率光谱路径
    r_usgs_path = "F:\\research\\Processing_program\\MODTRAN_dataset\\USGS\\selected\\sif_wl.csv"
    # MODTRAN输入参数组合路径
    mod_comb_path = "F:\\research\\Processing_program\\MODTRAN_dataset\MOD\\640_850_0.05_fine\\combs.csv"
    # 本次模拟数据的光谱分辨率
    sr = 0.05

    # (1) 读取数据
    # print("(1) 读取数据")
    # l0_data, s_data, ltoc_data, t_data, sif_scope, r_scope, r_usgs \
    #     = read_paras(i_path, r_scope_path, sif_path, r_usgs_path, sr)
    # # # 测试MODTRAN和SCOPE数据耦合效果
    # # dataset_test(i_path, ltoc_data, r_scope)
    # # (2) 构建训练集
    # print("(2) 构建训练集")
    # ltoa_data_training = train_set(i_path, l0_data, s_data, ltoc_data, t_data, r_usgs, sr)
    # print("-----训练集已保存-----")
    # # # (2) 构建测试集
    # print("(2) 构建测试集")
    # ltoa_data_test = test_set(i_path, l0_data, s_data, ltoc_data, t_data, r_scope, sif_scope, sr)
    # print("-----测试集已保存-----")

    # (3) 计算行星反射率和有效上下行透过率
    """
    涉及到行星反射率的计算（利用太阳光谱对星上辐亮度光谱进行归一化得到归一化的反射率）  太阳光谱的选取：1.MODTRAN的sol@obs 2. https://lweb.cfa.harvard.edu/atmosphere/
    根据分析 数据应该选用MODTRAN输出的sol@obs
    行星反射率 Rp = Ltoa/（Isol*μ0/pi）
    这里的μ0是太阳天顶角的余弦值
    MODTRAN中模拟时，数据对应的模拟输入值保存在combs.csv中(太阳天顶角的值)
    """
    # print("(3) 计算行星反射率和有效上下行透过率")
    # test = reflectance_planet(i_path,mod_comb_path,sr)
    # print("-----行星反射率和有效上下行透过率已保存-----")

    # (4) 对有效上下行透过率进行主成分分析
    """
    筛选拟合窗口 已知所有数据的波段范围均在640-850nm范围，SR取决于拟合的输入值
    现有拟合窗口，需要筛选出窗口对应的数据范围
    窗口1：O2-B,735-758nm
    窗口2：O2-A,682-692nm
    """
    wl_A = [735, 758]
    wl_B = [682, 692]
    """
    sr=0.3nm，使用下面的wl_A,wl_B 
    """
    # wl_A = [735.1, 758.2]
    # wl_B = [682, 692.2]

    # # 定义反射率拟合窗口（及分辨率）
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

    # print("(4) 对有效上下行透过率进行主成分分析")
    # test = pca(i_path, sr, wave)
    # print("-----主成分分析结果已保存-----")

    # (5) 基于PCA的反演算法
    """
    反演算法
    """
    print("(5) 基于PCA的反演算法")
    test = algorithm(i_path, sr, lambda_wl_a)