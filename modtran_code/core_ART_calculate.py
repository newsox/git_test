# -*- coding: utf-8 -*-
# @Time    : 2025/2/27 15:55
# @Author  : xu xinwen
# @FileName: core_ART_calculate.py
# @Software: PyCharm

import os
import pandas as pd
import itertools
import numpy as np
from contextlib import ExitStack
import h5py
from tqdm import tqdm

def elements_valculate(dir_path, combs, sr):

    file_name = []
    root_name = ['R0', 'R05', 'R1']
    # root_name = ['simulation1', 'simulation2', 'simulation3']
    for r in root_name:
        file_name.append(os.path.join(dir_path, r))
    root_dir = ['TOTAL_tot_trans.csv', 'TOTAL_pth_thrml.csv',
                'TOTAL_surf_emis.csv', 'TOTAL_sol_scat.csv',
                'TOTAL_grnd_rflt.csv', 'TOTAL_drct_rflt.csv', 'TOTAL_sol_obs.csv', 'TOTAL_total_rad.csv', 'TOTAL_ref_sol.csv']
    # 0TOT_TRANS 1PTEM 2SFEM 3PATH 4GRFL 5GSUN 6SOLOBS 7TOTALRAD
    path_comb = list(itertools.product(file_name, root_dir))

    # path_solar_obs = os.path.join(dir_path, root_name[0], root_dir[6])
    # solar = pd.read_csv(path_solar_obs, header=None).iloc[:,1]
    # solar.to_csv(os.path.join(dir_path, 'SOLAR.csv'), index=False, header=None)

    r0_path = []
    r05_path = []
    r1_path = []
    for i in range(0, len(root_dir)):
        r0_path.append(os.path.join(path_comb[i][0], path_comb[i][1]))
        r05_path.append(os.path.join(path_comb[i+9][0], path_comb[i+9][1]))
        r1_path.append(os.path.join(path_comb[i+18][0], path_comb[i+18][1]))

    print("开始计算辐射传输参数（反射率为0.5,1版）")

    print(combs)
    with ExitStack() as stack:
        """准备项: E*COSθ/π """
        Esun = []
        #  读取E: SOL@OBS
        E = np.array(pd.read_csv(r1_path[6], header=None))
        COSθ = np.cos(np.radians(combs.iloc[:, 0]))
        for i in tqdm(range(E.shape[1]), desc="计算Esun中"):
            Esun.append(E[:,i]*COSθ[i]/np.pi)
        Esun = pd.DataFrame(Esun).T
        """开始计算辐射传输参数"""
        GRT1 = pd.read_csv(r1_path[4], header=None)
        GRT05 = pd.read_csv(r05_path[4], header=None)

        # (1) 半球反照率S
        S = (GRT1-2*GRT05)/(GRT1-GRT05)
        f_s_h5 = stack.enter_context(h5py.File(os.path.join(dir_path, f'S_{sr}.h5'), 'w'))
        s_h5 = f_s_h5.create_dataset(f'S_{sr}', (S.shape[0],S.shape[1]), dtype=np.float64 )
        s_h5[:,:] = np.array(S)

        # (2) 上行透过率  T=τoo+τdo
        τoo = pd.read_csv(r1_path[0], header=None)

        PATH1 = pd.read_csv(r1_path[3], header=None)
        PATH05 = pd.read_csv(r05_path[3], header=None)
        τdo = ((PATH1-PATH05)/(GRT1-GRT05))*(τoo)

        T = τoo+τdo
        f_t_h5 = stack.enter_context(h5py.File(os.path.join(dir_path, f'T_{sr}.h5'), 'w'))
        t_h5 = f_t_h5.create_dataset(f'T_{sr}', (T.shape[0], T.shape[1]), dtype=np.float64)
        t_h5[:,:] = np.array(T)

        # (3) 大气程辐射 L0
        L0 = PATH1-((GRT1*τdo)/τoo)

        f_l0_h5 = stack.enter_context(h5py.File(os.path.join(dir_path, f'L0_{sr}.h5'), 'w'))
        l0_h5 =  f_l0_h5.create_dataset(f'L0_{sr}', (L0.shape[0], L0.shape[1]), dtype=np.float64)
        l0_h5[:,:] = np.array(L0)


        # (4) 冠层入射辐射 Ltoc
        GSUN1 = pd.read_csv(r1_path[5], header=None)

        τss = GSUN1*(1/Esun)*(1/τoo)
        τsd = (((GRT1*(1-S))/GSUN1)-1)*τss

        Ltoc = Esun*(τss+τsd)

        f_ltoc_h5 = stack.enter_context(h5py.File(os.path.join(dir_path, f'Ltoc_{sr}.h5'), 'w'))
        ltoc_h5 = f_ltoc_h5.create_dataset(f'Ltoc_{sr}', (Ltoc.shape[0], Ltoc.shape[1]), dtype=np.float64)
        ltoc_h5[:,:] = np.array(Ltoc)

        # （5）计算τssτoo （验证 finite spectral band effect, τssτoo?=τss*τoo）
        """读取REF_SOL:Eτssτoo"""
        ref_Sol = pd.read_csv(r1_path[8], header=None)
        """读取SOL@OBS:E"""
        sol_obs = pd.read_csv(r1_path[6], header=None)

        result = ref_Sol/(sol_obs*τss*τoo)

        result.to_csv(os.path.join(dir_path, 'finite_spectral_band_effect.csv'), index=False, header=None)

    print("hi")

if __name__ == '__main__':
    #  这里需要读取MODTRAN输入参数（paraname和paravalue）  输入文件格式见：https://github.com/newsox/MODTRAN
    dir_path = "F:\\research\\Processing_program\\MODTRAN_dataset\\MOD\\640_8500.05_fine"
    para_path = os.path.join(dir_path, 'para_data_default.xlsx')
    sr = 0.05

    para_data = pd.read_excel(para_path)
    para_name = para_data.columns.tolist()
    para_name_reverse = para_name[::-1]
    para_value = para_data.T.values
    para_value_reverse = para_value[::-1, ::]
    def func(list):
        newlist = [x for x in list if np.isnan(x) == False]
        return newlist
    para_value_list = []
    for i in range(0, len(para_name_reverse)):
        para_value_list.append(func(para_value_reverse[i]))
    combs = list(itertools.product(*para_value_list))
    combs = pd.DataFrame(columns=None, data=combs)

    elements_valculate(dir_path, combs, sr)