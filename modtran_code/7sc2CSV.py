# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 10:32
# @Author  : xu xinwen
# @FileName: 7sc2CSVs.py
# @Software: PyCharm

print("\ndatatxt2csvs\n")

# 以40001行为循环处理每组数据的代码（40001行取决于MODTRAN中这是的波段范围和步长）
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv
import gc

from datetime import datetime
def split_big_csv(file_path, loop_number, group_number, band_number):
    print("\n确认分组数据是否正确中，请稍等...")
    # 定义拆分后的小文件大小（行数）
    while True:
        try:
            dividend = loop_number
            divisor = group_number
            result = dividend / divisor
            if dividend % divisor == 0:
                print(f"\t结果：{dividend} 可以被 {divisor} 整除，结果为 {result}")
                break  # 输入合法且能整除，结束循环
            else:
                print(f"\t结果：{dividend} 不能被 {divisor} 整除，请重新输入。")
        except ValueError:
            print("\t结果：请输入有效的整数。")
        except ZeroDivisionError:
            print("\t结果：除数不能为零，请重新输入。")

    print("\n分割大型TXT中，请稍等...")
    # print(loop_number)  # 这里出错
    # print(group_number)
    # 设置拆分的行数
    chunk_size = int(loop_number / group_number) * int(band_number)  # 每个文件的行数
    print(f"\t按照输入参数，将大型TXT分割成{group_number}个smallCSV文件，每个smallCSV包括{int(loop_number / group_number)}组循环，"
          f"有{int(loop_number / group_number)}组×{band_number}行/组={chunk_size}行数据")
    batch_no = 1
    # 大型CSV所在文件夹
    file_folder = os.path.dirname(file_path)
    # 读取大型CSV文件
    total_size = os.path.getsize(file_path)
    # 新建一个列表存储smallCSV的路径
    smallcsv_path_list = []
    with tqdm(total=total_size, unit='B', unit_scale=True,  desc="Processing File") as pbar:
        cycle_number = 0
        # print(cycle_number)
        with pd.read_csv(file_path, chunksize=chunk_size, header=None) as reader:
            # 如果这里不加 header = None， chunk会把第一行读成标头，从而使得在每个smallCSV中会把data.txt的第一行当做标头，造成数据错位一行
            # print(reader)
            for chunk in reader:
                # print(chunk)
                cycle_number += 1
                # print(len(chunk))
                # 除最后一个chunk外，修剪其他chunk的行数（防止除了最后一个smallCSV，前面的smallCSV会多保存一行600.000的数据）
                if len(chunk) >= chunk_size:
                    # print('true')
                    chunk = chunk.iloc[:chunk_size]  # 保留前chunk_size行
                # print(batch_no)
                smallcsv_path = os.path.join(file_folder, f'smallCSV{batch_no}.csv')
                smallcsv_path_list.append(smallcsv_path)
                chunk.to_csv(smallcsv_path, header=None, index=False)
                batch_no += 1
                # Update the progress bar with the size of the processed chunk and the cycle number
                pbar.set_description(f"Spliting bigCSV - Cycle {cycle_number}/{group_number}, SmallCSV_path:{smallcsv_path}")
                pbar.update(chunk.memory_usage(deep=True).sum())
    return smallcsv_path_list  # 返回smallcsv的路径

def process_to_txts(file_path, smallcsv_loop_count, band_number, smallcsv_turn):

    # Determine the output directory from the input file path
    output_dir = os.path.dirname(file_path)

    # Calculate the total size in bytes of the file for progress bar
    total_size = os.path.getsize(file_path)
    # print(f'文件的字节大小为{total_size}')  # 获取文件的大小（字节）

    # Prepare CSV files to write data
    tot_trans_path = os.path.join(output_dir,  f'tot_path{smallcsv_turn}.csv') # TRANS
    drct_rflt_path = os.path.join(output_dir,  f'drct_rflt{smallcsv_turn}.csv')  # GRFL
    grnd_rflt_path = os.path.join(output_dir, f'grnd_rflt{smallcsv_turn}.csv')   # TOT
    path_pth_thrml_path = os.path.join(output_dir, f'path_pth_thrml{smallcsv_turn}.csv')  # PATH
    path_sol_scat_path = os.path.join(output_dir, f'path_sol_scat{smallcsv_turn}.csv')  # PATH
    surf_emis_path = os.path.join(output_dir, f'surf_emis{smallcsv_turn}.csv')  # SFEM
    sol_obs_path = os.path.join(output_dir, f'sol_obs{smallcsv_turn}.csv')  # SOL@OBS
    total_rad_path = os.path.join(output_dir, f'total_rad{smallcsv_turn}.csv')  # TOTAL_RAD
    ref_sol_path = os.path.join(output_dir, f'ref_sol{smallcsv_turn}.csv')  # REF_SOL


    loop_number = int(smallcsv_loop_count/band_number)
    # print(smallcsv_loop_count)
    # print(f'loop_number:{loop_number}')
    band_number = int(band_number)

    with (tqdm(total=total_size, unit='B', unit_scale=True, desc="Processing File") as pbar):
        cycle_number = 0
        # print(loop_number)
        # print(band_number)
        tot_trans_data = np.empty((loop_number, band_number), dtype=object) # TRANS
        drct_rflt_data = np.empty((loop_number, band_number), dtype=object)
        grnd_rflt_data = np.empty((loop_number, band_number), dtype=object)
        path_pth_thrml_data = np.empty((loop_number, band_number), dtype=object)
        path_sol_scat_data = np.empty((loop_number, band_number), dtype=object)
        surf_emis_data = np.empty((loop_number, band_number), dtype=object)
        sol_obs_data = np.empty((loop_number, band_number), dtype=object)
        total_rad_data = np.empty((loop_number, band_number), dtype=object)
        ref_sol_data = np.empty((loop_number, band_number), dtype=object)

        for chunk in pd.read_csv(file_path, chunksize=band_number, header=None, dtype=str, engine='python'):
            cycle_number += 1
            # print(cycle_number)
            # print(chunk)
            # 分割每一行，扩展成多列
            split_data = chunk[0].str.split(expand=True)
            """
            print(len(split_data))
            for i in range(len(split_data)):
                print(split_data[i])
            print("hi")
            # 根据 MODTRAN 的输出设置 PHRML_SCT不输出，虽然在7sc文件中占据一行（第4行），但是在计算列数时应该忽略。
            """
            tot_trans_data[cycle_number-1] = split_data[1]  # TRANS
            drct_rflt_data[cycle_number-1] = split_data[7]  # DRCT RFLT
            grnd_rflt_data[cycle_number-1] = split_data[6]  # GRND RFLT
            path_pth_thrml_data[cycle_number-1] = split_data[2]  # PTH THEML
            path_sol_scat_data[cycle_number-1] = split_data[4]  # SOL SCAT
            surf_emis_data[cycle_number-1] = split_data[3]      # SURF EMIS
            sol_obs_data[cycle_number-1] = split_data[10]       # SOL OBS
            total_rad_data[cycle_number-1] = split_data[8]      # TOTAL RAD
            ref_sol_data[cycle_number-1] = split_data[9]


            # print(split_data[1].tolist())
            # 在每个chunk处理完成后进行垃圾回收
            gc.collect()

            # Update the progress bar with the size of the processed chunk and the cycle number
            pbar.set_description(f"Processing File - Cycle {cycle_number}/{loop_number}")
            pbar.update(chunk.memory_usage(deep=True).sum())
        # print(solar_data)
        tot_trans_data = tot_trans_data.T  # TRANS
        drct_rflt_data = drct_rflt_data.T
        grnd_rflt_data = grnd_rflt_data.T
        path_pth_thrml_data = path_pth_thrml_data.T
        path_sol_scat_data = path_sol_scat_data.T
        surf_emis_data = surf_emis_data.T
        sol_obs_data = sol_obs_data.T
        total_rad_data = total_rad_data.T
        ref_sol_data = ref_sol_data.T

        with open(tot_trans_path, 'w', newline='') as tot_trans_data_file, \
                open(drct_rflt_path, 'w', newline='') as drct_rflt_data_file, \
                open(grnd_rflt_path, 'w', newline='') as grnd_rflt_data_file, \
                open(path_pth_thrml_path, 'w', newline='') as path_pth_thrml_data_file, \
                open(path_sol_scat_path, 'w', newline='') as path_sol_scat_data_file, \
                open(surf_emis_path, 'w', newline='') as surf_emis_data_file, \
                open(sol_obs_path, 'w', newline='') as sol_obs_data_file, \
                open(ref_sol_path, 'w', newline='') as ref_sol_data_file, \
                open(total_rad_path, 'w', newline='') as total_rad_data_file:

            writer_tot_trans_data = csv.writer(tot_trans_data_file)
            writer_tot_trans_data.writerows(tot_trans_data)

            # 分三次运行？？

            writer_rct_rflt_data = csv.writer(drct_rflt_data_file)
            writer_rct_rflt_data.writerows(drct_rflt_data)

            writer_grnd_rflt_data = csv.writer(grnd_rflt_data_file)
            writer_grnd_rflt_data.writerows(grnd_rflt_data)

            writer_path_pth_thrml_data = csv.writer(path_pth_thrml_data_file)
            writer_path_pth_thrml_data.writerows(path_pth_thrml_data)

            writer_sol_scat_data = csv.writer(path_sol_scat_data_file)
            writer_sol_scat_data.writerows(path_sol_scat_data)

            writer_surf_emis_data = csv.writer(surf_emis_data_file)
            writer_surf_emis_data.writerows(surf_emis_data)

            writer_sol_obs_data = csv.writer(sol_obs_data_file)
            writer_sol_obs_data.writerows(sol_obs_data)

            writer_total_rad_data = csv.writer(total_rad_data_file)
            writer_total_rad_data.writerows(total_rad_data)

            writer_ref_sol_data = csv.writer(ref_sol_data_file)
            writer_ref_sol_data.writerows(ref_sol_data)
        return tot_trans_path, drct_rflt_path, grnd_rflt_path, path_pth_thrml_path, path_sol_scat_path, surf_emis_path, sol_obs_path, total_rad_path, ref_sol_path

if __name__ == '__main__':
    ### 需要修改的默认值 ###
      # bigCSV的路径
    file_path ="F:\\research\\Processing_program\\MODTRAN_dataset\\MOD\\640_850_0.05_fine\\R05\\data.txt"
      # 计划将bigCSV分割的组数
    group_number = 2  # 为了代码的连续运行，建议直接在代码中修改，而非采用交互输入 另外建议大于2组 数值等于1是有点bug ORZ
      # 波段数目
    v1 = 640
    v2 = 850
    dv = 0.05
    # band_number = 1701  # 仅与MODTRAN模型设置的波段范围和步长有关（乘积+1） 300-2000 DV=1 填写1701
    # loop_number = 4608  # 默认4608 其数值会根据count（）函数返回值更改 这里的loop_number指的是bigCSV的总循环数 group_n * loop_n = MODTRAN设置的组数


    # 统计波段的数目（MODTRAN模拟的波段范围有多少）
    def bandcount():
        bandnum = (v2 - v1) * (1 / dv) + 1
        print(f'MODTRAN模拟的波段范围包含波段数目为：{bandnum}')
        return bandnum
    band_number = bandcount()

    # 统计bigCSV的循环数目（MODTRAN生成了多少中组合的数据）
    def count():
        # print('\n统计大型TXT的数据组数目,请稍等...')
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = []
            for i, line in enumerate(tqdm(file, desc="Processing lines")):
                lines.append(line)
                # print(i)
            total_lines = i + 1  # 获取文件总行数
        print(total_lines)
        loop_count = int(total_lines) / band_number
        print(f"\t结果：bigCSV共计{loop_count}组循环")
        return loop_count
    loop_number = count()
    print(loop_number)

    print("hi")
    # 调用分割函数number
    loop_count = loop_number*band_number  # 这里的loop_count指的是bigCSV的总行数
    smallcsv_path_list = split_big_csv(file_path, loop_number, group_number, band_number)
    print(smallcsv_path_list)

    # 调用函数处理文件
    loop = len(smallcsv_path_list)
    smallcsv_loop_count = int(loop_count/group_number)
    smalltot_trans_path = []
    smalldrct_rflt_path = []
    smallgrnd_rflt_path = []
    smallpath_pth_thrml_path = []
    smallpath_sol_scat_path = []
    smallpath_surf_emis_path = []
    smallpath_sol_obs_path = []
    smallpath_total_rad_path = []
    smallpath_ref_sol_path = []

    if loop > 1: # 这个loop是指smallCSV文件的个数
        print("\n单独对每个smallCSV文件进行处理中，请稍等...")
        for i in range(1, len(smallcsv_path_list)+1):
            print(f"\n\t目前在处理{i}/{len(smallcsv_path_list)}个文件夹")
            tot_trans, drct_rflt, grnd_rflt, path_pth_thrml, path_sol_scat, path_surf_emis, path_sol_obs, path_total_rad, path_ref_sol = process_to_txts(smallcsv_path_list[i-1], smallcsv_loop_count, band_number, i)
            smalltot_trans_path.append(tot_trans)
            smalldrct_rflt_path.append(drct_rflt)
            smallgrnd_rflt_path.append(grnd_rflt)
            smallpath_pth_thrml_path.append(path_pth_thrml)
            smallpath_sol_scat_path.append(path_sol_scat)
            smallpath_surf_emis_path.append(path_surf_emis)
            smallpath_sol_obs_path.append(path_sol_obs)
            smallpath_total_rad_path.append(path_total_rad)
            smallpath_ref_sol_path.append(path_ref_sol)
    else:
        print("\n对大型TXT文件进行直接处理（本质上将大型TXT文件转化为了smallCSV1.csv，是对副本进行操作）")
        process_to_txts(smallcsv_path_list[0], smallcsv_loop_count, band_number, 0)
    # print(smallTRAN_path)

    # 对于分割得到的零散文件进行合并，得到完整的TRAN、SOLTR和SOLAR文件
    # for t, st, sl in zip(smallTRAN_path,smallSOLTR_path,smallSOLAR_path):
        # print(f"{t}\n{st}\n{sl}")
    current_dirname = os.path.dirname(file_path)
    # print(f'测试：{current_dirname}')
    total_tot_trans = pd.concat([pd.read_csv(f, header=None) for f in smalltot_trans_path], axis=1)
    # print(total_tran)
    total_tot_trans.to_csv(os.path.join(current_dirname,'TOTAL_tot_trans.csv'), index=False, header=None)
    gc.collect()


    total_drct_rflt = pd.concat([pd.read_csv(f, header=None) for f in smalldrct_rflt_path], axis=1)
    total_drct_rflt.to_csv(os.path.join(current_dirname, 'TOTAL_drct_rflt.csv'), index=False, header=None)
    gc.collect()

    total_grnd_rflt = pd.concat([pd.read_csv(f, header=None) for f in smallgrnd_rflt_path], axis=1)
    total_grnd_rflt.to_csv(os.path.join(current_dirname, 'TOTAL_grnd_rflt.csv'), index=False, header=None)
    gc.collect()

    total_path_pth_thrml = pd.concat([pd.read_csv(f, header=None) for f in smallpath_pth_thrml_path], axis=1)
    total_path_pth_thrml.to_csv(os.path.join(current_dirname, 'TOTAL_pth_thrml.csv'), index=False, header=None)
    gc.collect()

    total_path_sol_scat = pd.concat([pd.read_csv(f, header=None) for f in smallpath_sol_scat_path], axis=1)
    total_path_sol_scat.to_csv(os.path.join(current_dirname, 'TOTAL_sol_scat.csv'), index=False, header=None)
    gc.collect()

    total_path_surf_emis = pd.concat([pd.read_csv(f, header=None) for f in smallpath_surf_emis_path], axis=1)
    total_path_surf_emis.to_csv(os.path.join(current_dirname, 'TOTAL_surf_emis.csv'), index=False, header=None)
    gc.collect()

    total_path_sol_obs = pd.concat([pd.read_csv(f, header=None) for f in smallpath_sol_obs_path], axis=1)
    total_path_sol_obs.to_csv(os.path.join(current_dirname, 'TOTAL_sol_obs.csv'), index=False, header=None)
    gc.collect()

    total_path_total_rad = pd.concat([pd.read_csv(f, header=None) for f in smallpath_total_rad_path], axis=1)
    total_path_total_rad.to_csv(os.path.join(current_dirname, 'TOTAL_total_rad.csv'), index=False, header=None)
    gc.collect()

    total_ref_sol = pd.concat([pd.read_csv(f, header=None) for f in smallpath_ref_sol_path], axis=1)
    total_ref_sol.to_csv(os.path.join(current_dirname, 'TOTAL_ref_sol.csv'), index=False, header=None)
    gc.collect()