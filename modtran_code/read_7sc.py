# -*- coding: utf-8 -*-
# @Time    : 2023/11/7 20:00
# @Author  : xu xinwen
# @FileName: read_7sc.py
# @Software: PyCharm

# 将7sc文件读取为表头文件和数据文件

import os
from tqdm import tqdm

print("Reading 7sc(7SC2datatxt)")
# Define the cycle length
# cycle_length = 20013  # 根据在mod5中设置的波长范围及步长来确定  cycle_length是每个循环的行数
cycle_length = 433  # 根据在mod5中设置的波长范围及步长来确定  cycle_length是每个循环的行数(包括tp5的行数和最后一行-9999)
# Paths to the files
head_file_path = "F:\\research\\Processing_program\\MODTRAN_dataset\\MOD\\640_850_0.5_fine\\R1\\head.txt"
data_file_path = "F:\\research\\Processing_program\\MODTRAN_dataset\\MOD\\640_850_0.5_fine\\R1\\data.txt"
os.makedirs(os.path.dirname(head_file_path), exist_ok=True)
os.makedirs(os.path.dirname(data_file_path), exist_ok=True)
input_file_path = "F:\\research\\Processing_program\\MODTRAN_dataset\\MOD\\640_850_0.5_fine\\R1\\dataset_05_R1.7sc"

# Get the total size of the file for the progress bar
total_size = os.path.getsize(input_file_path)

# Create a progress bar for counting lines
with tqdm(total=total_size, unit='B', unit_scale=True, desc='Counting lines') as pbar:
    total_lines = 0
    with open(input_file_path, 'rb') as f:
        for line in f:
            total_lines += 1
            pbar.update(len(line))
print('Total lines counted:', total_lines)
print('Step 1 over!')

# Create a progress bar for processing the file
with tqdm(total=total_lines, desc='Processing cycles', unit='line') as pbar:
    with open(input_file_path, 'r') as file, open(head_file_path, 'w') as head_file, open(data_file_path, 'w') as data_file:
        cycle_counter = 0  # Initialize a counter for the cycles
        current_cycle_lines = []  # Initialize a list to hold the current cycle of lines

        # Read the file line by line
        for line in file:
            # Add line to the current cycle list
            current_cycle_lines.append(line)

            # Check if we've reached the end of a cycle
            if len(current_cycle_lines) == cycle_length:
                # Increment the cycle counter
                cycle_counter += 1

                # Write the first 11 lines to the head file
                head_file.write(f'Cycle {cycle_counter}:\n')
                head_file.writelines(current_cycle_lines[:11])
                # head_file.write('\n')  # Add a newline for separation between cycles

                # Write the remaining lines to the data file
                # data_file.write(f'Cycle {cycle_counter}:\n')
                data_file.writelines(current_cycle_lines[11:cycle_length-1])
                # data_file.write('\n')  # Add a newline for separation between cycles

                # Reset the current cycle list for the next cycle
                current_cycle_lines = []

            # Update the progress bar
            pbar.update(1)

        # If there are any lines left after the last complete cycle, write them to the data file
        if current_cycle_lines:
            # Increment the cycle counter
            cycle_counter += 1

            # Write the first 11 lines to the head file
            head_file.write(f'Cycle {cycle_counter}:\n')
            head_file.writelines(current_cycle_lines[:11])
            head_file.write('\n')  # Add a newline for separation between cycles

            # Write the remaining lines to the data file
            data_file.write(f'Cycle {cycle_counter}:\n')
            data_file.writelines(current_cycle_lines[11:])
            data_file.write('\n')  # Add a newline for separation between cycles

print('Step 2 over!')