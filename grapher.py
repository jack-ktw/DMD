# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 02:42:45 2024

@author: Keith
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

def parse_excel_sheet(file_path):
    # Read Excel sheet into a DataFrame
    df = pd.read_excel(file_path)
    
    # Initialize nested dictionary to store average errors
    average_errors_dict = {}
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        delay_length = row['Delay Length']
        num_snapshots = row['Number of Snapshots']
        average_error = row['Average Error']
        
        # Check if delay_length exists as a key in the dictionary
        if delay_length not in average_errors_dict:
            average_errors_dict[delay_length] = {}
        
        # Store average error in the nested dictionary
        average_errors_dict[delay_length][num_snapshots] = average_error
    
    return average_errors_dict

# Example usage:
file_path = r"D:\Python Files\Research - DMD\research_paper\HankelDMD-rectangular\average_errors_all.xlsx"
average_errors_dict = parse_excel_sheet(file_path)
save_dir = r"D:\Python Files\Research - DMD\research_paper\HankelDMD-rectangular"
plt.figure(figsize=(10, 6))
data = []
for delay_length in average_errors_dict.keys():
    plt.plot(
        list(average_errors_dict[delay_length].keys()),
        list(average_errors_dict[delay_length].values()),
        marker='o', linestyle='-', label=f'Delay Length: {delay_length}'
    )

plt.xlabel('Number of Snapshots [N]')
plt.ylabel('Total Reconstruction Error/N')
# plt.title('Sensitivity Analysis: Average Error vs. Number of Snapshots')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, "sensitivity_analysis_delays.png"))