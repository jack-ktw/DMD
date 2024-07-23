# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 20:12:16 2024

@author: Keith
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_multiple_csv_data(csv_paths):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_mode_numbers = []
    
    # First pass: determine the global vmax for color normalization
    for csv_path in csv_paths:
        # Read data from CSV file
        df = pd.read_csv(csv_path)
        mode_numbers = df['Mode Number']
        all_mode_numbers.extend(mode_numbers)
    
    # Determine global vmax for color normalization
    global_vmax = max(all_mode_numbers)
    
    # Second pass: plot the data
    for csv_path in csv_paths:
        # Read data from CSV file
        df = pd.read_csv(csv_path)
        mode_numbers = df['Mode Number']
        frequencies = df['Frequency (Hz)']
        amplitudes = df['Amplitude']
        
        # Plot the amplitude vs frequency for each mode
        sc = ax.scatter(frequencies,
                        amplitudes,
                        c=mode_numbers, cmap='viridis', vmin=1, vmax=global_vmax, s=50, label=f"File: {csv_path}")
        
        # Add text labels for each point
        for i in range(len(frequencies)):
            ax.text(frequencies.iloc[i],
                    amplitudes.iloc[i],
                    str(mode_numbers.iloc[i]), ha='right', va='bottom', fontsize=16)
    
    # Set the plot title and axis labels
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Amplitude", fontsize=20)
    ax.set_xlim(0)
    
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add a colorbar to the plot
    norm = mcolors.Normalize(vmin=1, vmax=global_vmax)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
    cbar.set_label("Mode Number", fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    
    # Add a legend
    ax.legend(fontsize=12)
    
    # Save the plot
    plt.savefig(r"C:\Users\Keith\Documents\research_paper\amplitude_frequency.png")
    plt.close(fig)
    plt.clf()
    plt.close("all")
    
    
csv_files = [
    r'C:\Users\Keith\Documents\research_paper\Cp_v2_factor\HankelDMD-update_pressure_100_full_rank\mode_data_cfd.csv',
    r'C:\Users\Keith\Documents\research_paper\pressure-case\HankelDMD-update_pressure_100_full_rank\mode_data_wt.csv',
    # Add more file paths as needed
]

plot_multiple_csv_data(csv_files)