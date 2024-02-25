# %%
import pickle
import os
import glob
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pydmd import MrDMD, DMD, SpDMD, HankelDMD, FbDMD, BOPDMD, OptDMD, HAVOK
from pydmd.plotter import plot_eigs_mrdmd, plot_eigs, plot_summary
from pydmd.preprocessing.hankel import hankel_preprocessing

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    time_array = data[:, 0]
    data_array = data[:, 1:]
    print(f"Data shape: {data_array.shape}")
    return data

def calculate_mean_cumulative_error(original_data, reconstructed_data, num_snapshots):
    min_columns = min(original_data.shape[1], reconstructed_data.shape[1])
    # Taking the real part 
    error = (reconstructed_data[:num_snapshots, :min_columns].real -
             original_data[:num_snapshots, :min_columns].real)
    cumulative_error = np.sum(error**2)
    mean_cumulative_error = cumulative_error / (num_snapshots * min_columns)
    return mean_cumulative_error


def plot_error_vs_snapshots(original_data, reconstructed_data, max_snapshots, save_dir):
    mean_cumulative_errors = []
    snapshot_numbers = []
    increment = 10
    for snapshots in range(1, max_snapshots + 1, increment):
        # Calculate the mean cumulative error for the current number of snapshots
        mean_cumulative_error = calculate_mean_cumulative_error(
            original_data, reconstructed_data, snapshots)
        mean_cumulative_errors.append(mean_cumulative_error)
        snapshot_numbers.append(snapshots)

    # Plotting the error vs. number of snapshots
    plt.figure(figsize=(10, 5))
    plt.plot(snapshot_numbers, mean_cumulative_errors, marker='o', linestyle='-', color='blue', linewidth=1, markersize=5)
    plt.xlabel('Number of Snapshots')
    plt.ylabel('Mean Cumulative Error')
    plt.title('Mean Cumulative Error vs. Number of Snapshots')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'mean_cumulative_error_vs_snapshots.png'))
    plt.close()


    return mean_cumulative_errors

# Main section
if __name__ == "__main__":
    original_data_path = r"C:\Users\jason\OneDrive\桌面\Research\UofT\DMD_Data\SHARED\Reconstruction\original_data.pkl"
    reconstructed_data_path = r"C:\Users\jason\OneDrive\桌面\Research\UofT\DMD_Data\SHARED\Reconstruction\reconstructed_data.pkl"
    original_data = load_data(original_data_path)
    reconstructed_data = load_data(reconstructed_data_path)
    max_snapshots = original_data.shape[0]
    save_dir = r"C:\Users\jason\OneDrive\桌面\Research\UofT\DMD_Data\SHARED\Reconstruction"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    errors = plot_error_vs_snapshots(original_data, reconstructed_data, max_snapshots, save_dir)




