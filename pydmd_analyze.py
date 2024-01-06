# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:06:27 2023

@author: Jack
"""

from pydmd import DMD, FbDMD, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing.hankel import hankel_preprocessing
import matplotlib.pyplot as plt
import numpy as np

# %% A


fp = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\left_region\ux.csv"
data = np.loadtxt(fp, skiprows=1, delimiter=",")
fp = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\left_region\uy.csv"
data_y = np.loadtxt(fp, skiprows=1, delimiter=",")
fp = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\left_region\uz.csv"
data_z = np.loadtxt(fp, skiprows=1, delimiter=",")

# %%
fp = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\left_region\p.csv"
data_p = np.loadtxt(fp, skiprows=1, delimiter=",")

fp = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\building_p.csv"
data_p2 = np.loadtxt(fp, skiprows=1, delimiter=",")

# %%
import os
from sklearn import preprocessing
# normalized_data = preprocessing.normalize(data_select)
tn = 1200
data_select1 = data[:tn, 3001:6001]
data_select2 = data_y[:tn, 3001:6001]
data_select3 = data_z[:tn, 3001:6001]
data_select4 = data_p[:tn, 3001:6001]
data_select5 = data_p2[:tn, 1:]
t = data[:tn, 0]


save_dir = r"C:\Users\Jack\offline\MrDMD"
coord_fp = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\left_region\coords.csv"
coords = np.loadtxt(coord_fp, skiprows=1, delimiter=",")
coords_select = coords[3000:6000, 1:]

coord_fp_building = r"C:\Users\Jack\OneDrive - University of Toronto\Research\Projects\2023_DMD\SHARED\building_coords_flat.csv"
coords_building = np.loadtxt(coord_fp_building, skiprows=1, delimiter=",")[:, [1, 2]]

data_select1 = preprocessing.normalize(data_select1[:, coords_select[:, 0] < 0.05])
data_select2 = preprocessing.normalize(data_select2[:, coords_select[:, 0] < 0.05])
data_select3 = preprocessing.normalize(data_select3[:, coords_select[:, 0] < 0.05])
data_select4 = preprocessing.normalize(data_select4[:, coords_select[:, 0] < 0.05])
coords_select = coords_select[coords_select[:, 0] < 0.05]

data_select5 = preprocessing.normalize(data_select5)

# data_select = np.hstack([data_select1, data_select2, data_select3, data_select4])
# data_select = np.hstack([data_select1, data_select2])
data_select = np.hstack([data_select1, data_select2, data_select4, data_select5])
# data_select = data_select4.copy()

# %%

data_select = data_select - data_select.mean(axis=0)

normalized_data = data_select
n = normalized_data.shape[1]

# %%
# tn = 400
# data_select = data[:tn, 3000:6000]
# t = data[:tn, 0]
# dmd0 = FbDMD(svd_rank=20)
# dmd0 = BOPDMD(svd_rank=20, num_trials=0, eig_constraints={"imag", "conjugate_pairs"})

# delays = 1
# delay_dmd = hankel_preprocessing(dmd0, d=delays)
# num_t = len(t) - delays + 1
# delay_dmd.fit(data_select.T, t=t[:num_t])

from pydmd import MrDMD, DMD, SpDMD, HankelDMD, FbDMD, BOPDMD, OptDMD, HAVOK
from pydmd.plotter import plot_eigs_mrdmd
from pydmd.preprocessing.hankel import hankel_preprocessing



x = np.arange(data_select.shape[1])

# delay = 5
# sub_dmd = HankelDMD(svd_rank=0.9, exact=True, d=delay, sorted_eigs='abs', tikhonov_regularization=1e-8)
sub_dmd = DMD(svd_rank=-1)
# sub_dmd = FbDMD(svd_rank=0.9, sorted_eigs='abs', exact=True)

# 
# dmd0 = BOPDMD(svd_rank=0.5, num_trials=0, eig_sort="abs")
# dmd0._varpro_opts_dict["verbose"] = True
# dmd0 = hankel_preprocessing(dmd0, d=delay)
# num_t = len(t) - delay + 1
# dmd0.fit(X=normalized_data.T, t=t[:num_t])

# dmd0 = HAVOK()
# dmd0.fit(normalized_data.T, t[1] - t[0])


dmd0 = MrDMD(sub_dmd, max_level=7, max_cycles=10)
dmd0.fit(X=normalized_data.T)
print("# modes:", dmd0.modes.shape)

for level in range(dmd0.max_level):
    print(f"level: {level}, shape:{dmd0.partial_modes(level=level).shape}")

# %%

for j in [0, 50, 100, 200, 1000, 2000, 3000]:
    pdata = dmd0.reconstructed_data
    plt.plot(pdata[j, :], alpha=0.7, label=f"DMD")
    plt.plot(normalized_data[:, j], alpha=0.6, label="original")
    plt.legend()
    plt.show()
    
# %%

idx_li = dmd0.time_window_bins(0, 400)
freqs = []
amps = []
for idx in idx_li:
    if len(dmd0.dmd_tree[idx].frequency) > 0:
        freqs.append(dmd0.dmd_tree[idx].frequency[0])
        amps.append(abs(dmd0.dmd_tree[idx].amplitudes[0]))
        plt.text(freqs[-1], amps[-1], s=str(idx))
plt.scatter(freqs, amps)

# %%
plot_summary(dmd0, snapshots_shape=(60, 20*delay*3))
# plot_eigs_mrdmd(dmd0, show_axes=True, show_unit_circle=True, figsize=(8, 8))

def make_plot(X, x=None, y=None, figsize=(12, 8), title="", vmin=None, vmax=None):
    """
    Plot of the data X
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    X = np.real(X)
    CS = plt.pcolor(y, x, X.T, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(CS)
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.savefig(os.path.join(save_dir, f"{level}_{j}_spectrum.png"))

# vmin = min(dmd0.reconstructed_data.real[:, :].min(), data_select.min())
# vmax = max(dmd0.reconstructed_data.real[:, :].max(), data_select.max())
# make_plot(dmd0.reconstructed_data.T, x=x, y=t, vmin=vmin, vmax=vmax)
# make_plot(data_select, x=x, y=t, vmin=vmin, vmax=vmax)

# %%
# plot_summary(dmd0, snapshots_shape=(150, 60))

dim_labels = ["U", "V", "P"]

import matplotlib.pyplot as plt


div_idx = data_select5.shape[1]

dt = t[1] - t[0]
for level in range(0, dmd0.max_level+1):
    pmodes = dmd0.partial_modes(level=level)
    pdyna = dmd0.partial_dynamics(level=level)
    peigs = dmd0.partial_eigs(level=level)
    freq = np.log(peigs).imag / (2 * np.pi * dt)
    grow = peigs.real
    t = data[:tn, 0]
# t = t[:num_t]
# for level in range(1):
    # pmodes = dmd0.modes
    # pdyna = dmd0.dynamics

    plt.figure(figsize=(6,4))
    fig = plt.plot(t, pdyna.real.T)
    plt.legend(range(pdyna.real.shape[0]))
    plt.title(f"level:{level}")
    # plt.show()
    plt.savefig(os.path.join(save_dir, f"{level}_dynamics.png"))
    plt.close()

    n_dim = 3
    n_i, n_j, n_k = 60, 20, n_dim
    n_i2, n_j2 = 20, 25
    
    pmodes = pmodes[:n, :] # use only first timestep if there is delay embedding
    pmodes1 = pmodes[:-div_idx, :]
    pmodes2 = pmodes[-div_idx:, :]
    for j in range(pmodes.shape[1]):
        size = (20, 16)
        # fig = plt.plot(x, pmodes.real)
        X = coords_select[:, 0].reshape(n_j, n_i)[0, :]
        Y = coords_select[:, 1].reshape(n_j, n_i)[:, 0]
        
        Z = abs(pmodes1[:, j]).reshape(n_k, n_j, n_i)
        for dim in range(n_dim):
            Phi = Z[dim, :, :]
    
            plt.figure(figsize=(8, 6))
            cmap="viridis"
            ax = plt.subplot(111)
            vmin = 0
            vmax = Z.max()
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Phi, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            
            if n_dim > 1:
                plt.quiver(X, Y, Z[0, :, :], Z[1, :, :], scale_units='xy', angles='uv')
            ax.set_title(f"level:{level}, mode:{j}, {dim_labels[dim]}, {freq[j]:.1f} Hz, g:{grow[j]:.2f}")
            ax.set_aspect("equal")
            plt.savefig(os.path.join(save_dir, f"modeshape_{level}_{j}_{dim_labels[dim]}_{freq[j]:.1f}Hz.png"))
            plt.close()
            
        X2 = coords_building[:, 0].reshape(n_j2, n_i2)[0, :]
        Y2 = coords_building[:, 1].reshape(n_j2, n_i2)[:, 0]

        Z = abs(pmodes2[:, j]).reshape(n_j2, n_i2)
        Phi = Z
        plt.figure(figsize=(8, 6))
        cmap="viridis"
        ax = plt.subplot(111)
        vmin = 0
        vmax = Z.max()
        levels = np.linspace(vmin, vmax, 20)
        CS = plt.contourf(X2, Y2, Phi, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
        colorbar = plt.colorbar(CS)
        ax.set_title(f"level:{level}, mode:{j}, building_p, {freq[j]:.1f} Hz, g:{grow[j]:.2f}")
        ax.set_aspect("equal")
        line_value = 0.5 * 2/3
        ax.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=0.1, color='red', linestyle='-', linewidth=2)
        ax.axvline(x=0.2, color='red', linestyle='-', linewidth=2)
        ax.axvline(x=0.3, color='red', linestyle='-', linewidth=2)
        plt.savefig(os.path.join(save_dir, f"modeshape_{level}_{j}_building_p_{freq[j]:.1f}Hz.png"))
        plt.close()
            
        
        # Plot dynamics
        plt.figure(figsize=(8, 6))
        fig = plt.plot(t, pdyna.real.T[:, j])
        plt.title(f"level:{level}, mode:{j}")
        ax.set_aspect("equal")
        plt.savefig(os.path.join(save_dir, f"modeshape_{level}_{j}_dynamics.png"))
        plt.close()

        Z = np.angle(pmodes1[:n, j]).reshape(n_k, n_j, n_i)
        cmap="twilight"
        for dim in range(n_dim):
            Phi = Z[dim, :, :]
            
            # scale = 10
            
            plt.figure(figsize=size)
            ax = plt.subplot(111)
            vmin = -np.pi
            vmax = np.pi
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Phi, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            # plt.quiver(X, Y, U, V, scale_units='xy', angles='uv')
            ax.set_title(f"level:{level}, mode:{j}, {dim_labels[dim]}_im")
            ax.set_aspect("equal")
            plt.savefig(os.path.join(save_dir, f"modeshape_{level}_{j}_{dim_labels[dim]}_phase_{freq[j]:.1f}Hz.png"))
            plt.close()
            
        Z = np.angle(pmodes2[:, j]).reshape(n_j2, n_i2)
        Phi = Z
        plt.figure(figsize=(8, 6))
        cmap="twilight"
        ax = plt.subplot(111)
        vmin = -np.pi
        vmax = np.pi
        levels = np.linspace(vmin, vmax, 20)
        CS = plt.contourf(X2, Y2, Phi, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
        colorbar = plt.colorbar(CS)
        ax.set_title(f"level:{level}, mode:{j}, building_p_phase")
        ax.set_aspect("equal")
        
        line_value = 0.5 * 2/3
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=0.1, color='red', linestyle='-', linewidth=2)
        ax.axvline(x=0.2, color='red', linestyle='-', linewidth=2)
        ax.axvline(x=0.3, color='red', linestyle='-', linewidth=2)

        plt.savefig(os.path.join(save_dir, f"modeshape_{level}_{j}_building_p_phase_{freq[j]:.1f}Hz.png"))
        plt.close()
            
        '''
        plt.figure(figsize=(8, 6))
        fig = plt.plot(t, pdyna.imag.T[:, j])
        plt.title(f"level:{level}, mode:{j}_im")
        ax.set_aspect("equal")
        plt.savefig(os.path.join(save_dir, f"modeshape_{level}_{j}_im_dynamics.png"))
        plt.close()
        '''


# %%

pdyna = dmd0.dynamics
j = 2
plt.figure(figsize=(8, 6))
fig = plt.plot(t, pdyna.real.T[:, j])
plt.title(f"level:{level}, mode:{j}")
ax.set_aspect("equal")

    
# %%
tmp = np.array(dmd0.modes_activation_bitmask)
tmp[:] = True
dmd0.modes_activation_bitmask = tmp
    
ends = [0, 1, 2]
for end in ends:
    j = 50
    tmp = np.array(dmd0.modes_activation_bitmask)
    tmp[:end] = False
    dmd0.modes_activation_bitmask = tmp
    pdata = dmd0.reconstructed_data
    plt.plot(pdata[j, :], alpha=0.7, label=f"DMD-{end}")

plt.plot(data_select[:, j], alpha=0.6, label="original")
plt.legend()


# %%

# %%
fshed = 4.72
Tshed = 1/fshed
fs = 1/0.005
print("# of snapshots per shed", Tshed*fs)


# %%
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print(
    f"Frequencies (imaginary component): {np.round(delay_dmd.eigs, decimals=3)}"
)

vmin = min(delay_dmd.reconstructed_data.real[:, :].min(), data_select.min())
vmax = max(delay_dmd.reconstructed_data.real[:, :].max(), data_select.max())

plt.figure(figsize=(20, 5))
plt.title("Reconstructed Data")
plt.imshow(delay_dmd.reconstructed_data.real[:, :], cmap='jet', vmin=vmin, vmax=vmax)
colorbar = plt.colorbar()
plt.show()
plt.figure(figsize=(20, 5))
plt.title("Ground Truth Data")
plt.imshow(data_select[:, :].T, cmap='jet', vmin=vmin, vmax=vmax)
colorbar = plt.colorbar()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
X = np.linspace(0, 10, 25)  # Replace with your x coordinates
Y = np.linspace(0, 5, 120)  # Replace with your y coordinates
data = np.random.rand(3000, 1)  # Replace with your data values

# Reshape the data and grid coordinates
Z = data.reshape(len(Y), len(X))

# Set vmin and vmax based on your data
vmin, vmax = data.min(), data.max()

# Define contour levels based on the data range
levels = np.linspace(vmin, vmax, 20)  # Adjust the number of levels as needed

# Plot the contour plot with specified levels
CS = plt.contour(X, Y, Z, levels=levels, cmap="jet", vmin=vmin, vmax=vmax)
plt.colorbar(CS)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Contour Plot of Data')
plt.show()