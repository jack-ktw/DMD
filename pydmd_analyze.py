# -*- coding: utf-8 -*-
"""
@author: Jack
"""

# %%
import os
import glob
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import preprocessing
from pydmd import MrDMD, DMD, SpDMD, HankelDMD, FbDMD, BOPDMD, OptDMD, HAVOK
from pydmd.plotter import plot_eigs_mrdmd, plot_eigs, plot_summary
from pydmd.preprocessing.hankel import hankel_preprocessing
from scipy.integrate import quad
import time
from scipy.interpolate import griddata
from matplotlib.ticker import MaxNLocator
import cv2
from pydmd.plotter import plot_eigs
import pandas as pd
import math
from pydmd.plotter import plot_eigs, plot_summary

matplotlib.use('Agg')

class Dataset:
    def __init__(self, name="dataset", is_building=False) -> None:
        self.path = None
        self.coords_array = None
        self.coords_ln = None
        self.data_array = np.empty((0,1))
        self.time_array = np.empty((0,1))
        self.name = name
        self.scaler = preprocessing.RobustScaler()
        self.is_building = is_building
        
    def load_data(self, path):
        self.path = path
        data = np.loadtxt(path, skiprows=1, delimiter=",")
        self.time_array = data[:, 0]
        self.data_array = data[:, 1:]
    
    def load_coords(self, path):
        self.coords_array = np.loadtxt(path, skiprows=1, delimiter=",")[:, [1, 2, 3]]
        
    def assign_coords(self, from_dataset):
        while from_dataset.coords_ln is not None:
            from_dataset = from_dataset.coords_ln
        self.coords_ln = from_dataset
        
    def get_coords(self):
        if self.coords_array is None:
            return self.coords_ln.coords_array
        return self.coords_array
        
    def trim_data(self, t1=0, t2=None, i1=0, i2=None):
        self.time_array = self.time_array[t1:t2]
        self.data_array = self.data_array[t1:t2, i1:i2]
        
        if self.coords_array is not None:
            self.coords_array = self.coords_array[i1:i2, :]
        
    def filter_data(self, x_lower=-np.inf, x_upper=np.inf, y_lower=-np.inf, y_upper=np.inf):
        if self.coords_array is None:
            coords = self.coords_ln.coords_array
        else:
            coords = self.coords_array
        x = coords[:, 0]
        y = coords[:, 1]
        idx = np.logical_and(np.logical_and(x < x_upper, x >= x_lower), np.logical_and(y < y_upper, y >= y_lower))
        self.data_array = self.data_array[:, idx]
        
        if self.coords_array is not None:
            self.coords_array = self.coords_array[idx, :]

        
    def demean_data(self):
        self.data_mean = self.data_array.mean(axis=0)
        self.data_array = self.data_array - self.data_mean

    def normalize_data(self):
        original_shape = self.data_array.shape
        flattened_data = self.data_array.flatten().reshape(-1, 1)
        self.scaler.fit(flattened_data)
        flattened_data = self.scaler.transform(flattened_data)
        self.data_array = flattened_data.reshape(original_shape)
    

class DMDAnalysisBase:
    def __init__(self, data_dir=".", save_dir="dmd_output",
                 svd_rank=-1,) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.datasets = []
        self.dmd = None
        self.dt = None
        self.train_X = None
        self.ds_idx_to_trainX_idx = None
        self.svd_rank = svd_rank
        
    def make_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
    def add_dataset(self, dataset):
        self.datasets.append(dataset)
        self.dt = dataset.time_array[1] - dataset.time_array[0]
        
    def add_datasets(self, names, relative_paths, coords_relative_paths=-1,
                     is_building_li=None):
        if is_building_li is None:
            is_building_li = [False] * len(names)
        for i in range(len(names)):
            ds = Dataset(names[i], is_building=is_building_li[i])
            data_fp = os.path.join(self.data_dir, relative_paths[i])
            ds.load_data(data_fp)
            if coords_relative_paths[i] == -1:
                ds.assign_coords(analysis.datasets[-1])
            else:
                coord_fp = os.path.join(self.data_dir, coords_relative_paths[i])
                ds.load_coords(coord_fp)
            self.add_dataset(ds)
            
    def trim_datasets(self, t1=0, t2=None, i1=0, i2=None, ds_indices=None):
        if ds_indices is None:
            ds_indices = range(len(self.datasets))
        for i in ds_indices:
            self.datasets[i].trim_data(t1, t2, i1, i2)
        
    def filter_datasets(self, x_lower=-np.inf, x_upper=np.inf, y_lower=-np.inf, y_upper=np.inf, ds_indices=None):
        if ds_indices is None:
            ds_indices = range(len(self.datasets))

        # ds_indices needs to be in descending order so linked coords are filtered first
        ds_indices = sorted(ds_indices, reverse=True)
        for i in ds_indices:
            self.datasets[i].filter_data(x_lower, x_upper, y_lower, y_upper)
            
    def demean_datasets(self, ds_indices=None):
        if ds_indices is None:
            ds_indices = range(len(self.datasets))
        for i in ds_indices:
            self.datasets[i].demean_data()
            
    def normalize_datasets(self, ds_indices=None):
        if ds_indices is None:
            ds_indices = range(len(self.datasets))
        for i in ds_indices:
            self.datasets[i].normalize_data()
    
    def normalize_group(self, ds_indices):
        data = np.empty((0,1))
        original_shape_L = []
        for idx in ds_indices:
            original_shape_L.append(self.datasets[idx].data_array.shape)
            data = np.vstack([data, self.datasets[idx].data_array.flatten().reshape(-1, 1)])
        scaler = preprocessing.RobustScaler(with_centering=False)
        scaler.fit(data)
        print("Normalizing group:", [self.datasets[idx].name for idx in ds_indices])
        print("Mean:", scaler.center_)
        print("Scale:", scaler.scale_)
        transformed_data = scaler.transform(data)
        start_idx = 0
        for i, idx in enumerate(ds_indices):
            end_idx = start_idx + self.datasets[idx].data_array.size
            self.datasets[idx].data_array = transformed_data[start_idx:end_idx, :].reshape(original_shape_L[i])
            self.datasets[idx].scaler = scaler
            start_idx = end_idx
            
    def compose_data(self, ds_indices=None):
        if ds_indices is None:
            ds_indices = range(len(self.datasets))
            
        n = len(self.datasets[0].time_array)
        data = np.empty((n, 0))
        self.ds_idx_to_trainX_idx = {}
        
        for i in ds_indices:
            start_idx = data.shape[1]
            data = np.hstack([data, self.datasets[i].data_array])
            end_idx = data.shape[1]
            self.ds_idx_to_trainX_idx[i] = (start_idx, end_idx)
        self.train_X = data
            
    def fit(self, ds_indices=None):
        raise NotImplementedError("Subclasses must implement the 'fit' method.")
        
    def plot_timeseries(self, idx_li):
        raise NotImplementedError("Subclasses must implement the 'plot_timeseries' method.")
            
    def plot_dynamics(self):
        raise NotImplementedError("Subclasses must implement the 'plot_dynamics' method.")
        
    def save_dmd(self):
        raise NotImplementedError("Subclasses must implement the 'save_dmd' method.")
        
    def load_dmd(self):
        raise NotImplementedError("Subclasses must implement the 'load_dmd' method.")
            
    def clean_up_figures(self, pattern):
        matching_files = glob.glob(pattern)
        print("cleaning up", pattern)
        for file in matching_files:
            os.remove(file)
    
            
class MrDMDAnalysis(DMDAnalysisBase):
    def __init__(self, data_dir=".", save_dir="dmd_output",
                 max_level=4, max_cycles=10, 
                 svd_rank=-1,
                 tikhonov_regularization=1e-7) -> None:
        super().__init__(data_dir, save_dir, svd_rank)  
        self.max_level = max_level
        self.max_cycles = max_cycles
        self.tikhonov_regularization = tikhonov_regularization
        
    def fit(self, ds_indices=None):
        sub_dmd = DMD(svd_rank=self.svd_rank, tikhonov_regularization=self.tikhonov_regularization)
        self.dmd = MrDMD(sub_dmd, max_level=self.max_level, max_cycles=self.max_cycles)
        self.compose_data(ds_indices=ds_indices)
        self.dmd.fit(X=self.train_X.T)
        
        print("# modes:", self.dmd.modes.shape)
        for level in range(self.dmd.max_level):
            print(f"level: {level}, shape:{self.dmd.partial_modes(level=level).shape}")
            
    def plot_timeseries(self, idx_li):
        pdata = self.dmd.reconstructed_data
        for idx in idx_li:
            #fig_name = f"timeseries_{idx}"
            plt.figure(figsize=(6, 4))
            
            cumulative_error = np.sum((pdata[idx, :] - self.train_X[:, idx])**2)
            
            plt.plot(self.train_X[:, idx], alpha=0.6, label="CFD")
            plt.plot(pdata[idx, :], alpha=0.7, label=f"DMD")
            plt.ylabel(f"Pressure for Tap {idx}")
            plt.xlabel("Time Step")
            plt.text(0.5, 0.02, f'Cumulative Error: {cumulative_error:.2f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes)
    
            plt.legend()
            plt.title(fig_name)
            plt.savefig(os.path.join(self.save_dir, f"0_{fig_name}.png"))
            plt.close()

            
    def plot_dynamics(self, max_level=-1):
        pattern = os.path.join(self.save_dir, f"1_*_dynamics.png")
        self.clean_up_figures(pattern)
        
        print("plotting dynamics:")
        print("Saving to:", self.save_dir)
        if max_level == -1:
            max_level = self.dmd.max_level
        for level in range(max_level+1):
            pmodes = self.dmd.partial_modes(level=level)
            pdyna = self.dmd.partial_dynamics(level=level)
            t = self.datasets[0].time_array

            fig_name = f"{level}_dynamics"
            fig = plt.figure(figsize=(6,4))
            plt.plot(t, pdyna.real.T)
            plt.legend(range(pdyna.real.shape[0]))
            plt.title(f"level:{level}")
            plt.savefig(os.path.join(self.save_dir, f"1_{fig_name}.png"))
            plt.close(fig)
        plt.cla()
        plt.clf()
        plt.close("all")
        gc.collect()
            
    def save_dmd(self):
        self.dmd.save(os.path.join(self.save_dir, "dmd.pkl"))
        
    def load_dmd(self):
        self.dmd = MrDMD.load(os.path.join(self.save_dir, "dmd.pkl"))
            
    def clean_up_figures(self, pattern):
        matching_files = glob.glob(pattern)
        print("cleaning up", pattern)
        for file in matching_files:
            os.remove(file)

    def plot_modes(self, ds_idx, max_level=-1, plot_negative=False):
        if max_level == -1:
            max_level = self.dmd.max_level
        start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
        coords_array = self.datasets[ds_idx].get_coords()
        n_i = len(np.unique(coords_array[:, 0]))
        n_j = len(np.unique(coords_array[:, 1]))
        name = self.datasets[ds_idx].name
        is_building = self.datasets[ds_idx].is_building
        
        pattern = os.path.join(self.save_dir, f"2_modeshape_*_*_{name}_*Hz.png")
        self.clean_up_figures(pattern)
        
        print("plotting modes:", name)
        print("Saving to:", self.save_dir)
        for level in range(max_level+1):
            print("level:", level)
            pmodes = self.dmd.partial_modes(level=level)
            peigs = self.dmd.partial_eigs(level=level)
            
            for mode_idx in range(pmodes.shape[1]):
                Z_all = abs(pmodes[:, mode_idx])
                pmodes_select = pmodes[start_i:end_i, mode_idx].reshape(n_j, n_i)
                X = coords_array[:, 0].reshape(n_j, n_i)[0, :]
                Y = coords_array[:, 1].reshape(n_j, n_i)[:, 0]        
                Z = abs(pmodes_select)
                
                vmin = 0
                vmax = Z_all.max()
                cmap="viridis"
                
                if plot_negative:
                    phase = np.angle(pmodes_select)
                    Z[phase < 0] = -Z[phase < 0]
                    # Z[phase > np.pi] = -Z[phase > np.pi]
                    vmin = -vmax
                    cmap="RdBu"
                # Z = np.linalg.norm(pmodes_select)
                
                freq = np.log(peigs[mode_idx]).imag / (2 * np.pi * self.dt)
                grow = peigs[mode_idx].real

                fig = plt.figure(figsize=(8, 6))
                ax = plt.subplot(111)
                levels = np.linspace(vmin, vmax, 20)
                CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
                colorbar = plt.colorbar(CS)
                ax.set_title(f"level:{level}, mode:{mode_idx}, {name}, {freq:.1f} Hz, g:{grow:.2f}")
                ax.set_aspect("equal")
                
                if is_building:
                    ax.axvline(x=0.1, color='black', linestyle='-', linewidth=2)
                    ax.axvline(x=0.2, color='black', linestyle='-', linewidth=2)
                    ax.axvline(x=0.3, color='black', linestyle='-', linewidth=2)

                plt.savefig(os.path.join(save_dir, f"2_modeshape_{level}_{mode_idx}_{name}_{freq:.1f}Hz.png"))
                plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
                
    def plot_phase(self, ds_idx, max_level=-1, plot_negative=False):
        if max_level == -1:
            max_level = self.dmd.max_level
        start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
        coords_array = self.datasets[ds_idx].get_coords()
        n_i = len(np.unique(coords_array[:, 0]))
        n_j = len(np.unique(coords_array[:, 1]))
        name = self.datasets[ds_idx].name
        is_building = self.datasets[ds_idx].is_building
        
        pattern = os.path.join(self.save_dir, f"2_modeshape_*_*_{name}_*Hz_phase.png")
        self.clean_up_figures(pattern)
        print("plotting phases:", name)
        print("Saving to:", self.save_dir)
        for level in range(max_level+1):
            print("level:", level)
            pmodes = self.dmd.partial_modes(level=level)
            peigs = self.dmd.partial_eigs(level=level)
            
            for mode_idx in range(pmodes.shape[1]):
                pmodes_select = pmodes[start_i:end_i, mode_idx].reshape(n_j, n_i)
                X = coords_array[:, 0].reshape(n_j, n_i)[0, :]
                Y = coords_array[:, 1].reshape(n_j, n_i)[:, 0]        
                Z = np.angle(pmodes_select)
                
                vmin = -np.pi
                vmax = np.pi
                # cmap="twilight"
                cmap="hsv"
                
                # if plot_negative:
                #     Z[Z < 0] = -Z[Z < 0]
                #     Z[Z > np.pi] = Z[Z > np.pi] - np.pi
                #     cmap="hsv"
                #     vmin = 0
                
                freq = np.log(peigs[mode_idx]).imag / (2 * np.pi * self.dt)
                grow = peigs[mode_idx].real

                fig = plt.figure(figsize=(8, 6))
                ax = plt.subplot(111)
                levels = np.linspace(vmin, vmax, 20)
                CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
                colorbar = plt.colorbar(CS)
                ax.set_title(f"phase: level:{level}, mode:{mode_idx}, {name}, {freq:.1f} Hz, g:{grow:.2f}")
                ax.set_aspect("equal")
                
                if is_building:
                    ax.axvline(x=0.1, color='black', linestyle='-', linewidth=2)
                    ax.axvline(x=0.2, color='black', linestyle='-', linewidth=2)
                    ax.axvline(x=0.3, color='black', linestyle='-', linewidth=2)

                plt.savefig(os.path.join(save_dir, f"2_modeshape_{level}_{mode_idx}_{name}_{freq:.1f}Hz_phase.png"))
                plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
                
    def plot_all_ds(self, max_level=-1, plot_negative=False):
        for ds_idx in self.ds_idx_to_trainX_idx.keys():
            self.plot_modes(ds_idx, max_level, plot_negative=plot_negative)
            self.plot_phase(ds_idx, max_level, plot_negative=False)
    
class HankelDMDAnalysis(DMDAnalysisBase):
    def __init__(self, data_dir=".", save_dir="dmd_output",
                 svd_rank=-1,
                 delay_length=1) -> None:
        super().__init__(data_dir=data_dir, save_dir=save_dir, svd_rank=svd_rank)
        self.delay_length = delay_length
                    
    def fit(self, ds_indices=None):
        self.dmd = HankelDMD(svd_rank=self.svd_rank,d=self.delay_length)
        self.compose_data(ds_indices=ds_indices)
        self.dmd.fit(X=self.train_X.T)
        print("# modes:", self.dmd.modes.shape)
            
    
    def plot_timeseries(self, idx_li):
        pdata = self.dmd.reconstructed_data
        for idx in idx_li:
            fig_name = f"timeseries_{idx}"
            plt.figure(figsize=(6, 4))
            
            cumulative_error = np.sum((pdata[idx, :] - self.train_X[:, idx])**2)
            
            plt.plot(self.train_X[:, idx], alpha=0.6, label="CFD")
            plt.plot(pdata[idx, :], alpha=0.7, label=f"DMD")
            plt.ylabel(f"Pressure for Tap {idx}")
            plt.xlabel("Time Step")
            plt.legend()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            #plt.title(fig_name)
            plt.savefig(os.path.join(self.save_dir, f"0_{fig_name}.png"))
            plt.close()
            
    def plot_rms_timeseries(self):
        pdata = self.dmd.reconstructed_data  # Shape: (num_taps, time)
        original_data = self.train_X.T  # Transpose to match shape (num_taps, time)
    
        # Compute RMS over all pressure taps at each time step (shape: (time,))
        rms_reconstructed = np.sqrt(np.mean(pdata**2, axis=0))
        rms_original = np.sqrt(np.mean(original_data**2, axis=0))
    
        # Create time axis
        time = np.arange(pdata.shape[1])  # Shape: (time,)
    
        # Plot the RMS power over time
        plt.figure(figsize=(6, 4))
        plt.plot(time, rms_original, alpha=0.6, label="CFD", linestyle="-")
        plt.plot(time, rms_reconstructed, alpha=0.7, label="DMD", linestyle="-")
    
        # Labels and title
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        #plt.title("Time-Dependent RMS of Pressure Taps")
        plt.xlabel("Time Step")
        plt.ylabel("RMS Over All Taps")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
    
        # Save the figure
        plt.savefig(os.path.join(self.save_dir, "rms_timeseries.png"))
        plt.close()


    def plot_timeseries_single_mode(self, idx_li, mode_index):
        pdata = self.get_single_mode_reconstruction(mode_index)
        for idx in idx_li:
            fig_name = f"timeseries_{idx}_mode_{mode_index}"
            plt.figure(figsize=(12, 8))
            plt.plot(pdata[idx, :], alpha=0.7, label=f"Mode {mode_index}")
            plt.plot(self.train_X[:, idx], alpha=0.6, label="original")
            # plt.ylim([-1.1, 1.1])
            plt.legend()
            plt.title(fig_name)
            plt.savefig(os.path.join(self.save_dir, f"0_{fig_name}.png"))
            plt.close()
    
    def plot_timeseries_multiple_mode(self, idx_li, mode_indices):
        pdata = np.empty(self.get_single_mode_reconstruction(mode_indices[0]).shape, dtype=complex)
        for mode_index in mode_indices:
            pdata += self.get_single_mode_reconstruction(mode_index)
        for idx in idx_li:
            fig_name = f"timeseries_{idx}_mode_{len(mode_indices)}"
            plt.figure(figsize=(12, 8))
            plt.plot(pdata[idx, :], alpha=0.7, label=f"Mode {mode_indices}")
            plt.plot(self.train_X[:, idx], alpha=0.6, label="original")
            # plt.ylim([-1.1, 1.1])
            plt.legend()
            plt.title(fig_name)
            plt.savefig(os.path.join(self.save_dir, f"0_{fig_name}.png"))
            plt.close()
            
    def plot_dynamics(self):
        pattern = os.path.join(self.save_dir, f"1_*_dynamics.png")
        self.clean_up_figures(pattern)
        
        print("plotting dynamics:")
        print("Saving to:", self.save_dir)
        modes = self.get_original_modes()
        dyna = self.dmd.dynamics
        t = self.datasets[0].time_array

        fig_name = "dynamics"
        fig = plt.figure(figsize=(6,4))
        plt.plot(t, dyna.real.T)
        plt.legend(range(dyna.real.shape[0]))
        plt.savefig(os.path.join(self.save_dir, f"1_{fig_name}.png"))
        plt.close(fig)
        plt.cla()
        plt.clf()
        plt.close("all")
        gc.collect()
            
    def save_dmd(self):
        self.dmd.save(os.path.join(self.save_dir, "dmd.pkl"))
        
    def load_dmd(self):
        self.dmd = HankelDMD.load(os.path.join(self.save_dir, "dmd.pkl"))
            
    def clean_up_figures(self, pattern):
        matching_files = glob.glob(pattern)
        print("cleaning up", pattern)
        for file in matching_files:
            os.remove(file)

    def plot_modes(self, ds_idx, plot_negative=False, index=True):
        start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
        coords_array = self.datasets[ds_idx].get_coords()
        n_i = len(np.unique(coords_array[:, 0]))
        n_j = len(np.unique(coords_array[:, 1]))
        name = self.datasets[ds_idx].name
        is_building = self.datasets[ds_idx].is_building
        
        pattern = os.path.join(self.save_dir, f"2_modeshape_*_*_{name}_*Hz.png")
        self.clean_up_figures(pattern)
        
        print("plotting modes:", name)
        print("Saving to:", self.save_dir)
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        energies = self.rank_modes()
        for mode_idx in range(modes.shape[1]):
            Z_all = abs(modes[:, mode_idx])
            modes_select = modes[start_i:end_i, mode_idx].reshape(n_j, n_i)
            X = coords_array[:, 0].reshape(n_j, n_i)[0, :]
            Y = coords_array[:, 1].reshape(n_j, n_i)[:, 0]        
            Z = abs(modes_select)
            vmin = 0
            vmax = Z_all.max()
            cmap="viridis"
            
            if plot_negative:
                phase = np.angle(modes_select)
                Z[phase < 0] = -Z[phase < 0]
                vmin = -vmax
                cmap="RdBu"
            
            freq = np.log(eigs[mode_idx]).imag / (2 * np.pi * self.dt)
            grow = np.log(eigs[mode_idx]).real / self.dt
            energy = energies[mode_idx]
    
            fig = plt.figure(figsize=(6, 4))
            ax = plt.subplot(111)
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            ax.set_title(rf"Mode {mode_idx}, $\omega$ = {freq:.1f} Hz, $g$ = {grow:.2f}")
            ax.set_aspect("equal")
            
            # Adding coordinate index number at each coordinate
            index = False
            if index:
                for i in range(n_j):
                    for j in range(n_i):
                        ax.text(X[j], Y[i], f'{i*n_i + j}', color='black', fontsize=6, ha='center', va='center')
            
            if is_building:
                ax.axvline(x=0.1, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.2, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.3, color='black', linestyle='-', linewidth=2)
    
            plt.savefig(os.path.join(self.save_dir, f"2_modeshape_{mode_idx}_{name}_{freq:.1f}Hz.png"))
            plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
                
    def plot_phase(self, ds_idx, plot_negative=False):
        start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
        coords_array = self.datasets[ds_idx].get_coords()
        n_i = len(np.unique(coords_array[:, 0]))
        n_j = len(np.unique(coords_array[:, 1]))
        name = self.datasets[ds_idx].name
        is_building = self.datasets[ds_idx].is_building
        
        pattern = os.path.join(self.save_dir, f"2_modeshape_*_*_{name}_*Hz_phase.png")
        self.clean_up_figures(pattern)
        print("plotting phases:", name)
        print("Saving to:", self.save_dir)
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        
        for mode_idx in range(modes.shape[1]):
            modes_select = modes[start_i:end_i, mode_idx].reshape(n_j, n_i)
            X = coords_array[:, 0].reshape(n_j, n_i)[0, :]
            Y = coords_array[:, 1].reshape(n_j, n_i)[:, 0]        
            Z = np.angle(modes_select)
            
            vmin = -np.pi
            vmax = np.pi
            # cmap="twilight"
            cmap="hsv"
            
            # if plot_negative:
            #     Z[Z < 0] = -Z[Z < 0]
            #     Z[Z > np.pi] = Z[Z > np.pi] - np.pi
            #     cmap="hsv"
            #     vmin = 0
            
            freq = np.log(eigs[mode_idx]).imag / (2 * np.pi * self.dt)
            grow = eigs[mode_idx].real

            fig = plt.figure(figsize=(6, 4))
            ax = plt.subplot(111)
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            ax.set_title(rf"Phase Mode {mode_idx}, $\omega$ = {freq:.1f} Hz, $g$ = {grow:.2f}")
            ax.set_aspect("equal")
            
            if is_building:
                ax.axvline(x=0.1, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.2, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.3, color='black', linestyle='-', linewidth=2)

            plt.savefig(os.path.join(save_dir, f"2_modeshape_{mode_idx}_{name}_{freq:.1f}Hz_phase.png"))
            plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
                
    def plot_all_ds(self, max_level=-1, plot_negative=False):
        for ds_idx in self.ds_idx_to_trainX_idx.keys():
            self.plot_modes(ds_idx, plot_negative=plot_negative)
            self.plot_phase(ds_idx, plot_negative=False)
            
    def plot_amplitude_frequency(self):
        pattern = os.path.join(self.save_dir, "amplitude_frequency.png")
        self.clean_up_figures(pattern)
        
        csv_path = os.path.join(self.save_dir, "mode_data.csv")
        
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        mode_amplitudes = self.dmd.amplitudes
        
        df = pd.DataFrame({
            'Mode Number': np.arange(0, len(mode_frequencies)),
            'Frequency (Hz)': mode_frequencies,
            'Amplitude': np.abs(mode_amplitudes)
        })
        df.to_csv(csv_path, index=False)
        
        # Plot the amplitude vs frequency for each mode
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(mode_frequencies)):
            frequency = mode_frequencies[i]
            if frequency > 0:  # Exclude negative frequencies
                sc = ax.scatter(frequency,
                                np.abs(mode_amplitudes[i]),
                                c=i+1, cmap='viridis', vmin=0, vmax=200, label=f"Mode {i+1}", s=50)
                ax.text(frequency,
                        np.abs(mode_amplitudes[i]),
                        str(i), ha='right', va='bottom', fontsize = 16)
        
        # Set the plot title and axis labels
        #ax.set_title("DMD Mode Amplitudes vs Frequencies")
        ax.set_xlabel("Frequency (Hz)", fontsize = 20)
        ax.set_ylabel("Amplitude", fontsize = 20)
        ax.set_xlim(0)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Add a colorbar to the plot
        norm = mcolors.Normalize(vmin=0, vmax=len(mode_frequencies))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label("Mode Number", fontsize = 20)
        cbar.ax.tick_params(labelsize=16)
        
        plt.savefig(os.path.join(self.save_dir, "amplitude_frequency.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
        
    def get_original_modes(self):      
        return self.dmd.modes[:self.dmd.modes.shape[0] // self.delay_length,:]
    
    def get_denormalized_modes(self):
        modes = self.get_original_modes()
        
        for ds_idx in self.ds_idx_to_trainX_idx:
            start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
            modes[start_i:end_i] *= self.datasets[ds_idx].scaler.scale_
        
        return modes
    
    def get_single_mode_reconstruction(self, mode_index):
        modes = self.get_original_modes()  # Spatial modes (phi)
        selected_mode = modes[:, mode_index]
        selected_dynamics = self.dmd.dynamics[mode_index]
        eigenvalue = self.dmd.eigs[mode_index]
    
        # Incorporate the eigenvalue scaling
        temporal_evolution = eigenvalue * selected_dynamics
        return np.outer(selected_mode, temporal_evolution)

    def calculate_agg_power(self, mode_index):
        # Get the mode reconstruction matrix (rows: pressure taps, columns: timesteps)
        reconstruction = self.get_single_mode_reconstruction(mode_index)
        
        # Compute the standard deviation along the time axis (axis=1 for rows)
        std_devs = np.std(reconstruction, axis=1)
        
        # Compute the RMS of the standard deviations
        agg_power = np.sqrt(np.mean(std_devs**2))
        
        return agg_power
    
    def calculate_agg_power_original_data(self):
        orig_data = self.train_X.T  # Assuming self.train_X has shape (timesteps, pressure taps)
        # Compute the standard deviation across time (axis=1 means across columns/timesteps)
        std_per_tap = np.std(orig_data, axis=1)
    
        # Compute the RMS across all pressure taps
        agg_power = np.sqrt(np.mean(std_per_tap**2))
    
        return agg_power
    
    def categorize_and_compute_power(self):
        # Extract the frequencies of the modes from self.dmd
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
    
        # Initialize lists to hold the aggregate powers for each bin
        bin_0_5hz = []
        bin_5_20hz = []
        bin_20hz_plus = []
        
        # Loop through the modes and their corresponding frequencies
        for mode_index, freq in enumerate(mode_frequencies):
            # Calculate the aggregate power for the current mode
            agg_power = self.calculate_agg_power(mode_index)
            
            # Categorize the mode based on its frequency and add to the appropriate bin
            if 0 <= freq <= 5:
                bin_0_5hz.append(agg_power)
            elif 5 < freq <= 20:
                bin_5_20hz.append(agg_power)
            elif freq > 20:
                bin_20hz_plus.append(agg_power)
        
        # Calculate the RMS of the powers in each bin (only if there are powers in the bin)
        def compute_rms(power_list):
            if power_list:  # Ensure the list is not empty
                return np.sqrt(np.sum(np.array(power_list)**2))
            else:
                return 0  # Return 0 if the bin is empty
        
        rms_0_5hz = compute_rms(bin_0_5hz)
        rms_5_20hz = compute_rms(bin_5_20hz)
        rms_20hz_plus = compute_rms(bin_20hz_plus)
        
        # Create a bar graph to show the power for each bin
        bins = ['0-5 Hz', '5-20 Hz', '20+ Hz']
        rms_values = [rms_0_5hz, rms_5_20hz, rms_20hz_plus]
    
        plt.figure(figsize=(8, 6))
        plt.bar(bins, rms_values, color=['blue', 'orange', 'green'])
        plt.title('RMS Power in Frequency Bins')
        plt.xlabel('Frequency Range (Hz)')
        plt.ylabel('RMS Power')
        
        # Save the figure as a PNG
        image_path = os.path.join(self.save_dir, 'frequency_bin_rms_power.png') 
        plt.savefig(image_path)
        plt.close()
        
        return rms_0_5hz, rms_5_20hz, rms_20hz_plus
        

    def plot_single_mode_reconstruction(self, ds_idx, mode_index, plot_negative=False):
        start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
        coords_array = self.datasets[ds_idx].get_coords()
        n_i = len(np.unique(coords_array[:, 0]))
        n_j = len(np.unique(coords_array[:, 1]))
        name = self.datasets[ds_idx].name
        is_building = self.datasets[ds_idx].is_building
        pattern = os.path.join(self.save_dir, f"2_modeshape_*_{name}_*.png")
        self.clean_up_figures(pattern)
        
        print("plotting mode:", name, mode_index)
        print("Saving to:", self.save_dir)
        mode = self.get_single_mode_reconstruction(mode_index)
        image_list = []
        vmax = 2 * np.max(np.abs(mode[start_i:end_i, :].real))
        vmin = -vmax
        for snapshot in range(mode.shape[1]):
            Z_all = abs(mode[:, snapshot])
            mode_select = mode[start_i:end_i, snapshot].reshape(n_j, n_i)
            X = coords_array[:, 0].reshape(n_j, n_i)[0, :]
            Y = coords_array[:, 1].reshape(n_j, n_i)[:, 0]        
            Z = 2 * mode_select.real
            
            cmap="viridis"
            
 #           if plot_negative:
  #              phase = np.angle(mode_select)
   #             Z[phase < 0] = -Z[phase < 0]
    #            # Z[phase > np.pi] = -Z[phase > np.pi]
     #           vmin = -vmax
      #          cmap="RdBu"
            # Z = np.linalg.norm(pmodes_select)
            

            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(111)
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            ax.set_title(f"mode:{mode_index}, {name}")
            ax.set_aspect("equal")
            
            if is_building:
                ax.axvline(x=0.1, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.2, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.3, color='black', linestyle='-', linewidth=2)
            image_list.append(os.path.join(save_dir, f"2_modeshape_{mode_index}_{name}_{snapshot}.png"))
            plt.savefig(os.path.join(save_dir, f"2_modeshape_{mode_index}_{name}_{snapshot}.png"))
            plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
        video_name = os.path.join(self.save_dir, f"mode_{mode_index}_{name}.avi")
        frame = cv2.imread(os.path.join(self.save_dir, image_list[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 24, (width,height))
        for image in image_list:
            video.write(cv2.imread(os.path.join(self.save_dir, image)))
        cv2.destroyAllWindows()
        video.release()


    def plot_multiple_mode_reconstruction(self, ds_idx, mode_indices, plot_negative=False):
        start_i, end_i = self.ds_idx_to_trainX_idx[ds_idx]
        coords_array = self.datasets[ds_idx].get_coords()
        n_i = len(np.unique(coords_array[:, 0]))
        n_j = len(np.unique(coords_array[:, 1]))
        name = self.datasets[ds_idx].name
        is_building = self.datasets[ds_idx].is_building
        pattern = os.path.join(self.save_dir, f"2_modeshape_*_{name}_*.png")
        self.clean_up_figures(pattern)
        
        print("plotting modes:", name, mode_indices)
        print("Saving to:", self.save_dir)
        summed_mode = np.empty(self.get_single_mode_reconstruction(mode_indices[0]).shape, dtype=complex)
        for idx in mode_indices:
            summed_mode += self.get_single_mode_reconstruction(idx)
        image_list = []
        vmax = 2 * np.max(np.abs(summed_mode[start_i:end_i, :].real))
        vmin = -vmax
        for snapshot in range(summed_mode.shape[1]):
            Z_all = abs(summed_mode[:, snapshot])
            mode_select = summed_mode[start_i:end_i, snapshot].reshape(n_j, n_i)
            X = coords_array[:, 0].reshape(n_j, n_i)[0, :]
            Y = coords_array[:, 1].reshape(n_j, n_i)[:, 0]        
            Z = 2 * mode_select.real
            
            cmap="viridis"
            
 #           if plot_negative:
  #              phase = np.angle(mode_select)
   #             Z[phase < 0] = -Z[phase < 0]
    #            # Z[phase > np.pi] = -Z[phase > np.pi]
     #           vmin = -vmax
      #          cmap="RdBu"
            # Z = np.linalg.norm(pmodes_select)
            

            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(111)
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            ax.set_title(f"modes:{mode_indices}, {name}")
            ax.set_aspect("equal")
            
            if is_building:
                ax.axvline(x=0.1, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.2, color='black', linestyle='-', linewidth=2)
                ax.axvline(x=0.3, color='black', linestyle='-', linewidth=2)
            image_list.append(os.path.join(save_dir, f"2_modeshape_{mode_indices}_{name}_{snapshot}.png"))
            plt.savefig(os.path.join(save_dir, f"2_modeshape_{mode_indices}_{name}_{snapshot}.png"))
            plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()
        video_name = os.path.join(self.save_dir, f"mode_{mode_indices}_{name}.avi")
        frame = cv2.imread(os.path.join(self.save_dir, image_list[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(video_name, 0, 24, (width,height))
        for image in image_list:
            video.write(cv2.imread(os.path.join(self.save_dir, image)))
        cv2.destroyAllWindows()
        video.release()
    
    def single_cycle(self, mode_idx):
        
        frequency = np.log(self.dmd.eigs[mode_idx]).imag / (2 * np.pi * self.dt)
        mode = self.get_single_mode_reconstruction(mode_idx)
        #mode2 = self.get_single_mode_reconstruction(41)
       # mode3 = self.get_single_mode_reconstruction(43)
        #mode = mode + mode2 + mode3
        # Calculate number of timesteps per cycle
        timesteps_per_cycle = int(1 / frequency / self.dt)
        
        # Slice the first cycle, TODO: move slice to where pressure tap is at peak
        single_cycle = mode[:, :timesteps_per_cycle]
        return single_cycle
    
    def plot_reconstructed_streamplot(self, u_ds_idx, v_ds_idx, p_ds_idx, mode_index):
        u_start_i, u_end_i = self.ds_idx_to_trainX_idx[u_ds_idx]
        v_start_i, v_end_i = self.ds_idx_to_trainX_idx[v_ds_idx]
        p_start_i, p_end_i = self.ds_idx_to_trainX_idx[p_ds_idx]  

        coords_array = self.datasets[u_ds_idx].get_coords()
        name = self.datasets[u_ds_idx].name
        pattern = os.path.join(self.save_dir, f"2_streamplot_*_{name}_*.png")
        self.clean_up_figures(pattern)
        
        print("plotting streamplot:", name, mode_index)
        print("Saving to:", self.save_dir)
        mode = self.single_cycle(mode_index)
        
        x = np.unique(coords_array[:, 0])
        y = np.unique(coords_array[:, 1])
        x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
        image_list = []
        
        vmax = 2 * np.max(np.abs(mode[p_start_i:p_end_i, :].real))
        vmin = -vmax
        pressure_levels = np.linspace(vmin, vmax, 100)
        
        for snapshot in range(mode.shape[1]):
            U = 2 * mode[u_start_i:u_end_i, snapshot].reshape(-1).real
            V = 2 * mode[v_start_i:v_end_i, snapshot].reshape(-1).real 
            P = 2 * mode[p_start_i:p_end_i, snapshot].reshape(-1).real 
            
            u_interp = griddata((coords_array[:, 0], coords_array[:, 1]), U, (x_grid, y_grid), method='cubic')
            v_interp = griddata((coords_array[:, 0], coords_array[:, 1]), V, (x_grid, y_grid), method='cubic')
            p_interp = griddata((coords_array[:, 0], coords_array[:, 1]), P, (x_grid, y_grid), method='cubic')
            
            fig, ax = plt.subplots(figsize=(8, 6)) 

            # Plot the pressure contour first
            pressure_contour = ax.contourf(x_grid, y_grid, p_interp, levels=pressure_levels, cmap='coolwarm', alpha=0.5, vmin = vmin, vmax = vmax)
            cbar = plt.colorbar(pressure_contour)

            strm = ax.streamplot(x_grid, y_grid, u_interp, v_interp, density=[2,2], linewidth=0.75) #higher density = more lines
            ax.set_title(f"Mode: {mode_index}, {name}")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_aspect("equal")
            
            image_path = os.path.join(self.save_dir, f"2_streamplot_{mode_index}_{name}_{snapshot}.png") 
            plt.savefig(image_path)
            image_list.append(image_path)  
            plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()

    def plot_full_streamplot(self, u_ds_indices, v_ds_indices, p_ds_indices, mode_index):
        #u_start_i, u_end_i = self.ds_idx_to_trainX_idx[u_ds_idx]
        #v_start_i, v_end_i = self.ds_idx_to_trainX_idx[v_ds_idx]
        #p_start_i, p_end_i = self.ds_idx_to_trainX_idx[p_ds_idx]  
        u_start_indices = []
        v_start_indices = []
        p_start_indices = []
        u_end_indices = []
        v_end_indices = []
        p_end_indices = []
        coords_arrays = []
        x_grids = []
        y_grids = []
        for u_ds_idx, v_ds_idx, p_ds_idx in zip(u_ds_indices, v_ds_indices, p_ds_indices):
            coords_arrays.append(self.datasets[u_ds_idx].get_coords())
            u_start_i, u_end_i = self.ds_idx_to_trainX_idx[u_ds_idx]
            v_start_i, v_end_i = self.ds_idx_to_trainX_idx[v_ds_idx]
            p_start_i, p_end_i = self.ds_idx_to_trainX_idx[p_ds_idx]
            u_start_indices.append(u_start_i)
            v_start_indices.append(v_start_i)
            p_start_indices.append(p_start_i)
            u_end_indices.append(u_end_i)
            v_end_indices.append(v_end_i)
            p_end_indices.append(p_end_i)
            x = np.unique(coords_arrays[-1][:, 0])
            y = np.unique(coords_arrays[-1][:, 1])
            x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
            x_grids.append(x_grid)
            y_grids.append(y_grid)
        name = self.datasets[u_ds_idx].name
        pattern = os.path.join(self.save_dir, f"2_streamplot_*_{name}_*.png")
        self.clean_up_figures(pattern)
        
        print("plotting streamplot:", name, mode_index)
        print("Saving to:", self.save_dir)
        mode = self.single_cycle(mode_index)
        
        image_list = []
        

        
        combined_coords_array = np.concatenate(coords_arrays)
        x = combined_coords_array[:,0]
        y = combined_coords_array[:,1]
        p_only = None
        for p_start_i, p_end_i in zip(p_start_indices, p_end_indices):
            if p_only is None:
                p_only = mode[p_start_i:p_end_i, :]
            else:
                p_only = np.vstack((p_only, mode[p_start_i:p_end_i, :]))
        
        vmax = 2 * np.max(np.abs(p_only)) 
        vmin = -vmax
        pressure_levels = np.linspace(vmin, vmax, 100)
        cbar_created = False
        for snapshot in range(mode.shape[1]):
            fig, ax = plt.subplots(figsize=(8, 6))
            cbar_created = False
            for i in range(len(u_ds_indices)):
                U = 2 * mode[u_start_indices[i]:u_end_indices[i], snapshot].reshape(-1).real
                V = 2 * mode[v_start_indices[i]:v_end_indices[i], snapshot].reshape(-1).real 
                P = 2 * mode[p_start_indices[i]:p_end_indices[i], snapshot].reshape(-1).real 
                u_interp = griddata((coords_arrays[i][:, 0], coords_arrays[i][:, 1]), U, (x_grids[i], y_grids[i]), method='cubic')
                v_interp = griddata((coords_arrays[i][:, 0], coords_arrays[i][:, 1]), V, (x_grids[i], y_grids[i]), method='cubic')
                p_interp = griddata((coords_arrays[i][:, 0], coords_arrays[i][:, 1]), P, (x_grids[i], y_grids[i]), method='cubic')
                
                #fig, ax = plt.subplots(figsize=(8, 6)) 
    
                # Plot the pressure contour first
                pressure_contour = ax.contourf(x_grids[i], y_grids[i], p_interp, levels=pressure_levels, cmap='coolwarm', alpha=0.5, vmin = vmin, vmax = vmax)
                if not cbar_created:
                    cbar = plt.colorbar(pressure_contour)
                    cbar_created = True
    
                strm = ax.streamplot(x_grids[i], y_grids[i], u_interp, v_interp, density=[2,2], linewidth=0.75, color='black', arrowsize=0) #higher density = more lines
            #ax.set_title(f"Mode: {mode_index}")
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_aspect("equal")
            cbar.ax.tick_params(labelsize=20)
            image_path = os.path.join(self.save_dir, f"2_streamplot_{mode_index}_{name}_{snapshot}.png") 
            plt.savefig(image_path)
            image_list.append(image_path)
            video_name = os.path.join(self.save_dir, f"mode_{mode_index}_streamplot.avi")
            frame = cv2.imread(os.path.join(self.save_dir, image_list[0]))
            height, width, layers = frame.shape
            video = cv2.VideoWriter(video_name, 0, 24, (width,height))
            for image in image_list:
                video.write(cv2.imread(os.path.join(self.save_dir, image)))
            cv2.destroyAllWindows()
            video.release()
            plt.close(fig)
            plt.cla()
            plt.clf()
            plt.close("all")
            gc.collect()

    def rank_modes_old(self):
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        amplitudes = self.dmd.amplitudes
        x2 = self.datasets[0].data_array.shape[0] * self.dt  # Upper bound
        #x1 = x2 / 2  # Lower bound
        x1 = 0
        #x2 = 5
        energies = []
        for mode_idx in range(modes.shape[1]):
            a = np.abs(amplitudes[mode_idx])  # Amplitude
            w = np.log(eigs[mode_idx]).imag / (self.dt)  # Frequency
            g = np.log(eigs[mode_idx]).real / (self.dt)  # Gamma
            
            def E(t):
                return (a**2) * np.exp(2 * g * t) * np.sum((abs(modes[:, mode_idx]))**2)
            
            energy, error = quad(E, x1, x2) 
            energies.append(energy)
            
        #plt.figure(figsize=(10, 6))
        #plt.plot(range(modes.shape[1]), energies, 'o-', label='Energy')
        #plt.xlabel('Mode Index')
        #plt.ylabel('Energy')
        #plt.title('Energy vs Mode Index')
        #plt.xticks(range(modes.shape[1]))
        #plt.grid(True)
        #plt.legend()
        #plt.savefig(os.path.join(self.save_dir, "energies.png"))
        
        return energies
    
    def rank_modes(self):
        """
        Computes the energy contributions of DMD modes in their original order.
    
        Parameters:
        -----------
    
        Returns:
        --------
        contributions : list
            Contributions of the DMD modes in their original order.
        """
        # Retrieve modes, eigenvalues, and amplitudes
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        amplitudes = self.dmd.amplitudes
        N = self.dmd.dynamics.shape[1]  # Number of time steps
    
        # Compute the Frobenius norm of each mode (spatial structure)
        phi_norms_squared = np.linalg.norm(modes, axis=0)**2
    
        # Calculate the contribution of each mode
        contributions = []
        for j, (alpha_j, mu_j) in enumerate(zip(amplitudes, eigs)):
            # Time evolution factor
            time_evolution = sum(abs(alpha_j * (mu_j**(i - 1))) for i in range(1, N + 1))
            # Contribution
            contribution = time_evolution * phi_norms_squared[j] * self.dt
            contributions.append(contribution)
    
        return contributions

    def rank_modes_separate(self):
        """
        Computes the energy contributions of DMD modes for pressure and velocity components separately.
    
        Parameters:
        -----------
    
        Returns:
        --------
        pressure_contributions : list
            Contributions of the DMD modes from the pressure component.
        velocity_contributions : list
            Contributions of the DMD modes from the velocity component.
        """
        # Retrieve modes, eigenvalues, and amplitudes
        modes = self.get_original_modes()
    
        eigs = self.dmd.eigs
        amplitudes = self.dmd.amplitudes
        N = self.dmd.dynamics.shape[1]  # Number of time steps
    
        # Split modes into pressure (first 500) and velocity (last 450)
        pressure_modes = modes[:500, :]
        velocity_modes = modes[500:, :]
    
        # Compute the Frobenius norm squared for each mode, separately for pressure and velocity
        pressure_norms_squared = np.linalg.norm(pressure_modes, axis=0)**2
        velocity_norms_squared = np.linalg.norm(velocity_modes, axis=0)**2
    
        # Calculate the contribution of each mode
        pressure_contributions = []
        velocity_contributions = []
    
        for j, (alpha_j, mu_j) in enumerate(zip(amplitudes, eigs)):
            # Time evolution factor
            time_evolution = sum(abs(alpha_j * (mu_j**(i - 1))) for i in range(1, N + 1))
    
            # Separate contributions for pressure and velocity
            pressure_contribution = time_evolution * pressure_norms_squared[j] * self.dt
            velocity_contribution = time_evolution * velocity_norms_squared[j] * self.dt
    
            pressure_contributions.append(pressure_contribution)
            velocity_contributions.append(velocity_contribution)
    
        return pressure_contributions, velocity_contributions

    def rank_modes_1(self):
        """
        Computes the energy contributions of DMD modes using the given formula, in their original order.
    
        Parameters:
        -----------
        delta_t : float, optional
            Time step size of the dataset. Default is 0.1.
    
        Returns:
        --------
        contributions : list
            Contributions of the DMD modes in their original order.
        """
        # Retrieve modes, eigenvalues, and dynamics
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        N = self.dmd.dynamics.shape[1]  # Number of time steps
    
        # Compute the Frobenius norm of each mode (spatial structure)
        phi_norms_squared = np.linalg.norm(modes, axis=0)**2
    
        # Calculate the contribution of each mode
        contributions = []
        for j, lambda_j in enumerate(eigs):
            # Time evolution factor
            time_evolution = sum(abs(lambda_j**(i - 1)) for i in range(1, N + 1))
            # Contribution
            contribution = time_evolution * phi_norms_squared[j] * self.dt
            contributions.append(contribution)
    
        return contributions

    def rank_modes_over_time(self):
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        amplitudes = self.dmd.amplitudes
        energies_per_mode = []
        max_time = self.datasets[0].data_array.shape[0] * self.dt  # Maximum upper bound
        x1 = max_time / 2
        #max_time = 5
    
        for mode_idx in range(modes.shape[1]):
            a = np.abs(amplitudes[mode_idx])  # Amplitude
            w = np.log(eigs[mode_idx]).imag / (self.dt)  # Frequency
            g = np.log(eigs[mode_idx]).real / (2 * np.pi * self.dt)  # Gamma
            
            def E(t):
                return 0.5 * (a**2) * np.exp(2 * g * t) * np.sum((abs(modes[:, mode_idx]))**2)
    
            # Store energy values for increasing upper bounds
            energies = []
            upper_bounds = np.linspace(x1, max_time, num=100)
            
            for x2 in upper_bounds:
                energy, error = quad(E, x1, x2)
                energies.append(energy)
            
            energies_per_mode.append(energies)
            
            # Plot the energies as a function of upper bounds for this mode
            plt.figure(figsize=(10, 6))
            plt.plot(upper_bounds, energies, label=f'Mode {mode_idx}')
            plt.xlabel('Upper Bound of Integration')
            plt.ylabel('Energy')
            plt.title(f'Energy vs Upper Bound of Integration for Mode {mode_idx}')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, f"energy_mode_{mode_idx}.png"))
            
        return energies_per_mode

    def rank_modes_per_tap(self, tap):
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        amplitudes = self.dmd.amplitudes
        
        max_time = self.datasets[0].data_array.shape[0] * self.dt  # Maximum upper bound
        x1 = max_time / 2
        energies = []
        for mode_idx in range(modes.shape[1]):
            a = np.abs(amplitudes[mode_idx])  # Amplitude
            w = np.log(eigs[mode_idx]).imag / (self.dt)  # Frequency
            g = np.log(eigs[mode_idx]).real / (2 * np.pi * self.dt)  # Gamma
            
            def E(t):
                return 0.5 * (a**2) * np.exp(2 * g * t) * np.sum((abs(modes[tap, mode_idx]))**2)
    
            
            energy, error = quad(E, x1, max_time) 
            energies.append(energy)
            
        return energies
    
    def reconstruct_high_energy_modes(self, number):
        energies = self.rank_modes()
        modes = self.get_original_modes()
        
        indexed_energies = list(enumerate(energies))
        indexed_energies_sorted = sorted(indexed_energies, key=lambda x: x[1], reverse=True)
        top_energies = indexed_energies_sorted[:number]
        for index, energy in top_energies:
            max_tap = np.argmax(np.abs(modes[:, index]))
            self.plot_timeseries_single_mode([max_tap], index)
            
    def reconstruct_tap(self, index, number_of_modes):
        energies = self.rank_modes_per_tap(index)
        top_modes = np.argsort(energies)[-number_of_modes:][::-1].tolist()
        self.plot_timeseries_multiple_mode([index], top_modes)
        
    def rank_modes_per_tap(self, tap):
        modes = self.get_original_modes()
        eigs = self.dmd.eigs
        amplitudes = self.dmd.amplitudes
        
        max_time = self.datasets[0].data_array.shape[0] * self.dt  # Maximum upper bound
        x1 = max_time / 2
        energies = []
        for mode_idx in range(modes.shape[1]):
            a = np.abs(amplitudes[mode_idx])  # Amplitude
            w = np.log(eigs[mode_idx]).imag / (self.dt)  # Frequency
            g = np.log(eigs[mode_idx]).real / (2 * np.pi * self.dt)  # Gamma
            
            def E(t):
                return 0.5 * (a**2) * np.exp(2 * g * t) * np.sum((abs(modes[tap, mode_idx]))**2)
    
            
            energy, error = quad(E, x1, max_time) 
            energies.append(energy)
            
        return energies
    
    def plot_energy_frequency(self, title, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}.png")
        self.clean_up_figures(pattern)
        
        
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        mode_energies = self.rank_modes()
        
        df = pd.DataFrame({
            'Mode Number': np.arange(1, len(mode_frequencies) + 1),
            'Frequency (Hz)': mode_frequencies,
            'Energy': np.abs(mode_energies)
        })

        # Save the DataFrame to a CSV file
        csv_path = os.path.join(self.save_dir, "energy_frequency_data.csv")
        df.to_csv(csv_path, index=False)
        
        # Plot the amplitude vs frequency for each mode
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(mode_frequencies)):
            frequency = mode_frequencies[i]
            if frequency > 0:  # Exclude negative frequencies
                sc = ax.scatter(frequency,
                                np.abs(mode_energies[i]),
                                c=i+1, cmap='viridis', vmin=0, vmax=200, label=f"Mode {i+1}", s=50)
                ax.text(frequency,
                        np.abs(mode_energies[i]),
                        str(i), ha='right', va='bottom', fontsize = 16)
        
        # Set the plot title and axis labels
        #ax.set_title("DMD Mode Amplitudes vs Frequencies")
        ax.set_xlabel("Frequency (Hz)", fontsize = 20)
        ax.set_ylabel("Energy", fontsize = 20)
        ax.set_xlim(0)
        if xlim != 0:
            ax.set_xlim(0,xlim)
        
        if ylim != 0:
            ax.set_ylim(0,ylim)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Add a colorbar to the plot
        norm = mcolors.Normalize(vmin=0, vmax=len(mode_frequencies))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label("Mode Number", fontsize = 20)
        cbar.ax.tick_params(labelsize=16)
        
        plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()

    def plot_binned_energy_frequency(self, title, bin_width=10, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}_binned_energy.png")
        self.clean_up_figures(pattern)
    
        # Compute mode frequencies and energies
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        mode_energies = np.abs(self.rank_modes())
    
        # Keep only positive frequencies
        valid_indices = mode_frequencies > 0
        mode_frequencies = mode_frequencies[valid_indices]
        mode_energies = mode_energies[valid_indices]
    
        # Bin the frequencies
        min_freq = np.floor(mode_frequencies.min())
        max_freq = np.ceil(mode_frequencies.max())
        bins = np.arange(min_freq, max_freq + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        binned_means = []
        binned_stds = []
    
        for i in range(len(bins) - 1):
            in_bin = (mode_frequencies >= bins[i]) & (mode_frequencies < bins[i + 1])
            energies_in_bin = mode_energies[in_bin]
            if len(energies_in_bin) > 0:
                binned_means.append(np.mean(energies_in_bin))
                binned_stds.append(np.std(energies_in_bin))
            else:
                binned_means.append(0)
                binned_stds.append(0)
    
        # Save data to CSV
        df = pd.DataFrame({
            'Bin Center Frequency (Hz)': bin_centers,
            'Mean Energy': binned_means,
            'Standard Deviation': binned_stds
        })
        csv_path = os.path.join(self.save_dir, "binned_energy_frequency.csv")
        df.to_csv(csv_path, index=False)
    
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(bin_centers, binned_means, yerr=binned_stds, width=bin_width * 0.9,
               align='center', edgecolor='black', capsize=5)
    
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Mean Energy", fontsize=20)
    
        if xlim != 0:
            ax.set_xlim(0, xlim)
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title}_binned_energy.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()

    def plot_binned_stacked_energy(self, title, bin_width=10, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}_stacked_energy.png")
        self.clean_up_figures(pattern)
    
        # Compute frequencies
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        pressure_contributions, velocity_contributions = self.rank_modes_separate()
    
        # Filter positive frequencies
        valid_indices = mode_frequencies > 0
        mode_frequencies = mode_frequencies[valid_indices]
        pressure_contributions = np.array(pressure_contributions)[valid_indices]
        velocity_contributions = np.array(velocity_contributions)[valid_indices]
    
        # Bin setup
        min_freq = np.floor(mode_frequencies.min())
        max_freq = np.ceil(mode_frequencies.max())
        bins = np.arange(min_freq, max_freq + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        binned_pressure = []
        binned_velocity = []
    
        for i in range(len(bins) - 1):
            in_bin = (mode_frequencies >= bins[i]) & (mode_frequencies < bins[i + 1])
            total_pressure = pressure_contributions[in_bin].sum()
            total_velocity = velocity_contributions[in_bin].sum()
            binned_pressure.append(total_pressure)
            binned_velocity.append(total_velocity)
    
        # Save to CSV
        df = pd.DataFrame({
            'Bin Center Frequency (Hz)': bin_centers,
            'Total Pressure Energy': binned_pressure,
            'Total Velocity Energy': binned_velocity
        })
        csv_path = os.path.join(self.save_dir, "stacked_energy_binned.csv")
        df.to_csv(csv_path, index=False)
    
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(bin_centers, binned_pressure, width=bin_width * 0.9, label='Pressure', align='center', edgecolor='black')
        ax.bar(bin_centers, binned_velocity, width=bin_width * 0.9, bottom=binned_pressure,
               label='Velocity', align='center', edgecolor='black')
    
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Total Energy", fontsize=20)
    
        if xlim != 0:
            ax.set_xlim(0, xlim)
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title}_stacked_energy.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
        
    def plot_pressure_energy_frequency(self, title, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}_pressure_only.png")
        self.clean_up_figures(pattern)
    
        # Compute mode frequencies and pressure energy
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        pressure_contributions, _ = self.rank_modes_separate()
        pressure_contributions = np.abs(pressure_contributions)
    
        # Original mode numbers
        mode_numbers = np.arange(len(mode_frequencies))
    
        # Filter for positive frequencies
        valid_indices = mode_frequencies > 0
        filtered_frequencies = mode_frequencies[valid_indices]
        filtered_energies = pressure_contributions[valid_indices]
        filtered_mode_numbers = mode_numbers[valid_indices]
    
        # Save to CSV
        df = pd.DataFrame({
            'Mode Number': filtered_mode_numbers,
            'Frequency (Hz)': filtered_frequencies,
            'Pressure Energy': filtered_energies
        })
        csv_path = os.path.join(self.save_dir, "pressure_energy_frequency_data.csv")
        df.to_csv(csv_path, index=False)
    
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for freq, energy, mode_num in zip(filtered_frequencies, filtered_energies, filtered_mode_numbers):
            ax.scatter(freq, energy,
                       c=mode_num, cmap='plasma', vmin=0, vmax=len(mode_frequencies), s=50)
            ax.text(freq, energy, str(mode_num), ha='right', va='bottom', fontsize=16)
    
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Pressure Energy", fontsize=20)
    
        if xlim != 0:
            ax.set_xlim(0, xlim)
        else:
            ax.set_xlim(0)
    
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        # Add colorbar
        norm = mcolors.Normalize(vmin=0, vmax=len(mode_frequencies))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
        cbar.set_label("Mode Number", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title}_pressure_only.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()

    def plot_velocity_energy_frequency(self, title, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}_velocity_only.png")
        self.clean_up_figures(pattern)
    
        # Compute frequencies and get only velocity contributions
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        _, velocity_contributions = self.rank_modes_separate()
        velocity_contributions = np.abs(velocity_contributions)
    
        # Original mode numbers
        mode_numbers = np.arange(len(mode_frequencies))
    
        # Filter for positive frequencies
        valid_indices = mode_frequencies > 0
        filtered_frequencies = mode_frequencies[valid_indices]
        filtered_energies = velocity_contributions[valid_indices]
        filtered_mode_numbers = mode_numbers[valid_indices]
    
        # Save to CSV
        df = pd.DataFrame({
            'Mode Number': filtered_mode_numbers,
            'Frequency (Hz)': filtered_frequencies,
            'Velocity Energy': filtered_energies
        })
        csv_path = os.path.join(self.save_dir, "velocity_energy_frequency_data.csv")
        df.to_csv(csv_path, index=False)
    
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for freq, energy, mode_num in zip(filtered_frequencies, filtered_energies, filtered_mode_numbers):
            ax.scatter(freq, energy,
                       c=mode_num, cmap='cividis', vmin=0, vmax=len(mode_frequencies), s=50)
            ax.text(freq, energy, str(mode_num), ha='right', va='bottom', fontsize=16)
    
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Velocity Energy", fontsize=20)
    
        if xlim != 0:
            ax.set_xlim(0, xlim)
        else:
            ax.set_xlim(0)
    
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        # Colorbar
        norm = mcolors.Normalize(vmin=0, vmax=len(mode_frequencies))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cividis'), ax=ax)
        cbar.set_label("Mode Number", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title}_velocity_only.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
    
    def plot_pressure_velocity_ratio(self, title, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}.png")
        self.clean_up_figures(pattern)
    
        # Compute mode frequencies
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
    
        # Get pressure and velocity contributions
        pressure_contributions, velocity_contributions = self.rank_modes_separate()
    
        # Compute the ratio (avoiding division by zero)
        ratio = np.array(pressure_contributions) / np.array(velocity_contributions)
        ratio = np.nan_to_num(ratio, nan=0, posinf=0, neginf=0)  # Handle divide-by-zero issues
    
        # Create a DataFrame for saving
        df = pd.DataFrame({
            'Mode Number': np.arange(1, len(mode_frequencies) + 1),
            'Frequency (Hz)': mode_frequencies,
            'Pressure/Velocity Ratio': ratio
        })
    
        # Save to CSV
        csv_path = os.path.join(self.save_dir, "pressure_velocity_ratio.csv")
        df.to_csv(csv_path, index=False)
    
        # Plot the ratio vs frequency
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(mode_frequencies)):
            frequency = mode_frequencies[i]
            if frequency > 0:  # Exclude negative frequencies
                ax.scatter(frequency, ratio[i], c=i+1, cmap='plasma', vmin=0, vmax=200, s=50)
                ax.text(frequency, ratio[i], str(i), ha='right', va='bottom', fontsize=16)
    
        # Set labels and limits
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Pressure Contribution / Velocity Contribution", fontsize=20)
        if xlim != 0:
            ax.set_xlim(0, xlim)
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        # Add colorbar
        norm = mcolors.Normalize(vmin=0, vmax=len(mode_frequencies))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='plasma'), ax=ax)
        cbar.set_label("Mode Number", fontsize=20)
        cbar.ax.tick_params(labelsize=16)
    
        plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
        
    def plot_binned_pressure_velocity_ratio(self, title, bin_width=10, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}_binned.png")
        self.clean_up_figures(pattern)
    
        # Compute mode frequencies
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
    
        # Get pressure and velocity contributions
        pressure_contributions, velocity_contributions = self.rank_modes_separate()
    
        # Only keep positive frequencies
        valid_indices = mode_frequencies > 0
        mode_frequencies = mode_frequencies[valid_indices]
        pressure_contributions = np.array(pressure_contributions)[valid_indices]
        velocity_contributions = np.array(velocity_contributions)[valid_indices]
    
        # Compute per-mode ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            per_mode_ratios = np.where(velocity_contributions != 0,
                                       pressure_contributions / velocity_contributions,
                                       0)
    
        # Bin the frequencies
        min_freq = np.floor(mode_frequencies.min())
        max_freq = np.ceil(mode_frequencies.max())
        bins = np.arange(min_freq, max_freq + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
        binned_means = []
        binned_stds = []
    
        for i in range(len(bins) - 1):
            in_bin = (mode_frequencies >= bins[i]) & (mode_frequencies < bins[i + 1])
            ratios_in_bin = per_mode_ratios[in_bin]
            if len(ratios_in_bin) > 0:
                binned_means.append(np.mean(ratios_in_bin))
                binned_stds.append(np.std(ratios_in_bin))
            else:
                binned_means.append(0)
                binned_stds.append(0)
    
        # Save to CSV
        df = pd.DataFrame({
            'Bin Center Frequency (Hz)': bin_centers,
            'Binned Pressure/Velocity Ratio Mean': binned_means,
            'Standard Deviation': binned_stds
        })
    
        csv_path = os.path.join(self.save_dir, "binned_pressure_velocity_ratio.csv")
        df.to_csv(csv_path, index=False)
    
        # Plot with error bars
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(bin_centers, binned_means, yerr=binned_stds, width=bin_width * 0.9,
               align='center', edgecolor='black', capsize=5)
    
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Mean Pressure/Velocity Ratio", fontsize=20)
    
        if xlim != 0:
            ax.set_xlim(0, xlim)
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title}_binned.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
    
    def plot_cumulative_energy(self, title, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}.png")
        self.clean_up_figures(pattern)
        
        # Calculate mode frequencies and energies
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        mode_energies = self.rank_modes()
    
        # Create a DataFrame and sort by frequency
        df = pd.DataFrame({
            'Frequency (Hz)': mode_frequencies,
            'Energy': np.abs(mode_energies)
        }).sort_values(by='Frequency (Hz)').reset_index(drop=True)
    
        # Filter out negative frequencies
        df = df[df['Frequency (Hz)'] > 0]
    
        # Calculate cumulative energy
        df['Cumulative Energy'] = df['Energy'].cumsum()
    
        # Save the DataFrame to a CSV file
        csv_path = os.path.join(self.save_dir, "cumulative_energy_data.csv")
        df.to_csv(csv_path, index=False)
        
        # Plot cumulative energy vs frequency
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df['Frequency (Hz)'], df['Cumulative Energy'], color='blue', lw=2)
    
        # Set the plot title and axis labels
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Cumulative Energy", fontsize=20)
        ax.set_xlim(0)
        if xlim != 0:
            ax.set_xlim(0, xlim)
        
        if ylim != 0:
            ax.set_ylim(0, ylim)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()


    def plot_combined_energy_frequency(self, title, csv_path1, csv_path2, xlim_max=None, ylim_max=None):
        # Read the data from the two CSV files
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        
        # Filter out rows with negative frequencies without changing the indexing
        df1 = df1[df1['Frequency (Hz)'] > 0]
        df2 = df2[df2['Frequency (Hz)'] > 0]
        
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the first dataset
        sc1 = ax.scatter(df1['Frequency (Hz)'], df1['Energy'], 
                         color='blue', label='WT', s=50)
        
        # Annotate points with their original mode numbers for the first dataset
        #for i, row in df1.iterrows():
        #    mode_number = int(row['Mode Number']) - 1  # Keep the original mode number
        
        # Plot the second dataset
        sc2 = ax.scatter(df2['Frequency (Hz)'], df2['Energy'], 
                         color='orange', label='CFD', s=50)
        
        # Annotate points with their original mode numbers for the second dataset
        #for i, row in df2.iterrows():
        #    mode_number = int(row['Mode Number']) - 1  # Keep the original mode number
        
        # Set the plot title and axis labels
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Energy", fontsize=20)
        
        # Set the x and y limits based on provided values or auto-calculated ones
        xlim = xlim_max if xlim_max is not None else max(df1['Frequency (Hz)'].max(), df2['Frequency (Hz)'].max())
        ylim = ylim_max if ylim_max is not None else max(df1['Energy'].max(), df2['Energy'].max())
        
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        
        # Add a legend to distinguish between the datasets
        ax.legend(fontsize=16)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Save the plot as a PNG file
        plot_path = os.path.join(self.save_dir, f"{title}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        
    def plot_combined_amplitude_frequency(self, title, csv_path1, csv_path2, xlim_max=None, ylim_max=None):
        # Read the data from the two CSV files
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
        
        # Filter out rows with negative frequencies without changing the indexing
        df1 = df1[df1['Frequency (Hz)'] > 0]
        df2 = df2[df2['Frequency (Hz)'] > 0]
        
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the first dataset
        sc1 = ax.scatter(df1['Frequency (Hz)'], df1['Amplitude'], 
                         color='blue', label='WT', s=25, alpha=0.7)
        
        # Annotate points with their original mode numbers for the first dataset
        #for i, row in df1.iterrows():
        #    mode_number = int(row['Mode Number'])  # Keep the original mode number
        
        # Plot the second dataset
        sc2 = ax.scatter(df2['Frequency (Hz)'], df2['Amplitude'], 
                         color='orange', label='CFD', s=25, alpha=0.7)
        
        # Annotate points with their original mode numbers for the second dataset
        #for i, row in df2.iterrows():
        #    mode_number = int(row['Mode Number'])  # Keep the original mode number
        
        # Set the plot title and axis labels
        ax.set_xlabel("Frequency (Hz)", fontsize=20)
        ax.set_ylabel("Amplitude", fontsize=20)
        ax.grid(True, linestyle="--", alpha=0.7)
        
        # Set the x and y limits based on provided values or auto-calculated ones
        xlim = xlim_max if xlim_max is not None else max(df1['Frequency (Hz)'].max(), df2['Frequency (Hz)'].max())
        ylim = ylim_max if ylim_max is not None else max(df1['Amplitude'].max(), df2['Amplitude'].max())
        
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, 200)
        
        # Add a legend to distinguish between the datasets
        ax.legend(fontsize=16)
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        
        # Save the plot as a PNG file
        plot_path = os.path.join(self.save_dir, f"{title}.png")
        plt.savefig(plot_path)
        plt.close(fig)

    def plot_summed_energy_groups(self, title, xlim=0, ylim=0):
        pattern = os.path.join(self.save_dir, f"{title}.png")
        self.clean_up_figures(pattern)
    
        # Calculate mode frequencies and energies
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        mode_energies = self.rank_modes()
    
        # Create a DataFrame and filter for positive frequencies
        df = pd.DataFrame({
            'Frequency (Hz)': mode_frequencies,
            'Energy': np.abs(mode_energies)
        }).sort_values(by='Frequency (Hz)').reset_index(drop=True)
        
        # Filter out negative frequencies
        df = df[df['Frequency (Hz)'] > 0]
    
        # Define specific bins for frequency groups: 0-1 Hz, 1-20 Hz, and 20+ Hz
        bins = [0, 5, 20, df['Frequency (Hz)'].max() + 1]
        labels = ['0-5 Hz', '5-20 Hz', '20+ Hz']
        
        # Bin the frequencies into the specified groups
        df['Group'] = pd.cut(df['Frequency (Hz)'], bins=bins, labels=labels, right=False)
    
        # Calculate the summed energy for each group
        summed_energy = df.groupby('Group')['Energy'].sum().reset_index()
    
        # Save the DataFrame to a CSV file
        csv_path = os.path.join(self.save_dir, "summed_energy_groups.csv")
        summed_energy.to_csv(csv_path, index=False)
    
        # Plot summed energy for each group
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(summed_energy['Group'], summed_energy['Energy'], color='orange')
    
        # Set the plot title and axis labels
        ax.set_title("Summed Energy for Each Frequency Group", fontsize=20)
        ax.set_xlabel("Frequency Group", fontsize=20)
        ax.set_ylabel("Summed Energy", fontsize=20)
    
        if ylim != 0:
            ax.set_ylim(0, ylim)
    
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
        # Save the plot
        plt.savefig(os.path.join(self.save_dir, f"{title}.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
        
    def plot_summed_energy_groups_comparison(self, title, xlim=0, ylim=0):
        """
        Plots summed energy groups for each ranking method with separate y-axis scales, arranged side by side.
        
        Parameters:
        -----------
        title : str
            Title of the plot and file name prefix.
        xlim : float, optional
            X-axis limit (default is 0 for no limit).
        ylim : float, optional
            Y-axis limit (applies globally if set; default is 0 for individual scales).
        """
        pattern = os.path.join(self.save_dir, f"{title}.png")
        self.clean_up_figures(pattern)
    
        bins = [0, 5, 20, None]  # Frequency bins
        labels = ['0-5 Hz', '5-20 Hz', '20+ Hz']
    
        results = {}  # Store results for each method
        for method in ["rank_modes", "rank_modes_1", "rank_modes_old"]:
            if hasattr(self, method):
                # Calculate mode frequencies and energies
                mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
                mode_energies = getattr(self, method)()
    
                # Create a DataFrame and filter for positive frequencies
                df = pd.DataFrame({
                    'Frequency (Hz)': mode_frequencies,
                    'Energy': np.abs(mode_energies)
                }).sort_values(by='Frequency (Hz)').reset_index(drop=True)
                df = df[df['Frequency (Hz)'] > 0]
    
                # Bin the frequencies into groups
                df['Group'] = pd.cut(
                    df['Frequency (Hz)'],
                    bins=[0, 5, 20, df['Frequency (Hz)'].max() + 1],
                    labels=labels,
                    right=False
                )
    
                # Calculate summed energy for each group
                summed_energy = df.groupby('Group')['Energy'].sum().reset_index()
                results[method] = summed_energy.set_index('Group')['Energy']
    
        # Combine results into a DataFrame for saving
        combined_results = pd.DataFrame(results).fillna(0)
    
        # Save the DataFrame to a CSV file
        csv_path = os.path.join(self.save_dir, "summed_energy_groups_comparison.csv")
        combined_results.to_csv(csv_path, index_label="Frequency Group")
        print(f"Summed energy groups comparison saved to {csv_path}")
    
        # Plot the results side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        width = 0.5  # Width of each bar
    
        for i, (method, ax) in enumerate(zip(results.keys(), axes)):
            # Bar plot for each ranking method
            ax.bar(labels, combined_results[method], width, label=method, color=f"C{i}")
            ax.set_title(method, fontsize=14)
            ax.set_xlabel("Frequency Group", fontsize=12)
            if i == 0:
                ax.set_ylabel("Summed Energy", fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.tick_params(axis='x', rotation=45)
    
            # Set individual y-axis limits
            if ylim == 0:
                ax.set_ylim(0, combined_results[method].max() * 1.1)
            else:
                ax.set_ylim(0, ylim)
    
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{title}_comparison_individual_scales.png"))
        plt.close(fig)
        print(f"Comparison plot with individual scales saved to {os.path.join(self.save_dir, f'{title}_comparison_individual_scales.png')}")

    def plot_dmd_eigenvalues(self, title="dmd_eigenvalues"):
        """
        Plots the DMD eigenvalues on the complex plane with the unit circle.

        Parameters:
        - title: Title for the plot and filename.
        """
        # Extract eigenvalues
        eigenvalues = self.dmd.eigs

        # Create a unit circle
        theta = np.linspace(0, 2 * np.pi, 500)
        unit_circle = np.exp(1j * theta)

        # Plot setup
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(unit_circle.real, unit_circle.imag, 'r--', label="Unit Circle")  # Unit circle
        ax.scatter(eigenvalues.real, eigenvalues.imag, color='blue', label="DMD Eigenvalues")  # DMD eigenvalues

        # Formatting the plot
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Real axis
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Imaginary axis
        ax.set_xlabel("R(λ)", fontsize=14)
        ax.set_ylabel("Im(λ)", fontsize=14)
        #ax.set_title("DMD Eigenvalues on the Complex Plane", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axis('equal')  # Equal scaling for x and y axes

        # Save the plot
        plot_path = os.path.join(self.save_dir, f"{title}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"DMD eigenvalues plot saved to {plot_path}")
        
    def plot_mode_contributions(self, contributions, save_dir, filename="mode_contributions.png", title="Mode Contributions"):
        """
        Generates a bar plot of the contributions of DMD modes and saves it as an image file.
    
        Parameters:
        -----------
        contributions : list or numpy array
            Contributions of the DMD modes.
        save_dir : str
            Directory where the image file will be saved.
        filename : str, optional
            The name of the saved image file. Default is "mode_contributions.png".
        title : str, optional
            Title of the plot. Default is "Mode Contributions".
    
        Returns:
        --------
        None
        """
        # Ensure contributions are a numpy array
        contributions = np.array(contributions)
        
        # Generate x-axis labels (Mode indices)
        mode_indices = np.arange(0, len(contributions))
        
        
        # Create the bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(mode_indices, contributions, color='skyblue', edgecolor='black')
        
        # Add labels and title
        plt.xlabel("Mode Index")
        plt.ylabel("Contribution")
        plt.title(title)
        plt.xticks(mode_indices)
        
        # Show grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the plot as an image
        save_path = os.path.join(save_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
    
        print(f"Plot saved to: {save_path}")

    def plot_contribution_vs_frequency_separate(self, filename="contribution_vs_frequency_separate.png", title="Contribution vs Frequency Comparison"):
        """
        Generates separate scatter plots of DMD mode contributions vs frequencies for each ranking method,
        with a color map showing the damping ratio of the modes. Saves them side by side in a single image file.
    
        Parameters:
        -----------
        filename : str, optional
            The name of the saved image file. Default is "contribution_vs_frequency_separate.png".
        title : str, optional
            Title of the plot. Default is "Contribution vs Frequency Comparison".
    
        Returns:
        --------
        None
        """
        # Define ranking methods and plot titles
        ranking_methods = ["rank_modes", "rank_modes_1", "rank_modes_old"]
        plot_titles = {
            "rank_modes": "Rank Modes",
            "rank_modes_1": "Rank Modes 1",
            "rank_modes_old": "Rank Modes Old"
        }
    
        # Calculate mode frequencies and damping ratios
        eigenvalues = self.dmd.eigs
        mode_frequencies = np.log(eigenvalues).imag / (2 * np.pi * self.dt)
        damping_ratios = -np.log(eigenvalues).real / (2 * np.pi * self.dt)
    
        # Set up the figure and axes
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)  # Separate scales, shared y-axis off
    
        for i, method in enumerate(ranking_methods):
            if hasattr(self, method):
                contributions = getattr(self, method)()
                contributions = np.array(contributions)
    
                # Filter out negative frequencies
                positive_indices = mode_frequencies > 0
                positive_frequencies = mode_frequencies[positive_indices]
                positive_contributions = contributions[positive_indices]
                positive_damping_ratios = damping_ratios[positive_indices]
    
                # Plot on the corresponding axis with a color map for damping ratios
                ax = axes[i]
                scatter = ax.scatter(
                    positive_frequencies,
                    positive_contributions,
                    c=positive_damping_ratios,
                    cmap='viridis',
                    edgecolor='black',
                    s=50,
                    vmin=0,  # Cap color map at 0
                    vmax=1   # Cap color map at 1
                )
    
                # Add a color bar
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label("Damping Ratio", fontsize=12)
    
                # Set labels and title
                ax.set_title(plot_titles[method], fontsize=14)
                ax.set_xlabel("Frequency (Hz)", fontsize=12)
                ax.set_ylabel("Contribution", fontsize=12)
                ax.grid(linestyle='--', alpha=0.7)
    
        # Add a global title
        fig.suptitle(title, fontsize=16)
    
        # Adjust layout to fit titles and save the plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the global title
        save_path = os.path.join(self.save_dir, filename)
        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(save_path)
        plt.close()  # Free memory
    
        print(f"Side-by-side scatter plots excluding negative frequencies saved to: {save_path}")

    def calc_average_error(self):
        orig_pdata = self.dmd.reconstructed_data
        dmd_pdata = np.transpose(self.train_X)
        rmse = np.sqrt(np.sum(np.square(np.subtract(orig_pdata, dmd_pdata))) / (len(orig_pdata[0,:] * len(orig_pdata[:,0]))))
        return rmse
            

def collect_and_average_energy(data_dir, save_dir, start, end, window_size, step, params):
    """
    Collects summed energy groups for multiple overlapping time periods, averages them,
    and creates an averaged graph.

    Parameters:
    - data_dir: Path to the dataset directory.
    - save_dir: Path to save results and plots.
    - start: Start time for the first window.
    - end: End time for the last window.
    - window_size: Size of each time window.
    - step: Step size for shifting the time window.
    - params: Dictionary containing parameters for HankelDMDAnalysis (e.g., svd_rank, delay_length).
    """
    all_summed_energies = []
    bins = ['0-5 Hz', '5-20 Hz', '20+ Hz']  # Ensure consistent bin labels

    for t1 in range(start, end - window_size + 1, step):
        t2 = t1 + window_size
        print(f"Processing time window: t1={t1}, t2={t2}")

        try:
            # Create a new HankelDMDAnalysis object for each iteration
            analysis = HankelDMDAnalysis(
                data_dir=data_dir,
                save_dir=save_dir,
                svd_rank=params['svd_rank'],
                delay_length=params['delay_length']
            )

            # Prepare the analysis
            analysis.make_save_dir()
            names = ["p"]
            is_building_li = [False]
            relative_paths = [r"p.csv"]
            coords_relative_paths = [r"coords.csv"]
            analysis.add_datasets(names, relative_paths, coords_relative_paths, is_building_li)

            # Trim datasets for this time window
            analysis.trim_datasets(t1=t1, t2=t2, i1=0, i2=None, ds_indices=[0])

            # Process datasets
            analysis.demean_datasets()
            analysis.normalize_datasets()
            analysis.fit(ds_indices=[0])

            # Compute summed energy groups
            mode_frequencies = np.log(analysis.dmd.eigs).imag / (2 * np.pi * analysis.dt)
            mode_energies = analysis.rank_modes()
            df = pd.DataFrame({
                'Frequency (Hz)': mode_frequencies,
                'Energy': np.abs(mode_energies)
            }).sort_values(by='Frequency (Hz)').reset_index(drop=True)
            df = df[df['Frequency (Hz)'] > 0]

            df['Group'] = pd.cut(
                df['Frequency (Hz)'], bins=[0, 5, 20, df['Frequency (Hz)'].max() + 1],
                labels=bins, right=False
            )

            summed_energy = df.groupby('Group')['Energy'].sum()
            all_summed_energies.append(summed_energy)

        except ValueError as e:
            print(f"Error processing window t1={t1}, t2={t2}: {e}")
            continue

    # Create a DataFrame from the collected data and compute the average
    if not all_summed_energies:
        print("No valid data to average. Exiting.")
        return

    summed_energy_df = pd.DataFrame(all_summed_energies).fillna(0)  # Handle missing groups
    averaged_energy = summed_energy_df.mean()

    # Plot the averaged energy
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(averaged_energy.index, averaged_energy.values, color='blue')

    # Set the plot title and axis labels
    ax.set_title("Averaged Energy for Each Frequency Group", fontsize=20)
    ax.set_xlabel("Frequency Group", fontsize=16)
    ax.set_ylabel("Averaged Summed Energy", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Save the plot
    plot_path = os.path.join(save_dir, "averaged_energy_groups.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Averaged energy plot saved to {plot_path}")

    # Save the averaged data to a CSV file
    csv_path = os.path.join(save_dir, "averaged_energy_groups.csv")
    averaged_energy.to_csv(csv_path, header=["Averaged Energy"], index_label="Frequency Group")
    print(f"Averaged energy data saved to {csv_path}")
    
def collect_and_average_energy_with_rankings(data_dir, save_dir, start, end, window_size, step, params):
    """
    Collects summed energy groups for multiple overlapping time periods using three ranking methods,
    averages them, and creates a side-by-side comparison plot.

    Parameters:
    -----------
    data_dir : str
        Path to the dataset directory.
    save_dir : str
        Path to save results and plots.
    start : int
        Start time for the first window.
    end : int
        End time for the last window.
    window_size : int
        Size of each time window.
    step : int
        Step size for shifting the time window.
    params : dict
        Dictionary containing parameters for HankelDMDAnalysis (e.g., svd_rank, delay_length).
    """
    # Initialize data structures for storing results
    all_summed_energies = {"rank_modes": [], "rank_modes_1": [], "rank_modes_old": []}
    bins = ['0-5 Hz', '5-20 Hz', '20+ Hz']  # Ensure consistent bin labels

    for t1 in range(start, end - window_size + 1, step):
        t2 = t1 + window_size
        print(f"Processing time window: t1={t1}, t2={t2}")

        try:
            # Create a new HankelDMDAnalysis object for each iteration
            analysis = HankelDMDAnalysis(
                data_dir=data_dir,
                save_dir=save_dir,
                svd_rank=params['svd_rank'],
                delay_length=params['delay_length']
            )

            # Prepare the analysis
            analysis.make_save_dir()
            names = ["p"]
            is_building_li = [False]
            relative_paths = [r"p.csv"]
            coords_relative_paths = [r"coords.csv"]
            analysis.add_datasets(names, relative_paths, coords_relative_paths, is_building_li)

            # Trim datasets for this time window
            analysis.trim_datasets(t1=t1, t2=t2, i1=0, i2=None, ds_indices=[0])

            # Process datasets
            analysis.demean_datasets()
            analysis.normalize_datasets()
            analysis.fit(ds_indices=[0])

            # Calculate mode frequencies
            mode_frequencies = np.log(analysis.dmd.eigs).imag / (2 * np.pi * analysis.dt)

            # Process each ranking method
            for method in ["rank_modes", "rank_modes_1", "rank_modes_old"]:
                if hasattr(analysis, method):
                    mode_energies = getattr(analysis, method)()
                    df = pd.DataFrame({
                        'Frequency (Hz)': mode_frequencies,
                        'Energy': np.abs(mode_energies)
                    }).sort_values(by='Frequency (Hz)').reset_index(drop=True)
                    df = df[df['Frequency (Hz)'] > 0]

                    df['Group'] = pd.cut(
                        df['Frequency (Hz)'], bins=[0, 5, 20, df['Frequency (Hz)'].max() + 1],
                        labels=bins, right=False
                    )

                    summed_energy = df.groupby('Group')['Energy'].sum()
                    all_summed_energies[method].append(summed_energy)

        except ValueError as e:
            print(f"Error processing window t1={t1}, t2={t2}: {e}")
            continue

    # Compute averages for each ranking method
    averaged_energies = {}
    for method, energies in all_summed_energies.items():
        if energies:
            summed_energy_df = pd.DataFrame(energies).fillna(0)  # Handle missing groups
            averaged_energies[method] = summed_energy_df.mean()

    # Plot the averaged energies side by side
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.25  # Width of each bar
    x = np.arange(len(bins))  # x positions for groups

    for i, (method, averaged_energy) in enumerate(averaged_energies.items()):
        ax.bar(x + i * width, averaged_energy.values, width, label=method)

    # Add labels, title, and legend
    ax.set_title("Averaged Energy for Each Frequency Group by Ranking Method", fontsize=16)
    ax.set_xlabel("Frequency Group", fontsize=14)
    ax.set_ylabel("Averaged Summed Energy", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(bins)
    ax.legend()

    # Save the plot
    plot_path = os.path.join(save_dir, "averaged_energy_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Averaged energy comparison plot saved to {plot_path}")
        
            
        
        
         
        
if __name__ == "__main__":
    data_dir = r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\Data"
    base_save_dir = r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\HankelDMD-update_pressure_400_full_rank"
    cfd_base_save_dir = r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\HankelDMD-update_pressure_400_full_rank"
    svd_rank = -1
    delay_length = 30

    # Initialize lists to store the summed energy values for each window
    summed_energy_all_windows = []

    # Define time windows and shifts
    start_t1 = 1000
    start_t2 = 1400
    shift = 100
    num_windows = 1  # Number of windows to process

    for i in range(num_windows):
        # Calculate the current time window
        t1 = start_t1 + i * shift
        t2 = start_t2 + i * shift
        save_dir = f"{base_save_dir}_shift_{i * shift}"
        cfd_save_dir = f"{cfd_base_save_dir}_shift_{i * shift}"

        print(f"Processing time window {t1}-{t2}, saving to {save_dir}")

        # Initialize analysis object
        analysis = HankelDMDAnalysis(data_dir, save_dir, svd_rank, delay_length)
        analysis.make_save_dir()
        names = ["p"]
        is_building_li = [True]
        relative_paths = [r"p.csv"]
        coords_relative_paths = [r"coords.csv"]
        
        analysis.add_datasets(names, relative_paths, coords_relative_paths, is_building_li)

        # Trim datasets for this time window
        analysis.trim_datasets(t1=t1, t2=t2, i1=0, i2=None, ds_indices=[0])
        # Process datasets
        analysis.demean_datasets()
        analysis.normalize_group(ds_indices = [0]) 
        analysis.fit(ds_indices=[0])
        #analysis.plot_amplitude_frequency()
        #analysis.plot_energy_frequency("energy_frequency")
        #analysis.plot_pressure_energy_frequency("pressure_energy_frequency")
        #analysis.plot_pressure_velocity_ratio("ratio_frequency")
        #analysis.plot_binned_pressure_velocity_ratio(title="binned_ratio_frequency", bin_width=5)
        #analysis.plot_dmd_eigenvalues()
        #analysis.plot_binned_energy_frequency(title="binned_energy_frequency")
        #analysis.plot_binned_stacked_energy(title="binned_stacked_energy")
        #analysis.plot_velocity_energy_frequency(title="velocity_energy_frequency")
        #analysis.plot_timeseries([166, 161, 176])
        analysis.save_dmd()
        analysis.plot_all_ds(plot_negative=True)
        #analysis.plot_combined_amplitude_frequency("combined_amplitude_frequency", f"{save_dir}\mode_data.csv", f"{cfd_save_dir}\mode_data.csv")

        #analysis.plot_modes()
        # Save the summed energy to CSV
        #analysis.plot_summed_energy_groups("energy_bins")
        analysis.plot_rms_timeseries()
        #analysis.plot_cumulative_energy("cumulative_energy")
        
        #plot_cumulative_energy_comparison(f"{save_dir}\cumulative_energy_data.csv", f"{cfd_save_dir}\cumulative_energy_data.csv", ("WT", "CFD"), "Cumulative Energy Comparison", save_dir)
        #analysis.plot_combined_energy_frequency("combined_energy_frequency", f"{save_dir}\energy_frequency_data.csv", f"{cfd_save_dir}\energy_frequency_data.csv")

    #     # Load the saved summed energy CSV for this window
    #     summed_energy_csv_path = os.path.join(save_dir, "summed_energy_groups.csv")
    #     summed_energy_df = pd.read_csv(summed_energy_csv_path)
        
    #     # Collect the summed energy values
    #     summed_energy_all_windows.append(summed_energy_df['Energy'].values)

    # # Convert list to a numpy array for easier manipulation
    # summed_energy_all_windows = np.array(summed_energy_all_windows)

    # # Calculate the mean and standard deviation across all windows for each bin
    # mean_summed_energy = summed_energy_all_windows.mean(axis=0)
    # std_summed_energy = summed_energy_all_windows.std(axis=0)

    # # Create a DataFrame to store the results
    # result_df = pd.DataFrame({
    #     'Frequency Group': summed_energy_df['Group'],
    #     'Average Summed Energy': mean_summed_energy,
    #     'Standard Deviation': std_summed_energy
    # })

    # # Save the results to a CSV file
    # output_csv_path = r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\average_summed_energy_with_std.csv"
    # result_df.to_csv(output_csv_path, index=False)

    # # Plot the average and standard deviation
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.bar(summed_energy_df['Group'], mean_summed_energy, color='orange', yerr=std_summed_energy, capsize=5)
    
    # # Set the plot title and axis labels
    # ax.set_title("Average Summed Energy for Each Frequency Group with Standard Deviation", fontsize=20)
    # ax.set_xlabel("Frequency Group", fontsize=20)
    # ax.set_ylabel("Average Summed Energy", fontsize=20)
    
    # plot_png_path = r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\average_summed_energy_plot.png"
    # plt.savefig(plot_png_path, dpi=300)  # Save with 300 dpi for high resolution
    # plt.close(fig)  # Close the figure after saving to free memory



    # Function to compute mean and standard deviation
    def compute_mean_rms(values):
        values = np.array(values)
        return np.sqrt(np.mean(values**2))

    def plot_cumulative_energy_comparison(csv_path1, csv_path2, labels=("Dataset 1", "Dataset 2"), title="Cumulative Energy Comparison", save_dir="."):
        # Read CSV files
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
    
        # Plot cumulative energy vs frequency for both datasets
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df1['Frequency (Hz)'], df1['Cumulative Energy'], label=labels[0], lw=2, color='blue')
        ax.plot(df2['Frequency (Hz)'], df2['Cumulative Energy'], label=labels[1], lw=2, color='red')
    
        # Set labels and title
        ax.set_xlabel("Frequency (Hz)", fontsize=16)
        ax.set_ylabel("Cumulative Energy", fontsize=16)
        ax.set_title(title, fontsize=18)
    
        # Add legend
        ax.legend(fontsize=14)
    
        # Save the plot
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.show()
        
        return save_path

    # Function to compute mean and standard deviation
    def compute_mean_rms(values):
        values = np.array(values)
        return np.sqrt(np.mean(values**2))


    # Function to compute mean and standard deviation
    def compute_mean_rms(values):
        values = np.array(values)
        return np.sqrt(np.mean(values**2))

    def plot_cumulative_energy_comparison(csv_path1, csv_path2, labels=("Dataset 1", "Dataset 2"), title="Cumulative Energy Comparison", save_dir="."):
        # Read CSV files
        df1 = pd.read_csv(csv_path1)
        df2 = pd.read_csv(csv_path2)
    
        # Plot cumulative energy vs frequency for both datasets
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df1['Frequency (Hz)'], df1['Cumulative Energy'], label=labels[0], lw=2, color='blue')
        ax.plot(df2['Frequency (Hz)'], df2['Cumulative Energy'], label=labels[1], lw=2, color='red')
    
        # Set labels and title
        ax.set_xlabel("Frequency (Hz)", fontsize=16)
        ax.set_ylabel("Cumulative Energy", fontsize=16)
        ax.set_title(title, fontsize=18)
    
        # Add legend
        ax.legend(fontsize=14)
    
        # Save the plot
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.show()
        
        return save_path



    # %% combined summed energy plot

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths to the two CSV files
csv_path_1 = r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\average_summed_energy_with_std.csv"
csv_path_2 = r"C:\Users\Keith\Documents\research_paper\pressure-case\average_summed_energy_with_std.csv"

# Load the data
df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

# Ensure they have the same frequency groups
assert all(df1['Frequency Group'] == df2['Frequency Group']), "Frequency groups do not match!"

# Extract values
groups = df1['Frequency Group']
mean_energy_1 = df1['Average Summed Energy']
std_energy_1 = df1['Standard Deviation']
mean_energy_2 = df2['Average Summed Energy']
std_energy_2 = df2['Standard Deviation']

# Plot settings
bar_width = 0.4  # Width of bars
x = np.arange(len(groups))  # X positions for bars

fig, ax = plt.subplots(figsize=(8, 6))

# Plot bars for dataset 1
ax.bar(x - bar_width/2, mean_energy_1, bar_width, yerr=std_energy_1, capsize=5, label="CFD", color="orange", alpha=0.8)

# Plot bars for dataset 2
ax.bar(x + bar_width/2, mean_energy_2, bar_width, yerr=std_energy_2, capsize=5, label="WT", color="blue", alpha=0.8)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=14)
ax.set_xlabel("Frequency Group", fontsize=16)
ax.set_ylabel("Group-Summed DMD Power", fontsize=16)
#ax.set_title("Comparison of Summed Energy Across Frequency Groups", fontsize=18)
ax.legend(fontsize=14)

ax.grid(True, linestyle="--", alpha=0.7)

# Save the plot
plot_png_path = r"C:\Users\Keith\Documents\research_paper\comparison_plot.png"
plt.savefig(plot_png_path, dpi=300)  # Save as high-resolution PNG
plt.close(fig)  # Close the figure

    # %%
    fshed = 4.72
    Tshed = 1/fshed
    fs = 1/0.005
    print("# of snapshots per shed", Tshed*fs)