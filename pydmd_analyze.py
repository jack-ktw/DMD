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

matplotlib.use('Agg')

class Dataset:
    def __init__(self, name="dataset", is_building=False) -> None:
        self.path = None
        self.coords_array = None
        self.coords_ln = None
        self.data_array = np.empty((0,1))
        self.time_array = np.empty((0,1))
        self.name = name
        self.scalers = None
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
        # Separate real and imaginary parts
        real_part = self.data_array.real
        imag_part = self.data_array.imag
        
        # Normalize real and imaginary parts separately
        real_scaler = preprocessing.RobustScaler()
        imag_scaler = preprocessing.RobustScaler()
        
        real_scaler.fit(real_part)
        imag_scaler.fit(imag_part)
        
        real_normalized = real_scaler.transform(real_part)
        imag_normalized = imag_scaler.transform(imag_part)
        
        # Combine back into complex-valued data
        normalized_data = real_normalized + 1j * imag_normalized
        
        self.data_array = normalized_data
        self.scalers = (real_scaler, imag_scaler)  # Store the scalers for inverse transformation
    

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
            fig_name = f"timeseries_{idx}"
            plt.figure(figsize=(12, 8))
            
            cumulative_error = np.sum((pdata[idx, :] - self.train_X[:, idx])**2)
    
            plt.plot(pdata[idx, :], alpha=0.7, label=f"DMD")
            plt.plot(self.train_X[:, idx], alpha=0.6, label="original")
    
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
            # print("removed", file)

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
                    line_value = 0.5 * 2/3
                    ax.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
                    ax.axvline(x=0.1, color='red', linestyle='-', linewidth=2)
                    ax.axvline(x=0.2, color='red', linestyle='-', linewidth=2)
                    ax.axvline(x=0.3, color='red', linestyle='-', linewidth=2)

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
                    line_value = 0.5 * 2/3
                    ax.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
                    ax.axvline(x=0.1, color='red', linestyle='-', linewidth=2)
                    ax.axvline(x=0.2, color='red', linestyle='-', linewidth=2)
                    ax.axvline(x=0.3, color='red', linestyle='-', linewidth=2)

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
        self.scalers = []  # Initialize the scalers attribute
                    
    def fit(self, ds_indices=None):
        print(self.svd_rank)
        self.dmd = HankelDMD(svd_rank=self.svd_rank,d=self.delay_length)
        self.compose_data(ds_indices=ds_indices)
        self.dmd.fit(X=self.train_X.T)
        print("# modes:", self.dmd.modes.shape)
            
    def plot_timeseries(self, idx_li):
        pdata = self.dmd.reconstructed_data
        for idx in idx_li:
            fig_name = f"timeseries_{idx}"
            plt.figure(figsize=(12, 8))
            plt.plot(pdata[idx, :], alpha=0.7, label=f"DMD")
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
        modes = self.dmd.modes
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
        self.dmd = MrDMD.load(os.path.join(self.save_dir, "dmd.pkl"))
            
    def clean_up_figures(self, pattern):
        matching_files = glob.glob(pattern)
        print("cleaning up", pattern)
        for file in matching_files:
            os.remove(file)
            # print("removed", file)

    def plot_modes(self, ds_idx, plot_negative=False):
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
        modes = self.dmd.modes
        eigs = self.dmd.eigs
            
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
                # Z[phase > np.pi] = -Z[phase > np.pi]
                vmin = -vmax
                cmap="RdBu"
            # Z = np.linalg.norm(pmodes_select)
            
            freq = np.log(eigs[mode_idx]).imag / (2 * np.pi * self.dt)
            grow = eigs[mode_idx].real

            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(111)
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            ax.set_title(f"mode:{mode_idx}, {name}, {freq:.1f} Hz, g:{grow:.2f}")
            ax.set_aspect("equal")
            
            if is_building:
                line_value = 0.5 * 2/3
                ax.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
                ax.axvline(x=0.1, color='red', linestyle='-', linewidth=2)
                ax.axvline(x=0.2, color='red', linestyle='-', linewidth=2)
                ax.axvline(x=0.3, color='red', linestyle='-', linewidth=2)

            plt.savefig(os.path.join(save_dir, f"2_modeshape_{mode_idx}_{name}_{freq:.1f}Hz.png"))
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
        modes = self.dmd.modes
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

            fig = plt.figure(figsize=(8, 6))
            ax = plt.subplot(111)
            levels = np.linspace(vmin, vmax, 20)
            CS = plt.contourf(X, Y, Z, cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
            colorbar = plt.colorbar(CS)
            ax.set_title(f"phase: , mode:{mode_idx}, {name}, {freq:.1f} Hz, g:{grow:.2f}")
            ax.set_aspect("equal")
            
            if is_building:
                line_value = 0.5 * 2/3
                ax.axhline(y=line_value, color='red', linestyle='--', linewidth=2)
                ax.axvline(x=0.1, color='red', linestyle='-', linewidth=2)
                ax.axvline(x=0.2, color='red', linestyle='-', linewidth=2)
                ax.axvline(x=0.3, color='red', linestyle='-', linewidth=2)

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
        
        mode_frequencies = np.log(self.dmd.eigs).imag / (2 * np.pi * self.dt)
        mode_amplitudes = self.dmd.amplitudes
        
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
                        str(i), ha='right', va='bottom')
        
        # Set the plot title and axis labels
        ax.set_title("DMD Mode Amplitudes vs Frequencies")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0)
        
        # Add a colorbar to the plot
        norm = mcolors.Normalize(vmin=0, vmax=len(mode_frequencies))
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
        cbar.set_label("Mode Number")
        
        plt.savefig(os.path.join(self.save_dir, "amplitude_frequency.png"))
        plt.close(fig)
        plt.clf()
        plt.close("all")
        gc.collect()
    
    def normalize_datasets(self, ds_indices=None):
        if ds_indices is None:
            ds_indices = range(len(self.datasets))
        
        self.scalers = []  # Reset the scalers list
        for i in ds_indices:
            self.datasets[i].normalize_data()
            self.scalers.append(self.datasets[i].scalers)  # Store the scaler for each dataset

    def denormalize_modes(self):
        if not hasattr(self, 'scalers'):
            raise ValueError("Scalers have not been initialized. Please normalize the datasets first.")
        print(self.scalers)
        real_scaler, imag_scaler = self.scalers
        
        # Separate real and imaginary parts
        real_normalized = self.dmd.modes.real
        imag_normalized = self.dmd.modes.imag
        
        # Apply inverse transformation to real and imaginary parts separately
        real_denormalized = real_scaler.inverse_transform(real_normalized)
        imag_denormalized = imag_scaler.inverse_transform(imag_normalized)
        
        # Combine back into complex-valued data
        denormalized_modes = real_denormalized + 1j * imag_denormalized
        
        self.dmd.modes = denormalized_modes
            
if __name__ == "__main__":
    data_dir = r"C:\Users\Keith\Documents\Research\data"
    save_dir = r"C:\Users\Keith\Documents\Research\HankelDMD_short100_update"

    max_level = 6
    max_cycles = 4
    svd_rank = 200
    tikhonov_regularization = 1e-7
    delay_length = 25
    analysis = HankelDMDAnalysis(data_dir, save_dir, svd_rank, delay_length)
    analysis.make_save_dir()

    names = ["U", "V", "W", "p", "building_p"]
    is_building_li = [False, False, False, False, True]
    relative_paths = [r"left_region/ux.csv", r"left_region/uy.csv", r"left_region\uz.csv", r"left_region\p.csv", r"building_p.csv"]
    coords_relative_paths = [r"left_region/coords.csv", -1, -1, -1, r"building_coords_flat.csv"]
    analysis.add_datasets(names, relative_paths, coords_relative_paths, is_building_li)
    analysis.trim_datasets(t1=0, t2=100, i1=3000, i2=6000, ds_indices=[0, 1, 2, 3])
    analysis.trim_datasets(t1=0, t2=100, ds_indices=[4])
    analysis.filter_datasets(x_upper=0.05, ds_indices=[0, 1, 2, 3])
    analysis.demean_datasets()
    analysis.normalize_datasets(ds_indices=[0, 1, 4])
    #analysis.compose_data(ds_indices=[0, 1, 4])
    analysis.fit(ds_indices=[0, 1, 4])
    analysis.save_dmd()
    # analysis.load_dmd()
    analysis.denormalize_modes()

    analysis.plot_timeseries([0, 50, 100, 200, 1000, 2000])
    analysis.plot_dynamics()
    analysis.plot_all_ds(plot_negative=True)
    analysis.plot_amplitude_frequency()

    # %%

    # idx_li = dmd0.time_window_bins(0, 400)
    # freqs = []
    # amps = []
    # for idx in idx_li:
    #     if len(dmd0.dmd_tree[idx].frequency) > 0:
    #         freqs.append(dmd0.dmd_tree[idx].frequency[0])
    #         amps.append(abs(dmd0.dmd_tree[idx].amplitudes[0]))
    #         plt.text(freqs[-1], amps[-1], s=str(idx))
    # plt.scatter(freqs, amps)

    # %%
    fshed = 4.72
    Tshed = 1/fshed
    fs = 1/0.005
    print("# of snapshots per shed", Tshed*fs)