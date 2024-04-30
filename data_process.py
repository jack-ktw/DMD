# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:04:47 2024

@author: Keith



"""
from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np


def load_exp_data(exp_df):
    data_d = loadmat(exp_df)
    fs = 500
    p_array = data_d["Wind_pressure_coefficients"]
    exp_time_array = np.arange(p_array.shape[0]) / fs
    location_array = data_d["Location_of_measured_points"]
    location_df = pd.DataFrame(location_array.T, columns=["x", "y", "tap_number", "face_number"])
    return p_array, location_df, exp_time_array


p_array, location_df, exp_time_array = load_exp_data(r"C:\Users\Keith\Documents\research_paper\CFD-pressure-case\Data\1_5_CFD.mat")

p_array = pd.DataFrame(p_array)
p_array.insert(0, 'Time', exp_time_array)
p_array.columns = ['Time'] + [str(i) for i in range(len(p_array.columns) - 1)]

p_array.to_csv(r'C:\Users\Keith\Documents\research_paper\CFD-pressure-case\Data\p.csv', index=False)

location_df.insert(0, 'Index', range(len(location_df)))
location_df.to_csv(r'C:\Users\Keith\Documents\research_paper\CFD-pressure-case\Data\coords.csv', index=False)
