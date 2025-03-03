# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:16:37 2025

@author: pingy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# Function to downsample data to ~10 points per decade

def downsample_logspace(x, y, y_err, num_per_decade=10):
    log_x = np.log10(x)
    min_exp = np.floor(log_x.min())
    max_exp = np.ceil(log_x.max())
    bins = np.logspace(min_exp, max_exp, int((max_exp - min_exp) * num_per_decade))
    indices = np.digitize(x, bins)
    unique_indices = np.unique(indices, return_index=True)[1]
    return x[unique_indices], y[unique_indices], y_err[unique_indices]

# Read in the data files
zh_data = pd.read_csv("D:/Sat_MV/A12_ZH_SlidingWindows_Results.csv")
q_data = pd.read_csv("D:/Sat_MV/A12_Q_SlidingWindows_Results.csv")

# Extract relevant columns
periods_zh = zh_data["Period (Hour)"].values
rho_zh = zh_data["ZH_Apparent_Resistivity"].values
rho_err_zh = zh_data["ZH_Rho_Error"].values

periods_q = q_data["Period (Hour)"].values
rho_q = q_data["Q_Apparent_Resistivity"].values
rho_err_q = q_data["Q_Rho_Error"].values

# Downsample the data
periods_zh_down, rho_zh_down, rho_err_zh_down = downsample_logspace(periods_zh, rho_zh, rho_err_zh)
periods_q_down, rho_q_down, rho_err_q_down = downsample_logspace(periods_q, rho_q, rho_err_q)

# Plot the apparent resistivity with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(periods_zh_down, rho_zh_down, yerr=rho_err_zh_down, fmt='o', capsize=3, color='red', label='ZH-method', markersize=8)
plt.errorbar(periods_q_down, rho_q_down, yerr=rho_err_q_down, fmt='^', capsize=3, color='blue', label='Q-method', markersize=8)
plt.xscale('log')
plt.xlabel('Period (Hour)', fontsize=14)
plt.ylabel('Apparent Resistivity (Ω·m)', fontsize=14)
plt.title('Apollo12 Apparent Resistivity with Error Bars', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0.1, 12)
plt.ylim(0, 20000)
plt.legend(fontsize=14)
#plt.grid()
plt.show()

