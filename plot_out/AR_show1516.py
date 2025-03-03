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
zh0427_data = pd.read_csv("D:/Sat_MV/A15_ZH0427_SlidingWindows_Results.csv")
q0427_data = pd.read_csv("D:/Sat_MV/A15_Q0427_SlidingWindows_Results.csv")
zh0527_data = pd.read_csv("D:/Sat_MV/A15_ZH0527_SlidingWindows_Results.csv")
q0527_data = pd.read_csv("D:/Sat_MV/A15_Q0527_SlidingWindows_Results.csv")

# Extract relevant columns
period0427s_zh = zh0427_data["Period (Hour)"].values
rho0427_zh = zh0427_data["ZH_Apparent_Resistivity"].values
rho_err0427_zh = zh0427_data["ZH_Rho_Error"].values

periods0427_q = q0427_data["Period (Hour)"].values
rho0427_q = q0427_data["Q_Apparent_Resistivity"].values
rho_err0427_q = q0427_data["Q_Rho_Error"].values

period0527s_zh = zh0527_data["Period (Hour)"].values
rho0527_zh = zh0527_data["ZH_Apparent_Resistivity"].values
rho_err0527_zh = zh0527_data["ZH_Rho_Error"].values

periods0527_q = q0527_data["Period (Hour)"].values
rho0527_q = q0527_data["Q_Apparent_Resistivity"].values
rho_err0527_q = q0527_data["Q_Rho_Error"].values

# Downsample the data
period0427s_zh_down, rho0427_zh_down, rho_err0427_zh_down = downsample_logspace(period0427s_zh, rho0427_zh, rho_err0427_zh)
periods0427_q_down, rho0427_q_down, rho_err0427_q_down = downsample_logspace(periods0427_q, rho0427_q, rho_err0427_q)
period0527s_zh_down, rho0527_zh_down, rho_err0527_zh_down = downsample_logspace(period0527s_zh, rho0527_zh, rho_err0527_zh)
periods0527_q_down, rho0527_q_down, rho_err0527_q_down = downsample_logspace(periods0527_q, rho0527_q, rho_err0527_q)

# Plot all datasets in the same figure
plt.figure(figsize=(10, 6))
plt.errorbar(period0427s_zh_down, rho0427_zh_down, yerr=rho_err0427_zh_down, fmt='o', capsize=3, color='red', label='ZH 0427', markersize=8)
plt.errorbar(periods0427_q_down, rho0427_q_down, yerr=rho_err0427_q_down, fmt='^', capsize=3, color='blue', label='Q 0427', markersize=8)
plt.errorbar(period0527s_zh_down, rho0527_zh_down, yerr=rho_err0527_zh_down, fmt='s', capsize=3, markerfacecolor='none', color='red', label='ZH 0527', markersize=8)
plt.errorbar(periods0527_q_down, rho0527_q_down, yerr=rho_err0527_q_down, fmt='d', capsize=3, markerfacecolor='none', color='blue', label='Q 0527', markersize=8)
plt.xscale('log')
plt.xlabel('Period (Hour)', fontsize=14)
plt.ylabel('Apparent Resistivity (Ω·m)', fontsize=14)
plt.title('Apollo15 Apparent Resistivity with Error Bars', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0.1, 2.5)
plt.ylim(0, 20000)
plt.legend(fontsize=14)
#plt.grid()
plt.show()

