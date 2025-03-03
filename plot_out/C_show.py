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
zh_data = pd.read_csv("D:/Sat_MV/A12_ZH_C_Response.csv")
q_data = pd.read_csv("D:/Sat_MV/A12_Q_C_Response.csv")

# Extract relevant columns
periods_zh = zh_data["Period (Hour)"].values
Creal_zh = zh_data["C_real"].values
Creal_err_zh = zh_data["C_real_Error"].values
Cimag_zh = zh_data["C_imag"].values
Cimag_err_zh = zh_data["C_imag_Error"].values

periods_q = q_data["Period (Hour)"].values
Creal_q = q_data["C_real"].values
Creal_err_q = q_data["C_real_Error"].values
Cimag_q = q_data["C_imag"].values
Cimag_err_q = q_data["C_imag_Error"].values

# Downsample the data
periods_zh_down, Creal_zh_down, Creal_err_zh_down = downsample_logspace(periods_zh, Creal_zh, Creal_err_zh)
periods_zh_down, Cimag_zh_down, Cimag_err_zh_down = downsample_logspace(periods_zh, Cimag_zh, Cimag_err_zh)
periods_q_down, Creal_q_down, Creal_err_q_down = downsample_logspace(periods_q, Creal_q, Creal_err_q)
periods_q_down, Cimag_q_down, Cimag_err_q_down = downsample_logspace(periods_q, Cimag_q, Cimag_err_q)

# Plot the apparent resistivity with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(periods_zh_down, Creal_zh_down, yerr=Creal_err_zh_down, fmt='o', capsize=3, color='red', label='Real part of ZH-method', markersize=8)
plt.errorbar(periods_zh_down, Cimag_zh_down, yerr=Cimag_err_zh_down, fmt='^', capsize=3, color='red', label='Imag part of ZH-method', markersize=8)
plt.errorbar(periods_q_down, Creal_q_down, yerr=Creal_err_q_down, fmt='o', capsize=3, color='blue', label='Real part of Q-method', markersize=8)
plt.errorbar(periods_q_down, Cimag_q_down, yerr=Cimag_err_q_down, fmt='^', capsize=3, color='blue', label='Imag part of Q-method', markersize=8)
plt.xscale('log')
plt.xlabel('Period (Hour)', fontsize=14)
plt.ylabel('C-response (km)', fontsize=14)
plt.title('Apollo 12 C-response with Error Bars', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.1, 12)
plt.ylim(-500, 2000)
plt.legend(fontsize=14)
plt.show()

