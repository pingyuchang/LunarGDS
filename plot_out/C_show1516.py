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
zh0427_data = pd.read_csv("D:/Sat_MV/A16_ZH0427_C_Response.csv")
q0427_data = pd.read_csv("D:/Sat_MV/A16_Q0427_C_Response.csv")
zh0527_data = pd.read_csv("D:/Sat_MV/A16_ZH0527_C_Response.csv")
q0527_data = pd.read_csv("D:/Sat_MV/A16_Q0527_C_Response.csv")

# Extract relevant columns
periods0427_zh = zh0427_data["Period (Hour)"].values
Creal0427_zh = zh0427_data["C_real"].values
Creal0427_err_zh = zh0427_data["C_real_Error"].values
Cimag0427_zh = zh0427_data["C_imag"].values
Cimag0427_err_zh = zh0427_data["C_imag_Error"].values

periods0427_q = q0427_data["Period (Hour)"].values
Creal0427_q = q0427_data["C_real"].values
Creal0427_err_q = q0427_data["C_real_Error"].values
Cimag0427_q = q0427_data["C_imag"].values
Cimag0427_err_q = q0427_data["C_imag_Error"].values

periods0527_zh = zh0527_data["Period (Hour)"].values
Creal0527_zh = zh0527_data["C_real"].values
Creal0527_err_zh = zh0527_data["C_real_Error"].values
Cimag0527_zh = zh0527_data["C_imag"].values
Cimag0527_err_zh = zh0527_data["C_imag_Error"].values

periods0527_q = q0527_data["Period (Hour)"].values
Creal0527_q = q0527_data["C_real"].values
Creal0527_err_q = q0527_data["C_real_Error"].values
Cimag0527_q = q0527_data["C_imag"].values
Cimag0527_err_q = q0527_data["C_imag_Error"].values

# Downsample the data
periods0427_zh_down, Creal0427_zh_down, Creal0427_err_zh_down = downsample_logspace(periods0427_zh, Creal0427_zh, Creal0427_err_zh)
periods0427_zh_down, Cimag0427_zh_down, Cimag0427_err_zh_down = downsample_logspace(periods0427_zh, Cimag0427_zh, Cimag0427_err_zh)
periods0427_q_down, Creal0427_q_down, Creal0427_err_q_down = downsample_logspace(periods0427_q, Creal0427_q, Creal0427_err_q)
periods0427_q_down, Cimag0427_q_down, Cimag0427_err_q_down = downsample_logspace(periods0427_q, Cimag0427_q, Cimag0427_err_q)

periods0527_zh_down, Creal0527_zh_down, Creal0527_err_zh_down = downsample_logspace(periods0527_zh, Creal0527_zh, Creal0527_err_zh)
periods0527_zh_down, Cimag0527_zh_down, Cimag0527_err_zh_down = downsample_logspace(periods0527_zh, Cimag0527_zh, Cimag0527_err_zh)
periods0527_q_down, Creal0527_q_down, Creal0527_err_q_down = downsample_logspace(periods0527_q, Creal0527_q, Creal0527_err_q)
periods0527_q_down, Cimag0527_q_down, Cimag0527_err_q_down = downsample_logspace(periods0527_q, Cimag0527_q, Cimag0527_err_q)

# Plot the apparent resistivity with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(periods0427_zh_down, Creal0427_zh_down, yerr=Creal0427_err_zh_down, fmt='o', capsize=3, color='red', label='0427 Real part of ZH-method', markersize=8)
plt.errorbar(periods0427_zh_down, Cimag0427_zh_down, yerr=Cimag0427_err_zh_down, fmt='^', capsize=3, color='red', label='0427 Imag part of ZH-method', markersize=8)
plt.errorbar(periods0427_q_down, Creal0427_q_down, yerr=Creal0427_err_q_down, fmt='o', capsize=3, color='blue', label='0427 Real part of Q-method', markersize=8)
plt.errorbar(periods0427_q_down, Cimag0427_q_down, yerr=Cimag0427_err_q_down, fmt='^', capsize=3, color='blue', label='0427 Imag part of Q-method', markersize=8)
plt.errorbar(periods0527_zh_down, Creal0527_zh_down, yerr=Creal0527_err_zh_down, fmt='s', capsize=3, markerfacecolor='none', color='red', label='0527 Real part of ZH-method', markersize=8)
plt.errorbar(periods0527_zh_down, Cimag0527_zh_down, yerr=Cimag0527_err_zh_down, fmt='d', capsize=3, markerfacecolor='none', color='red', label='0527 Imag part of ZH-method', markersize=8)
plt.errorbar(periods0527_q_down, Creal0527_q_down, yerr=Creal0527_err_q_down, fmt='s', capsize=3, markerfacecolor='none', color='blue', label='0527 Real part of Q-method', markersize=8)
plt.errorbar(periods0527_q_down, Cimag0527_q_down, yerr=Cimag0527_err_q_down, fmt='d', capsize=3, markerfacecolor='none', color='blue', label='0527 Imag part of Q-method', markersize=8)
plt.xscale('log')
plt.xlabel('Period (Hour)', fontsize=14)
plt.ylabel('C-response (km)', fontsize=14)
plt.title('Apollo 16 C-response with Error Bars', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0.1, 2.5)
plt.ylim(-200, 1000)
plt.legend(fontsize=14)
plt.show()

