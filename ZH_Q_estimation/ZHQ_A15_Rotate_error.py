# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:56:38 2025

@author: pingy
All calculations are based on the ME coordinate system
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats

# =========================
# Original Code for Apollo15 (Unchanged)
# =========================

# Read Apollo 15 data
# (Assumes the CSV file has an index column that can be parsed as dates)
#kny = pd.read_csv("D:/Sat_MV/APollo15_0427.csv", parse_dates=True, index_col=0, sep=',').iloc[74500:129600]
kny = pd.read_csv("D:/Sat_MV/APollo15_0527.csv", parse_dates=True, index_col=0, sep=',').iloc[242875:286202]
Decimate_factor=1
kny = kny.iloc[::Decimate_factor]  # Downsample: Keep every nth row

# Sampling frequency and time calculations
fs = 3 / Decimate_factor           # Sampling frequency (3 Hz)
dt = 1 / fs     # Sampling interval in seconds
endtime = len(kny) * dt   # Total time in seconds
tt = np.arange(0, endtime, dt)
miu0 = 4 * np.pi / 10**7
rLunar = 1737   # Moon radius in km
Coh15_threshold = 0.7  # Coherence threshold for C-response
Q15_threshold = 0.7    # Coherence threshold for Q-response
seg = 1
segnum = int(len(kny) / seg)
overlapnum = int(segnum / 8)

# Adjust nperseg to avoid warnings
nperseg = min(segnum, len(kny))

# Read three-axis magnetic field vectors X, Y, Z (adding offsets)
knyX = signal.medfilt(kny['BX'] + 0.2 , 1)
knyY = signal.medfilt(kny['BY'] - 0.9 , 1)
knyZ = signal.medfilt(kny['BZ'] - 3.3 , 1)

# Rotate the magnetic field to maximize Bz
def rotate_to_max_bz(Bx, By, Bz):
    By_mean = np.mean(By)
    Bz_mean = np.mean(Bz)
    thetaRR = np.arctan2(By_mean, Bz_mean)  # Use averaged By and Bz values
    
    cos_theta = np.cos(thetaRR)
    sin_theta = np.sin(thetaRR)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    
    # Stack By and Bz for transformation
    B_vec = np.vstack((By, Bz))
    rotated = R @ B_vec  # Matrix multiplication for transformation
    
    return Bx, rotated[0], rotated[1]

knyX, knyY, knyZ = rotate_to_max_bz(knyX, knyY, knyZ)

# Apply perturbation
perturb_Mag = 100
knyX += perturb_Mag
knyY += perturb_Mag
knyZ += perturb_Mag
                                         
# Plot the variation of X, Y, Z magnetic fields over time (in hours)
plt.figure(figsize=(15, 6))
plt.plot(tt / 3600, knyX, label='Magnetic Field X', alpha=0.8)
plt.plot(tt / 3600, knyY, label='Magnetic Field Y', alpha=0.8)
plt.plot(tt / 3600, knyZ, label='Magnetic Field Z', alpha=0.8)
plt.title("Magnetic Field Variation Over Time (Apollo15)")
plt.xlabel("Time (Hours)")
plt.ylabel("Magnetic Field (nT)")
plt.ylim(-20 + perturb_Mag, 20 + perturb_Mag)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ZH Method: colatitude input and corresponding perturbed longitude calculation
KnymeanX = np.average(knyX)
KnymeanY = np.average(knyY)
KnymeanZ = np.average(knyZ)

# For Apollo15, abgle15 is computed as follows:
#abgle15 = 117.43 + 14  # for 0427
abgle15 = 129.51 + 3  # for 0527
Theta15 = abgle15 * np.pi / 180
#Rphi15 = (3.76) * np.pi / 180 # for 0427
Rphi15 = (6.41) * np.pi / 180 # for 0527
#Rphi15 = np.arctan(-1 * np.cos(Theta15) * KnymeanY / KnymeanZ)

# =========================
# New Section: Sliding Windows Analysis for Apollo15
# Each window's length is 90% of the original dataset.
# We choose 5 equally spaced windows, compute the apparent resistivity and phase for each window,
# then estimate the error (standard deviation) across the windows,
# plot the results with error bars (with the horizontal axis in log scale),
# and save the mean and error into CSV files.
# =========================

N = len(kny)
win_len = int(0.8 * N)

# Compute 5 equally spaced starting indices between 0 and N - win_len
start_indices = np.linspace(0, N - win_len, num=5, dtype=int)
print("Sliding window start indices (each window 90% of dataset):", start_indices)

# Lists to store results for each window (for both methods)
# For ZH-based method:
ZH_rho_windows = []
ZH_phase_windows = []
ZH_C_windows = []
# For Q-based method:
Q_rho_windows = []
Q_phase_windows = []
Q_C_windows = []

nperseg_win = min(int(0.9*win_len), win_len)  # You may adjust this parameter
noverlap_win = int(nperseg_win * 0.5)
freqs_win = None

for start in start_indices:
    window_data = kny.iloc[start:start+win_len].reset_index(drop=True)
    # Process magnetic field vectors for this window
    X_win = signal.medfilt(window_data['BX'] + 25.8 + perturb_Mag, 1)
    Y_win = signal.medfilt(window_data['BY'] - 11.9 + perturb_Mag, 1)
    Z_win = signal.medfilt(window_data['BZ'] + 25.8 + perturb_Mag, 1)
    
    # ---------- ZH-based processing ----------
    f_win, t_win, Hx_win = signal.stft(X_win, fs, window='hann', nperseg=nperseg_win,
                                       noverlap=noverlap_win, return_onesided=False)
    f_win, t_win, Hz_win = signal.stft(Z_win, fs, window='hann', nperseg=nperseg_win,
                                       noverlap=noverlap_win, return_onesided=False)
    if freqs_win is None:
        freqs_win = f_win.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_win = Hx_win / np.where(np.abs(Hz_win) > 1e-10, Hz_win, np.nan)
    median_ratio_win = np.nanmedian(ratio_win, axis=1)
    Cr_win = (-1/2) * np.tan(Theta15) * rLunar * median_ratio_win
    ZH_rho = 1e6 * 2 * np.pi * f_win * miu0 * (np.abs(Cr_win))**2
    ZH_phase = 0 + np.angle(Cr_win, deg=True)
    ZH_C_windows.append(Cr_win)
    ZH_rho_windows.append(ZH_rho)
    ZH_phase_windows.append(ZH_phase)
    
    # ---------- Q-based processing ----------
    array121 = np.array([[-np.cos(Rphi15) * np.sin(Theta15), 2 * np.cos(Rphi15) * np.sin(Theta15)],
                          [-np.cos(Rphi15) * np.cos(Theta15), -np.cos(Rphi15) * np.cos(Theta15)],
                          [np.sin(Rphi15), np.sin(Rphi15)]])
    array122 = np.array([X_win, Y_win, Z_win])
    q_win, g_win = np.linalg.lstsq(array121, array122, rcond=None)[0]
    fq_win, tq_win, Qq_win = signal.stft(q_win, fs, window='hann', nperseg=nperseg_win,
                                         noverlap=noverlap_win, return_onesided=False)
    fq_win, tq_win, Qg_win = signal.stft(g_win, fs, window='hann', nperseg=nperseg_win,
                                         noverlap=noverlap_win, return_onesided=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        Q_ratio_win = np.where(np.abs(np.nanmean(Qq_win, axis=1)) > 1e-10,
                                np.nanmean(Qg_win, axis=1) / np.nanmean(Qq_win, axis=1),
                                np.nan)
    Qresponse_win = (-rLunar / 2) * (1 - 2 * Q_ratio_win) / (1 + Q_ratio_win)
    Q_rho = 1e6 * 2 * np.pi * fq_win * miu0 * (np.abs(Qresponse_win))**2
    Q_phase = 0 + np.angle(Qresponse_win, deg=True)
    Q_C_windows.append(Qresponse_win)
    Q_rho_windows.append(Q_rho)
    Q_phase_windows.append(Q_phase)

# Filter the frequency axis to use only positive frequencies
mask = (freqs_win) > 0
freqs_win = freqs_win[mask]

# Convert frequency axis to period (in hours) using the absolute value
periods_win = 1 / np.abs(freqs_win) / 3600

# Also apply the mask to the sliding windows results:
ZH_rho_array = np.array(ZH_rho_windows)[:, mask]
ZH_phase_array = np.array(ZH_phase_windows)[:, mask]
ZH_C_array = np.array(ZH_C_windows)[:, mask]
Q_rho_array = np.array(Q_rho_windows)[:, mask]
Q_phase_array = np.array(Q_phase_windows)[:, mask]
Q_C_array = np.array(Q_C_windows)[:, mask]

# For each frequency bin, compute the mean and standard deviation (error) across the 5 windows
ZH_rho_mean = np.nanmean(ZH_rho_array, axis=0)
ZH_rho_std  = np.nanstd(ZH_rho_array, axis=0)
ZH_phase_mean = np.nanmean(ZH_phase_array, axis=0)
ZH_phase_std  = np.nanstd(ZH_phase_array, axis=0)
ZH_C_mean = np.nanmean(ZH_C_array, axis=0)
ZH_C_stdR  = np.nanstd(np.real(ZH_C_array), axis=0)
ZH_C_stdI  = np.nanstd(np.imag(ZH_C_array), axis=0)

Q_rho_mean = np.nanmean(Q_rho_array, axis=0)
Q_rho_std  = np.nanstd(Q_rho_array, axis=0)
Q_phase_mean = np.nanmean(Q_phase_array, axis=0)
Q_phase_std  = np.nanstd(Q_phase_array, axis=0)
Q_C_mean = np.nanmean(Q_C_array, axis=0)
Q_C_stdR  = np.nanstd(np.real(Q_C_array), axis=0)
Q_C_stdI  = np.nanstd(np.imag(Q_C_array), axis=0)

# -------------------------
# Plot separate error-bar plots for ZH-based method (with horizontal axis in log scale)
# -------------------------
def plot_c_response_with_error(periods, C_meanZH, C_meanQ, C_stdZHR, C_stdZHI, C_stdQR, C_stdQI ):
    plt.figure(figsize=(8, 6))
    plt.errorbar(periods, np.real(C_meanZH), yerr=np.abs(C_stdZHR), fmt='o', capsize=3, color='blue', label='Real Part_ZH')
    plt.errorbar(periods, np.real(C_meanQ), yerr=np.abs(C_stdQR), fmt='^', capsize=3, color='blue', label='Real Part_Q')
    plt.errorbar(periods, np.imag(C_meanZH), yerr=np.abs(C_stdZHI), fmt='ro', capsize=3, label='Imaginary Part_ZH')
    plt.errorbar(periods, np.imag(C_meanQ), yerr=np.abs(C_stdQI), fmt='r^', capsize=3, label='Imaginary Part_Q')
    plt.xscale('log')
    plt.xlabel('Period (Hour)', fontsize=14)
    plt.ylabel('C-response (km)', fontsize=14)
    #plt.title('C-response with Error Bars', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0.1, 2)
    plt.ylim(-500, 1500)
    plt.legend()
    plt.show()

# Plot C-response with error bars for both methods
plot_c_response_with_error(periods_win, ZH_C_mean, Q_C_mean, ZH_C_stdR, ZH_C_stdI, Q_C_stdR, Q_C_stdI)

# -------------------------
# Plot separate error-bar plots for Q-based method (with horizontal axis in log scale)
# -------------------------
plt.figure(figsize=(8, 6))
plt.errorbar(periods_win, ZH_rho_mean, yerr=np.abs(ZH_rho_std), fmt='o', capsize=3, color='red', label='ZH-method')
plt.errorbar(periods_win, Q_rho_mean, yerr=np.abs(Q_rho_std), fmt='^', capsize=3, color='blue', label='Q-method')
plt.xscale('log')
plt.xlabel('Period (Hour)', fontsize=14)
plt.ylabel('Apparent Resistivity (Ω·m)', fontsize=14)
#plt.title('Apparent Resistivity with Error Bars', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0.1, 2)
plt.ylim(0, 20000)
plt.legend()
#plt.grid(True, which="both")
plt.show()


#plt.figure(figsize=(8, 6))
#plt.errorbar(periods_win, Q_phase_mean, yerr=np.abs(Q_phase_std), fmt='o', capsize=3, color='red')
#plt.xscale('log')
#plt.xlabel('Period (Hour)')
#plt.ylabel('Q-based Phase (degrees)')
#plt.title('Q-based Phase with Error Bars')
#plt.xlim(0.3, 12)
#plt.grid(True, which="both")
#plt.show()

# -------------------------
# Save the sliding windows results (mean and error) to CSV files.
# For ZH-based method:
ZH_df = pd.DataFrame({
    "Period (Hour)": periods_win,
    "ZH_Apparent_Resistivity": ZH_rho_mean,
    "ZH_Rho_Error": ZH_rho_std,
    "ZH_Phase": ZH_phase_mean,
    "ZH_Phase_Error": ZH_phase_std
})
ZH_csv_filename = "A15_ZH_SlidingWindows_Results.csv"
ZH_df.to_csv(ZH_csv_filename, index=False)
print(f"Saved ZH-based sliding window results to {ZH_csv_filename}")

# For Q-based method:
Q_df = pd.DataFrame({
    "Period (Hour)": periods_win,
    "Q_Apparent_Resistivity": Q_rho_mean,
    "Q_Rho_Error": Q_rho_std,
    "Q_Phase": Q_phase_mean,
    "Q_Phase_Error": Q_phase_std
})
Q_csv_filename = "A15_Q_SlidingWindows_Results.csv"
Q_df.to_csv(Q_csv_filename, index=False)
print(f"Saved Q-based sliding window results to {Q_csv_filename}")

# For C-response of ZH estimation:
C_ZHdf = pd.DataFrame({
    "Period (Hour)": periods_win,
    "C_real":  np.real(ZH_C_mean),
    "C_real_Error":np.abs(ZH_C_stdR),
    "C_imag": np.imag(ZH_C_mean),
    "C_imag_Error": np.abs(ZH_C_stdI)
})
CZH_csv_filename = "A15_ZH_C_Response.csv"
C_ZHdf.to_csv(CZH_csv_filename, index=False)
print(f"Saved ZH-based C-response to {CZH_csv_filename}")

# For C-response of Q-estimation:
C_Qdf = pd.DataFrame({
    "Period (Hour)": periods_win,
    "C_real":  np.real(Q_C_mean),
    "C_real_Error":np.abs(Q_C_stdR),
    "C_imag": np.imag(Q_C_mean),
    "C_imag_Error": np.abs(Q_C_stdI)
})
CQ_csv_filename = "A15_Q_C_Response.csv"
C_Qdf.to_csv(CQ_csv_filename, index=False)
print(f"Saved Q-based C-response to {CQ_csv_filename}")