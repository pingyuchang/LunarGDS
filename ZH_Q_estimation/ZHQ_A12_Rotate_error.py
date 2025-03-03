# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:46:38 2025

@author: pingy
All calculations are based on the ME coordinate system
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats
from scipy.signal import welch,stft
# =========================
# Original Code (Unchanged)
# =========================

# Read Apollo 12 data
kny = pd.read_csv("D:/Sat_MV/AP12_1969_1min.csv", sep=',').iloc[5520:6960] 
Decimate_factor=1
kny = kny.iloc[::Decimate_factor]  # Downsample: Keep every 180th row

# Sampling frequency and time calculations
"""..........Define parameters................"""
miu0 = 4 * np.pi / 10**7
rLunar = 1737  # Moon radius in km
fs = 1 / 60 / Decimate_factor   # Sampling frequency (1 sample per minute)
dt = 1 / fs   # Sampling time
Coh12_threshold = 0.9  # Coherence threshold for C-response
Q12_threshold = 0.9    # Coherence threshold for Q-response
seg = 1
segnum = int(len(kny) / seg)
overlapnum = int(segnum / 8)

# Adjust nperseg to avoid warnings
nperseg = min(segnum, len(kny))

"""""..........Read Apollo 12 data (5520-6960 Rows 1969-11-23)................"""
# Rename problematic columns 'min' and 'sec' to avoid conflicts
kny.rename(columns={'min': 'minute', 'sec': 'second'}, inplace=True)

# Combine 'year', 'month', 'day', 'hour', 'minute', and 'second' into a single timestamp
kny['timestamp'] = pd.to_datetime(kny[['year', 'month', 'day', 'hour', 'minute', 'second']])
# Calculate the time difference (in minutes) from the starting timestamp
tt = (kny['timestamp'] - kny['timestamp'].iloc[0]).dt.total_seconds() / 60

# Read three-axis magnetic field vectors X, Y, Z (adding offsets)
knyX = signal.medfilt(kny['BX'] + 25.8 , 1)
knyY = signal.medfilt(kny['BY'] - 11.9 , 1)
knyZ = signal.medfilt(kny['BZ'] + 25.8 , 1)
# According to Dyal et al. (1973) remanent field

def rotate_to_max_bz(Bx, By, Bz):
    """
    Rotates (Bx, By, Bz) such that Bz is maximized and By is minimized, keeping Bx unchanged.
    Works for both scalar and array inputs.
    
    Parameters:
    Bx, By, Bz : array-like
        Magnetic field components in the original coordinate system (can be scalars or arrays).

    Returns:
    Bx_new, By_new, Bz_new : array-like
        Rotated magnetic field components.
    """
    # Convert inputs to numpy arrays
    Bx, By, Bz = np.asarray(Bx), np.asarray(By), np.asarray(Bz)

    # Compute the rotation angle (theta) for each element
    thetaRR = np.arctan2(By, Bz)  # Shape: (N,)

    # Construct the rotation matrix in vectorized form
    cos_theta = np.cos(thetaRR)
    sin_theta = np.sin(thetaRR)

    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])  # Shape: (2, 2, N)

    # Stack By and Bz for transformation
    B_vec = np.vstack((By, Bz))  # Shape: (2, N)

    # Perform element-wise rotation using broadcasting
    By_new, Bz_new = np.einsum('ijk,jk->ik', R, B_vec)  # Properly aligns dimensions

    # Bx remains unchanged
    return Bx, By_new, Bz_new

# Rotate the magnetic field
knyX, knyY, knyZ = rotate_to_max_bz(knyX, knyY, knyZ)

# Apply perturbation
perturb_Mag =100
knyX += perturb_Mag
knyY += perturb_Mag
knyZ += perturb_Mag

# **Subtract DC component**
#knyX -= np.mean(knyX)
#knyY -= np.mean(knyY)
#knyZ -= np.mean(knyZ)

# Plot the variation of X, Y, Z magnetic fields over time (in minutes)
plt.figure(figsize=(15, 6))
plt.plot(tt / 60, knyX, label='Magnetic Field X', alpha=0.8)
plt.plot(tt / 60, knyY, label='Magnetic Field Y', alpha=0.8)
plt.plot(tt / 60, knyZ, label='Magnetic Field Z', alpha=0.8)
plt.title("Magnetic Field Variation Over Time (Apollo12)")
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
abgle12 = 138.09-3
Theta12 = abgle12 * np.pi / 180
#Rphi12 = 5.95 * np.pi / 180
Rphi12 = np.arctan(-1 * np.cos(Theta12) * KnymeanY / KnymeanZ)

# Calculate cross power spectral densities and coherence
f_coh, Cxy = signal.coherence(knyX, knyZ, fs, window='hann', nperseg=nperseg, noverlap=overlapnum)

# Compute C-response
fx12, tx12, Hxx12 = signal.stft(knyX, fs, window='hann', nperseg=nperseg,
                                noverlap=overlapnum, return_onesided=False)
fz12, tz12, Hzz12 = signal.stft(knyZ, fs, window='hann', nperseg=nperseg,
                                noverlap=overlapnum, return_onesided=False)

with np.errstate(divide='ignore', invalid='ignore'):
    Hxx12_conj = np.conj(Hxx12)
    Hzz12_valid = np.where(np.abs(Hzz12) > 1e-10, Hzz12, np.nan)
    Cr12 = (-1 / 2) * np.tan(Theta12) * rLunar * np.nanmedian(Hxx12 / Hzz12_valid, axis=1)

Cxy_interp = np.interp(fx12, f_coh, Cxy)

valid_indices = (fx12 != 0) & (Cxy_interp >= Coh12_threshold)
fx12 = fx12[valid_indices]
Cr12 = Cr12[valid_indices]

plt.semilogx(1 / fx12 / 3600, signal.medfilt(Cr12.real, 1), 'o', label='real')
plt.semilogx(1 / fx12 / 3600, signal.medfilt(Cr12.imag, 1), 'ro', label='imaginary')
plt.title('Apollo12 ZH-based C-response (Filtered by Coherence)')
plt.xlabel('Time (Hour)')
plt.ylabel('C-response (km)')
plt.xlim(0.1, 12)
plt.ylim(-1000, 2000)
plt.legend()
plt.show()


#--------------------------------------

# Compute C-response
fx12, tx12, Hxx12 = signal.stft(knyX, fs, window='hann', nperseg=nperseg,
                                noverlap=overlapnum, return_onesided=False)
fz12, tz12, Hzz12 = signal.stft(knyZ, fs, window='hann', nperseg=nperseg,
                                noverlap=overlapnum, return_onesided=False)

with np.errstate(divide='ignore', invalid='ignore'):
    Hxx12_conj = np.conj(Hxx12)
    Hzz12_valid = np.where(np.abs(Hzz12) > 1e-10, Hzz12, np.nan)
    Cr12 = (-1 / 2) * np.tan(Theta12) * rLunar * np.nanmedian(Hxx12 / Hzz12_valid, axis=1)

Cxy_interp = np.interp(fx12, f_coh, Cxy)

valid_indices = (fx12 != 0) & (Cxy_interp >= Coh12_threshold)
fx12 = fx12[valid_indices]
Cr12 = Cr12[valid_indices]

plt.semilogx(1 / fx12 / 3600, signal.medfilt(Cr12.real, 1), 'o', label='real')
plt.semilogx(1 / fx12 / 3600, signal.medfilt(Cr12.imag, 1), 'ro', label='imaginary')
plt.title('Apollo12 ZH-based C-response (Filtered by Coherence)')
plt.xlabel('Time (Hour)')
plt.ylabel('C-response (km)')
plt.xlim(0.1, 12)
plt.ylim(-1000, 2000)
plt.legend()
plt.show()

# Q-response Calculation
def find_best_solution(A, b):
    solution, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return solution

array121 = np.array([[-np.cos(Rphi12) * np.sin(Theta12), 2 * np.cos(Rphi12) * np.sin(Theta12)],
                     [-np.cos(Rphi12) * np.cos(Theta12), -np.cos(Rphi12) * np.cos(Theta12)],
                     [np.sin(Rphi12), np.sin(Rphi12)]])
array122 = np.array([knyX, knyY, knyZ])
q12, g12 = find_best_solution(array121, array122)

fq12, tq12, qq12 = signal.stft(q12, fs, window='hann', nperseg=nperseg,
                               noverlap=overlapnum, return_onesided=False)
fg12, tg12, gg12 = signal.stft(g12, fs, window='hann', nperseg=nperseg,
                               noverlap=overlapnum, return_onesided=False)

f_coh_q, Qxy = signal.coherence(np.mean(qq12, axis=1), np.mean(gg12, axis=1), fs)

Qxy_interp = np.interp(fq12, f_coh_q, Qxy)

valid_indices_q = (fq12 != 0) & (Qxy_interp >= Q12_threshold)
fq12 = fq12[valid_indices_q]
qq12 = qq12[valid_indices_q, :]
gg12 = gg12[valid_indices_q, :]

qq12 = np.mean(qq12, axis=1)
gg12 = np.mean(gg12, axis=1)
Q12 = gg12 / qq12
C_qresponse12 = (-rLunar / 2) * (1 - 2 * Q12) / (1 + Q12)

plt.semilogx(1 / fq12 / 3600, signal.medfilt(np.real(C_qresponse12), 1), 'o', label='real')
plt.semilogx(1 / fq12 / 3600, signal.medfilt(np.imag(C_qresponse12), 1), 'ro', label='imaginary')
plt.legend()
plt.title('Apollo12 Q-based C-response (Filtered by Coherence)')
plt.xlabel('Time (Hour)')
plt.ylabel('C-response (km)')
plt.xlim(0.1, 12)
plt.ylim(-1000, 2000)
plt.show()

# Function for combined apparent resistivity and phase (unchanged)
def calculate_and_plot_combined_apparent_resistivity_and_phase(
    frequencies_zh, C_response_zh, frequencies_q, C_response_q):
    periods_zh = 1 / frequencies_zh / 3600
    periods_q = 1 / frequencies_q / 3600

    rho_a_zh = 1e6 * 2 * np.pi * frequencies_zh * miu0 * np.abs(C_response_zh) ** 2
    rho_a_q = 1e6 * 2 * np.pi * frequencies_q * miu0 * np.abs(C_response_q) ** 2

    phase_zh = 0 + np.angle(C_response_zh, deg=True)
    phase_q = 0 + np.angle(C_response_q, deg=True)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.semilogx(periods_zh, rho_a_zh, 'o', label='ZH-based Apparent Resistivity', alpha=0.8)
    plt.semilogx(periods_q, rho_a_q, 'ro', label='Q-based Apparent Resistivity', alpha=0.8)
    plt.xlabel('Period (Hour)')
    plt.ylabel('Apparent Resistivity (Ω·m)')
    plt.title('Apollo12')
    plt.xlim(0.1, 12)
    plt.ylim(0, 10000)
    plt.grid()
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.semilogx(periods_zh, phase_zh, 'o', label='ZH-based Phase', alpha=0.8)
    plt.semilogx(periods_q, phase_q, 'ro', label='Q-based Phase', alpha=0.8)
    plt.xlabel('Period (Hour)')
    plt.ylabel('Phase (degrees)')
    plt.title('Apollo12')
    plt.xlim(0.1, 12)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

calculate_and_plot_combined_apparent_resistivity_and_phase(fx12, Cr12, fq12, C_qresponse12)

# =========================
# New Section: Sliding Windows Analysis
# Each window's length is 90% of the original dataset.
# We choose 5 equally spaced windows, compute the apparent resistivity and phase for each window,
# then estimate the error (standard deviation) across the windows, plot the results with error bars,
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

nperseg_win = min(int(0.6*win_len), win_len)
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
    Cr_win = (-1/2) * np.tan(Theta12) * rLunar * median_ratio_win
    ZH_rho = 1e6 * 2 * np.pi * f_win * miu0 * (np.abs(Cr_win))**2
    ZH_phase = 0 + np.angle(Cr_win, deg=True)
    ZH_C_windows.append(Cr_win)
    ZH_rho_windows.append(ZH_rho)
    ZH_phase_windows.append(ZH_phase)
    
    # ---------- Q-based processing ----------
    array121 = np.array([[-np.cos(Rphi12) * np.sin(Theta12), 2 * np.cos(Rphi12) * np.sin(Theta12)],
                          [-np.cos(Rphi12) * np.cos(Theta12), -np.cos(Rphi12) * np.cos(Theta12)],
                          [np.sin(Rphi12), np.sin(Rphi12)]])
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

# Convert frequency axis to period (in hours)
periods_win = 1 / freqs_win / 3600

# Convert the lists to arrays (shape: (5, n_freq))
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

# Function to plot C-response with error bars
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
    plt.xlim(0.1, 12)
    plt.ylim(-500, 1500)
    plt.legend()
    plt.show()

# Plot C-response with error bars for both methods
plot_c_response_with_error(periods_win, ZH_C_mean, Q_C_mean, ZH_C_stdR, ZH_C_stdI, Q_C_stdR, Q_C_stdI)

# -------------------------
# Plot separate error-bar plots for ZH-based method
# -------------------------
# ZH-based Apparent Resistivity
plt.figure(figsize=(8, 6))
plt.errorbar(periods_win, ZH_rho_mean, yerr=np.abs(ZH_rho_std), fmt='o', capsize=3, color='red', label='ZH-method')
plt.errorbar(periods_win, Q_rho_mean, yerr=np.abs(Q_rho_std), fmt='^', capsize=3, color='blue', label='Q-method')
plt.xscale('log')
plt.xlabel('Period (Hour)', fontsize=14)
plt.ylabel('Apparent Resistivity (Ω·m)', fontsize=14)
#plt.title('Apparent Resistivity with Error Bars', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(0.1, 12)
plt.ylim(0, 17000)
plt.legend()
#plt.grid(True, which="both")
plt.show()

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
ZH_csv_filename = "A12_ZH_SlidingWindows_Results.csv"
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
Q_csv_filename = "A12_Q_SlidingWindows_Results.csv"
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
CZH_csv_filename = "A12_ZH_C_Response.csv"
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
CQ_csv_filename = "A12_Q_C_Response.csv"
C_Qdf.to_csv(CQ_csv_filename, index=False)
print(f"Saved Q-based C-response to {CQ_csv_filename}")
