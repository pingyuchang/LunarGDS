# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:08:32 2025

@author: pingy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import statsmodels.api as sm
#from scipy.signal import coherence
# =========================
# Load Apollo 16 Data
# =========================
file_path = "D:/Sat_MV/AP12_1969_1min.csv"
data = pd.read_csv(file_path, parse_dates=True, index_col=0,
                   sep=',').iloc[5520:8060]#.iloc[5520:6960]

# Downsample data
Decimate_factor = 1
data = data.iloc[::Decimate_factor]

# Define constants
miu0 = 4 * np.pi / 10**7  # Magnetic permeability
rLunar = 1737  # Moon radius (km)
fs = 1 / 60 / Decimate_factor  # Sampling frequency
dt = 1 / fs  # Sampling interval
Coh_threshold = 0.6  # Coherence threshold

# Define time array
endtime = len(data) * dt
tt = np.arange(0, endtime, dt)

# Subtract remanent field offsets
offsets = {'BX': -25.8, 'BY': 11.9, 'BZ': -25.8}
for key in offsets:
    data[key] -= offsets[key]

# =========================
# Rotate the Magnetic Field to Maximize Bz (Applied to Entire DataFrame)
# =========================
def apply_rotation(data):
    """Apply rotation to the entire dataset to maximize Bz."""
    By_mean = np.mean(data['BY'])
    Bz_mean = np.mean(data['BZ']) 
    thetaRR = np.arctan2(By_mean, Bz_mean)  # Rotation angle from By and Bz means

    # Rotation matrix
    cos_theta = np.cos(thetaRR)
    sin_theta = np.sin(thetaRR)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Apply transformation to all BY and BZ values
    rotated = R @ np.vstack((data['BY'].values, data['BZ'].values))

    # Update the DataFrame with rotated values
    data['BY'] = rotated[0]
    data['BZ'] = rotated[1]   

    return data  # Return modified DataFrame

# Apply rotation once to the entire dataset
data = apply_rotation(data)

# Decimate after rotation correction
data = data.apply(lambda x: signal.decimate(x, Decimate_factor))

# =========================
# Pre-whitening (First-order differencing)
# =========================
def prewhiten(data):
    """Apply first-order differencing to prewhiten the data."""
    return data.diff().dropna()

# Apply pre-whitening to each channel (column-wise)
data = data.apply(prewhiten)

# =========================
# Sliding Window Analysis using Welch's Method
# =========================
n_segments = 42  # Desired number of time windows
overlap_percentage = 0.5  # 50% overlap

# Compute the step size between windows
# Note: If segment_length is L, and step = L * (1 - overlap), then:
# Total length needed = L + (n_segments - 1) * step
# So: L + (n_segments - 1) * L * (1 - overlap) <= data length
# => L * [1 + (n_segments - 1)(1 - overlap)] <= data length

denominator = 1 + (n_segments - 1) * (1 - overlap_percentage)
max_segment_length = int(len(data['BX']) / denominator)
step = int(max_segment_length * (1 - overlap_percentage))

# Generate start indices
start_indices = np.arange(0, len(data['BX']) - max_segment_length + 1, step)

print(f"[INFO] Segment length: {max_segment_length}, Step: {
      step}, Segments: {len(start_indices)}")

# Dictionary to store Welch's power spectral estimates
welch_dict = {}

# Loop through segments and compute Welch’s Power Spectrum
for start in start_indices:
    end = start + max_segment_length

    if end > len(data['BX']):
        break  # Avoid going out of bounds

    # Extract segment data
    segment_Bz = data['BX'].iloc[start:end].values
    segment_Bx = data['BZ'].iloc[start:end].values #transfer ME coordinate

    # Apply Welch’s Method for spectral estimation
    freqs, Pxx = signal.welch(
        segment_Bx, fs, window='hann', nperseg=len(segment_Bx)//3)
    _, Pzz = signal.welch(segment_Bz, fs, window='hann',
                          nperseg=len(segment_Bx)//3)
    _, Pxz = signal.csd(segment_Bz, segment_Bx, fs,
                        window='hann', nperseg=len(segment_Bx)//3)
    _, Cxz = signal.coherence(segment_Bz, segment_Bx, fs, window='hann',
                              nperseg=len(segment_Bx)//3)
    
        # Store power spectra per frequency
    for i, f in enumerate(freqs):
        if f > 0:
            if f not in welch_dict:
                welch_dict[f] = {'Pxx': [], 'Pzz': [], 'Pxz': [], 'Cxz': []}
            welch_dict[f]['Pxx'].append(Pxx[i])
            welch_dict[f]['Pzz'].append(Pzz[i])
            welch_dict[f]['Pxz'].append(Pxz[i])
            welch_dict[f]['Cxz'].append(Cxz[i])
# =========================
# Least Squares Estimation of TF using Welch's Method
# =========================
freqs_list = sorted(welch_dict.keys())  # Sort frequencies

TF_estimates = []
TF_errors = []
Coherence_P = []
TFR_errors = []
TFI_errors = []

for f in freqs_list:
    Pxx_all = np.array(welch_dict[f]['Pxx'])  # Collect all Pxx for this frequency
    Pxz_all = np.array(welch_dict[f]['Pxz'])  # Collect all Pxz for this frequency
    Pzz_all = np.array(welch_dict[f]['Pzz'])  # Collect all Pxz for this frequency
    Cxz_all = np.array(welch_dict[f]['Cxz'])  # Collect all Cxz for this frequency
    # =========================
    # Compute Coherence
    # =========================
    #coherence = np.median(Cxz_all)  # average all segments
    
    # =========================
    # M-estimator for Transfer Function Estimation using statsmodels
    # =========================
    def m_estimator_tf(Pxz, Pxx):
        eps = 1e-12
        N = len(Pxx)

        if N == 0:
            return np.nan + 1j * np.nan, np.nan, np.nan, np.nan

        elif N == 1:
            tf = Pxz[0] / (Pxx[0] + eps)
            return tf, 0.0, 0.0, 0.0

        elif N == 2:
            tf_vals = Pxz / (Pxx + eps)
            tf = np.mean(tf_vals)
            tf_std_error = np.std(tf_vals) / np.sqrt(N)
            return tf, tf_std_error, tf_std_error, tf_std_error

        else:
            # Robust regression using Huber's M-estimator
            X_real = Pxx.real + eps
            y_real = Pxz.real
            model_real = sm.RLM(y_real, sm.add_constant(X_real), M=sm.robust.norms.HuberT()).fit()
            tf_real = model_real.params[1]
            tf_real_std = model_real.bse[1]

            y_imag = Pxz.imag
            model_imag = sm.RLM(y_imag, sm.add_constant(X_real), M=sm.robust.norms.HuberT()).fit()
            tf_imag = model_imag.params[1]
            tf_imag_std = model_imag.bse[1]

            tf_estimate = tf_real + 1j * tf_imag
            tf_std_error = np.sqrt(tf_real_std**2 + tf_imag_std**2)

            return tf_estimate, tf_std_error, tf_real_std, tf_imag_std

    # Apply M-estimator instead of mean for TF estimation
    TF_freq, TF_freq_std, tfr_std, tfi_std = m_estimator_tf(Pxz_all, Pxx_all)
    
    # Append results
    TF_estimates.append(TF_freq)
    TF_errors.append(TF_freq_std) # Store standard deviation as error
    TFR_errors.append(tfr_std)
    TFI_errors.append(tfi_std)
    #Coherence_P.append(coherence)

# Convert to numpy arrays
TF_estimates = np.array(TF_estimates)
TFR_errors = np.array(TFR_errors)
TFI_errors = np.array(TFI_errors)
#Coherence_P = np.array(Coherence_P)
Coherence_P = np.array(Cxz)

# =========================
# Fix for Frequency Length Mismatch
# =========================
valid_idx = np.where(np.array(freqs_list) > 0)[0]  # Get valid frequencies

# Ensure arrays have correct shapes before computation
TF_estimates_filtered = np.array(TF_estimates)[valid_idx]
TF_errors_filtered = np.array(TF_errors)[valid_idx]
freqs_filtered = np.array(freqs_list)[valid_idx]
coherence_filtered = np.array(Coherence_P)[valid_idx]
TFR_errors_filtered = np.array(TFR_errors)[valid_idx]
TFI_errors_filtered = np.array(TFI_errors)[valid_idx]

# Compute C-response
Theta12 = np.radians(138.09) 
Cr12 = (-1 / 2) * rLunar * np.tan(Theta12) * TF_estimates_filtered

# Compute error propagation
Cr12_err = 1 * np.abs(Cr12) * (TF_errors_filtered / np.abs(TF_estimates_filtered))

# =========================
# Compute Apparent Resistivity
# =========================
def compute_apparent_resistivity(freqs, C_response, C_response_err, mu0=4 * np.pi * 1e-7):
    abs_C = np.abs(C_response)
    abs_C_err = np.abs(C_response_err)

    rho_a = 1e6 * 2 * np.pi * mu0 * freqs * abs_C**2
    rho_a_err = 1e6 *2 * np.pi * mu0 * freqs * 2 * abs_C_err * abs_C

    return rho_a, rho_a_err

# Compute apparent resistivity and its error
rho_a, rho_a_err = compute_apparent_resistivity(freqs_filtered, Cr12, Cr12_err)

# Filter out values where coherence is above the threshold
valid_Coh = np.where(coherence_filtered > Coh_threshold)[0]

# Apply filtering
filtered_freqs = freqs_filtered[valid_Coh]
filtered_Cr12 = Cr12[valid_Coh]
filtered_Cr12_err = Cr12_err[valid_Coh]
filtered_rho_a = rho_a[valid_Coh]
filtered_rho_a_err = rho_a_err[valid_Coh]
filtered_coherence = coherence_filtered[valid_Coh]
filtered_TFR_err = TFR_errors_filtered[valid_Coh]
filtered_TFI_err = TFI_errors_filtered[valid_Coh]

# Compute phase and its error (C_response phase if Z impedance:+90 degrees)
filtered_phase = np.angle(filtered_Cr12, deg=True)  # Phase in degrees
filtered_phase_err =filtered_phase * np.abs(filtered_Cr12_err)/np.abs(filtered_Cr12)

# =========================
# Plot C-response
# =========================
T_lower=0.05 # Lower period limit in Hour
plt.figure(figsize=(10, 5))
plt.errorbar(1 / filtered_freqs.ravel() / 3600, filtered_Cr12.real, 
             yerr= np.abs(filtered_Cr12_err.real), fmt='o', label='Real')
plt.errorbar(1 / filtered_freqs.ravel() / 3600, filtered_Cr12.imag, 
             yerr= np.abs(filtered_Cr12_err.real), fmt='ro', label='Imaginary')

plt.xscale('log')
plt.xlim(T_lower, 12)
plt.ylim(-500, 2000)
plt.xlabel('Period (Hours)')
plt.ylabel('C-response (km)')
plt.title('Apollo 12 C-response with Error (Welch’s Method)')
plt.legend()
plt.grid()
plt.show()

# =========================
# Plot Coherence
# =========================
plt.figure(figsize=(10, 5))
plt.semilogx(1 / filtered_freqs.ravel() / 3600, filtered_coherence, 'o', label='Coherence')
plt.xlabel('Period (Hours)')
plt.ylabel('Coherence')
plt.xlim(T_lower, 12)
plt.ylim(0, 1.1)  # Ensure coherence values are between 0 and 1
plt.title('Coherence')
plt.grid()
plt.legend()
plt.show()

# =========================
# Plot Apparent Resistivity
# =========================
plt.figure(figsize=(10, 5))
plt.errorbar(1 / filtered_freqs.ravel() / 3600, filtered_rho_a.ravel(), yerr=np.abs(filtered_rho_a_err.ravel()), fmt='o')

plt.xscale('log')
#plt.yscale('log')
plt.xlim(T_lower, 12)
#plt.ylim(0, 4000)
plt.xlabel('Period (Hours)')
plt.ylabel('Apparent Resistivity (Ω·m)')
plt.title('Apollo 12 Apparent Resistivity with Error')
plt.grid()
plt.show()

# =========================
# Plot Phase
# =========================
plt.figure(figsize=(10, 5))
plt.errorbar(1 / filtered_freqs.ravel() / 3600, filtered_phase.ravel(), yerr=np.abs(filtered_phase_err.ravel()), fmt='o')

plt.xscale('log')
plt.xlim(T_lower, 12)
plt.ylim(-180, 180)
plt.xlabel('Period (Hours)')
plt.ylabel('Phase (Degree)')
plt.title('Apollo 12 Phase with Error')
plt.grid()
plt.show()
# =========================
# Save Data
# =========================
def save_c_response_results(
    base_filename,
    freqs,
    Cr12_real,
    Cr12_real_err,
    Cr12_imag,
    Cr12_imag_err,
    coherence,
    rho_a,
    rho_a_err,
    phase,
    phase_err,
    T_lower
):
    """
    Save C-response analysis results to CSV for periods greater than T_lower (in hours).
    Filename will be formatted as: base_filename_T_lowerHr_out.csv
    """
    # Convert frequency to period in hours
    period_hours = 1 / freqs / 3600

    # Filter by lower period bound
    valid = period_hours >= T_lower

    # Create DataFrame
    df = pd.DataFrame({
        "Period_hr": period_hours[valid],
        "Freq_Hz": freqs[valid],
        "C_real_km": Cr12_real[valid],
        "C_real_err_km": Cr12_real_err[valid],
        "C_imag_km": Cr12_imag[valid],
        "C_imag_err_km": Cr12_imag_err[valid],
        "Coherence": coherence[valid],
        "AppRes_ohm_m": rho_a[valid],
        "AppRes_err_ohm_m": rho_a_err[valid],
        "Phase_deg": phase[valid],
        "Phase_err_deg": phase_err[valid],
    })

    # Format filename
    filename = f"{base_filename}_{T_lower}Hr_out.csv"
    df.to_csv(filename, index=False)
    print(f"Saved C-response results to: {filename}")
    
save_c_response_results(
    base_filename="Apollo12_results",
    freqs=filtered_freqs,
    Cr12_real=filtered_Cr12.real,
    Cr12_real_err=np.abs(filtered_Cr12_err.real),
    Cr12_imag=filtered_Cr12.imag,
    Cr12_imag_err=np.abs(filtered_Cr12_err.imag),
    coherence=filtered_coherence,
    rho_a=filtered_rho_a,
    rho_a_err=filtered_rho_a_err,
    phase=filtered_phase,
    phase_err=filtered_phase_err,
    T_lower=T_lower
)