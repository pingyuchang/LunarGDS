# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:22:32 2025

@author: pingy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import math
import cmath

def read_input_csv(file_path):
    df = pd.read_csv(file_path)
    required_columns = {"Period (Hour)","ZH_Apparent_Resistivity","ZH_Rho_Error","ZH_Phase", "ZH_Phase_Error"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")
    
    df = df.iloc[0:116] #if only look at the rows from first 1 to 57 of data
    
    periods = df["Period (Hour)"].values * 3600
    apparent_resistivity = df["ZH_Apparent_Resistivity"].values
    rho_error = df["ZH_Rho_Error"].values   
    phase = df["ZH_Phase"].values
    error_phase = df["ZH_Phase_Error"].values
    log10_resistivity = np.log10(apparent_resistivity)
    error_log10_resistivity = np.log10(rho_error)
    return np.column_stack([periods, log10_resistivity, error_log10_resistivity, phase, error_phase])

def generate_initial_model(data, n_layers=20, max_depth=1737000):
    layer_depth = np.linspace(0, max_depth, n_layers + 1)
    log10_res = np.linspace(np.log10(2e4), np.log10(1e1), n_layers)
    return layer_depth[:-1], log10_res

def forward_model(freq, log10_res, z, opt='rho'):
    mu0 = 4.0 * math.pi * 1E-7
    omega = 2 * math.pi * freq
    n_layers = len(z)
    Z = np.zeros(len(freq), dtype=complex)
    resistivities = 10 ** log10_res
    
    for ifreq, w in enumerate(omega):
        impedances = [0] * n_layers
        impedances[n_layers - 1] = cmath.sqrt(w * mu0 * resistivities[-1] * 1j)
        
        for j in range(n_layers - 2, -1, -1):
            resistivity = resistivities[j]
            thickness = z[j + 1] - z[j] if j < n_layers - 1 else z[j]
            dj = cmath.sqrt((w * mu0 * (1 / resistivity)) * 1j)
            wj = dj * resistivity
            ej = cmath.exp(-2 * thickness * dj)
            belowImpedance = impedances[j + 1]
            rj = (wj - belowImpedance) / (wj + belowImpedance)
            re = rj * ej
            Zj = wj * ((1 - re) / (1 + re))
            impedances[j] = Zj
        
        Z[ifreq] = impedances[0]
    
    absZ = np.abs(Z)
    rho = (absZ ** 2) / (mu0 * omega)
    phase = np.angle(Z, deg=True)
    
    return (np.log10(rho), phase) if opt == 'rho' else Z

def plot_observed_vs_estimated(periods, observed_log10_rho, initial_log10_rho, optimized_log10_rho):
    plt.figure(figsize=(10, 6))
    plt.errorbar(periods / 3600, 10**observed_log10_rho, yerr=10**error_log10_rho, fmt='o', label='Observed', color='blue', capsize=5)
    plt.plot(periods / 3600, 10**initial_log10_rho, 'r--', label="Initial Model")
    plt.plot(periods / 3600, 10**optimized_log10_rho, 'g-', label="Final Estimated Model")
    plt.xlim(0.1, 3)
    plt.ylim(1e2, 1e5)    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Period (Hour)')
    plt.ylabel('Apparent Resistivity (Ω·m)')
    plt.title('Comparison of Observed, Initial, and Final Estimated Apparent Resistivity')
    plt.legend()
    plt.grid(True, which='both')
    plt.show()

def plot_initial_final_models(layer_depth, log10_res, optimized_log10_res):
    plt.figure(figsize=(6, 8))
    plt.step(log10_res, layer_depth / 1000, where='post', label='Initial Model', linestyle='--', color='r')
    plt.step(optimized_log10_res, layer_depth / 1000, where='post', label='Final Model', linestyle='-', color='g')
    plt.gca().invert_yaxis()
    plt.ylim(1200,0)
    plt.xlabel("Log10 Resistivity (Ohm-m)")
    plt.ylabel("Depth (km)")
    plt.title("Comparison of Initial and Final Resistivity Models")
    plt.legend()
    plt.grid(True)
    plt.show()

def save_inverted_model_to_csv(layer_depth, optimized_log10_res, output_file="inverted_model.csv"):
    """
    Saves the inverted resistivity model to a CSV file with depth and resistivity.
    
    :param layer_depth: Depth values of the model.
    :param optimized_log10_res: Optimized resistivity values (log10 scale).
    :param output_file: File path for saving the CSV.
    """
    resistivity = 10 ** optimized_log10_res  # Convert log10 resistivity to linear scale
    df = pd.DataFrame({
        "Depth (m)": layer_depth,
        "Resistivity (Ω·m)": resistivity
    })
    df.to_csv(output_file, index=False)
    print(f"Inverted model saved to {output_file}")
    
def objective_function_with_smoothness(log10_res):
    modeled_log10_rho, _ = forward_model(1 / periods, log10_res, layer_depth, opt='rho')
    rho_misfit = np.sum(((observed_log10_rho - modeled_log10_rho) / error_log10_rho ) ** 2)  # Fixed error scaling
    rms_misfit = np.sqrt(rho_misfit / len(observed_log10_rho))
    smoothness_penalty = np.sum(np.diff(log10_res, n=2)**2)
    total_misfit = rms_misfit + lambda_smooth * smoothness_penalty

    print(f"Misfit: {total_misfit:.4f}, Rho Misfit: {rho_misfit:.4f}")

    return total_misfit

# Model Initialization
lambda_smooth = 1e-3
#wr = 1
misfit_threshold = 0.05  # Stop optimization if misfit is already small

input_file = r"D:\Sat_MV\A12_ZH_SlidingWindows_Results.csv"
data = read_input_csv(input_file)
periods, observed_log10_rho, error_log10_rho, observed_phs, error_phs = data.T
layer_depth, log10_res = generate_initial_model(data, n_layers=21)

# Compute Initial Misfit
initial_misfit = objective_function_with_smoothness(log10_res)
print(f"Initial Misfit: {initial_misfit}")

if initial_misfit < misfit_threshold:
    print("Initial model is already a good fit. Skipping optimization.")
    optimized_log10_res = log10_res
else:
    print("Running optimization...")
    result = minimize(objective_function_with_smoothness, log10_res, method='L-BFGS-B',
                      bounds=[(-1, 8)] * len(log10_res),
                      options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True})
    optimized_log10_res = result.x

# Print Final Misfit for Debugging
final_misfit = objective_function_with_smoothness(optimized_log10_res)
print(f"Final Misfit: {final_misfit}")

# Save the inverted model to CSV
save_inverted_model_to_csv(layer_depth, optimized_log10_res, "inverted_model.csv")

# Plot Results
plot_initial_final_models(layer_depth, log10_res, optimized_log10_res)
initial_log10_rho, _ = forward_model(1 / periods, log10_res, layer_depth, opt='rho')
optimized_log10_rho, _ = forward_model(1 / periods, optimized_log10_res, layer_depth, opt='rho')
plot_observed_vs_estimated(periods, observed_log10_rho, initial_log10_rho, optimized_log10_rho)
