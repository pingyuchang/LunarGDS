# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:28:09 2025

@author: pingy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline

def read_resistivity_model(file_path):
    """
    Reads a resistivity model CSV file and returns depth (in km) and resistivity values.
    """
    df = pd.read_csv(file_path)
    depth_km = df["Depth (m)"] / 1000  # Convert depth to km
    resistivity = df["Resistivity (Ω·m)"]
    return depth_km.values, resistivity.values

def interpolate_resistivity(depth, resistivity, common_depth):
    """
    Interpolates resistivity values to a common depth scale.
    """
    interpolation_func = interp1d(depth, resistivity, bounds_error=False, fill_value="extrapolate")
    return interpolation_func(common_depth)

def calculate_average_resistivity(file_paths, common_depth):
    """
    Calculates the average resistivity model from multiple CSV files.
    """
    resistivity_models = []
    for file_path in file_paths:
        depth, resistivity = read_resistivity_model(file_path)
        interpolated_resistivity = interpolate_resistivity(depth, resistivity, common_depth)
        resistivity_models.append(interpolated_resistivity)
    average_resistivity = np.mean(resistivity_models, axis=0)
    return average_resistivity

def estimate_temperature(resistivity, material):
    """
    Estimates temperature from resistivity using the Arrhenius equation for different materials.
    """
    # Fitted coefficients from experimental studies
    fit_params = {
        "Lunar Sample 12053.47": (1.4e7, 1.6),
        "Basalt": (3e7, 1.5),
        "Lunar Sample 10024.22": (3.1e6, 1.2),
            }
    
    if material not in fit_params:
        raise ValueError("Invalid material. Choose from: 'Lunar Sample 12053.47', 'Basalt', 'Lunar Sample 10024.22'")
    
    sigma_0, Ea_k = fit_params[material]
    k = 8.617e-5  # Boltzmann constant in eV/K
    
    # Convert resistivity to conductivity
    conductivity = 1 / resistivity
    
    # Estimate temperature using the Arrhenius equation
    temperature = 273 + Ea_k / (k * np.log(sigma_0 / conductivity))
    
    return temperature

# File paths for Apollo 12, 15, and 16 resistivity models
file_paths = [
    r"D:\Sat_MV\inverted_model_A12.csv",
    r"D:\Sat_MV\inverted_model_A15.csv",
    r"D:\Sat_MV\inverted_model_A16.csv"
]

# Define a common depth scale (in km)
common_depth = np.linspace(0, 1400, 20)  # From 0 to 1400 km with 500 points

# Calculate the average resistivity model
average_resistivity = calculate_average_resistivity(file_paths, common_depth)

# Estimate temperature profiles for different materials
temperature_profiles = {}
for material in ["Lunar Sample 12053.47", "Basalt", "Lunar Sample 10024.22"]:
    estimated_temp = estimate_temperature(average_resistivity, material)
    # Apply smoothing using UnivariateSpline
    spline = UnivariateSpline(common_depth, estimated_temp, s=10)
    temperature_profiles[material] = spline(common_depth)
    
# Create side-by-side plots
#fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Plot the averaged resistivity model
#axes[0].step(average_resistivity, common_depth, where='post', linestyle='-', label="Averaged Resistivity")
#axes[0].set_xscale("log")  # Log scale for resistivity
#axes[0].invert_yaxis()  # Invert y-axis so depth increases downward
#axes[0].set_xlabel("Resistivity (Ω·m)")
#axes[0].set_ylabel("Depth (km)")
#axes[0].set_ylim(1400, 200)
#axes[0].set_title("Comparison of Resistivity Models")
#axes[0].legend()
#axes[0].grid(True, which="both", linestyle="--", alpha=0.7)

# Plot the estimated temperature profiles
#for material, temperature in temperature_profiles.items():
#    axes[1].plot(temperature, common_depth, label=f"{material}")
#axes[1].invert_yaxis()  # Invert y-axis so depth increases downward
#axes[1].set_xlabel("Temperature (K)")
#axes[1].set_ylabel("Depth (km)")
#axes[1].set_ylim(1400, 200)
#axes[1].set_title("Estimated Temperature Profiles vs. Depth")
#axes[1].legend()
#axes[1].grid(True, which="both", linestyle="--", alpha=0.7)

# Show the side-by-side plots
#plt.show()

# Define Vp and Vs depth arrays
vp_depths = np.array([200, 400, 600, 750, 800, 1000, 1200, 1250, 1350, 1400])
vp_lower = np.array([7.5, 7.5, 7.5, 7.8, 7.8, 7.8, 7.5, 6, 5.5, 4.5])
vp_upper = np.array([7.8, 7.8, 8, 8.2, 8.5, 8.5, 8.5, 8.5, 8.5, 7.5])

vs_depths = np.array([200, 400, 600, 800, 1000, 1150, 1200, 1250, 1300, 1350, 1400])
vs_lower = np.array([4.0, 4.1, 4.2, 4.2, 4.3, 3, 3, 2.5, 0.5, 0.5, 0.5])
vs_upper = np.array([4.5, 4.6, 4.7, 4.7, 4.8, 4.9, 5, 4.8, 4.8, 3.5, 3.5])

# Define deep moonquake range for the near side
dmq_near_side = [600, 1000]  # Depth range in km

# Create side-by-side plots with adjusted widths
fig, axes = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1, 2]})

# Plot the averaged resistivity model
axes[0].step(average_resistivity, common_depth, where='post', linestyle='-', label="Averaged Resistivity")
axes[0].axhspan(dmq_near_side[0], dmq_near_side[1], color='red', alpha=0.1)
axes[0].set_xscale("log")  # Log scale for resistivity
axes[0].invert_yaxis()  # Invert y-axis so depth increases downward
axes[0].set_xlabel("Resistivity (Ω·m)")
axes[0].set_ylabel("Depth (km)")
axes[0].set_ylim(1400, 200)
axes[0].set_title("Comparison of Resistivity Models")
axes[0].legend()
axes[0].grid(True, which="both", linestyle="--", alpha=0.7)

# Plot the estimated temperature profiles
for material, temperature in temperature_profiles.items():
    axes[1].plot(temperature, common_depth, linestyle="--", label=f"{material}")
axes[1].axhspan(dmq_near_side[0], dmq_near_side[1], color='red', alpha=0.1)
axes[1].invert_yaxis()  # Invert y-axis so depth increases downward
axes[1].set_xlabel("Temperature (K)")
axes[1].set_ylabel("Depth (km)")
axes[1].set_ylim(1400, 200)
axes[1].set_title("Estimated Temperature Profiles vs. Depth")
axes[1].legend()
axes[1].grid(True, which="both", linestyle="--", alpha=0.7)

# Plot Vp and Vs with depth
axes[2].fill_betweenx(vp_depths, vp_lower, vp_upper, step='pre', color='blue', alpha=0.3, label="Vp Range")
axes[2].fill_betweenx(vs_depths, vs_lower, vs_upper, step='pre', color='green', alpha=0.3, label="Vs Range")
axes[2].step(vp_lower, vp_depths, 'k--', where='post', label="Vp Lower Bound")
axes[2].step(vp_upper, vp_depths, 'k--', where='post', label="Vp Upper Bound")
axes[2].step(vs_lower, vs_depths, 'k--', where='post', label="Vs Lower Bound")
axes[2].step(vs_upper, vs_depths, 'k--', where='post', label="Vs Upper Bound")

# Highlight the deep moonquake region for the near side
axes[2].axhspan(dmq_near_side[0], dmq_near_side[1], color='red', alpha=0.1, label="Near-Side DMQs")

axes[2].invert_yaxis()  # Invert y-axis so depth increases downward
axes[2].set_xlabel("Velocity (km/s)")
axes[2].set_ylabel("Depth (km)")
axes[2].set_ylim(1400, 200)
axes[2].set_title("Vp and Vs with Depth")
axes[2].legend()
axes[2].grid(True, which="both", linestyle="--", alpha=0.7)

# Show the side-by-side plots
plt.show()
