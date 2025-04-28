# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 16:06:52 2025

@author: pingy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline

# === Configurable option ===
use_apollo12_only = False

# === File paths ===
apollo12_path = r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo12_1D_cond.csv"
apollo15_path = r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo15+12_1D.csv"
apollo16_path = r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo16+12_1D.csv"
file_paths = [apollo12_path, apollo15_path, apollo16_path]

# === Common depth grid ===
common_depth = np.linspace(0, 1400, 20)

# === Read CSV (conductivity format) ===
def read_conductivity_with_bounds(file_path):
    df = pd.read_csv(file_path)
    depth_km = df["depth"].values
    avg = df["avg"].values
    lower95 = df["upper95"].values
    upper95 = df["lower95"].values
    return depth_km, avg, lower95, upper95

def interpolate_profile(x, y, xout):
    return interp1d(x, y, bounds_error=False, fill_value="extrapolate")(xout)

# === Apollo 12 as base ===
d12, c12, l12, u12 = read_conductivity_with_bounds(apollo12_path)

if use_apollo12_only:
    avg_cond = interpolate_profile(d12, c12, common_depth)
    low_cond = interpolate_profile(d12, l12, common_depth)
    upp_cond = interpolate_profile(d12, u12, common_depth)
else:
    shallow_idx = np.where(common_depth <= 1400)[0]
    deep_idx = np.where(common_depth > 1400)[0]
    depth_shallow = common_depth[shallow_idx]
    depth_deep = common_depth[deep_idx]

    avg_list, low_list, upp_list = [], [], []
    for path in file_paths:
        d, c, l, u = read_conductivity_with_bounds(path)
        avg_list.append(interpolate_profile(d, c, depth_shallow))
        low_list.append(interpolate_profile(d, l, depth_shallow))
        upp_list.append(interpolate_profile(d, u, depth_shallow))

    avg_cond_shallow = np.mean(avg_list, axis=0)
    low_cond_shallow = np.mean(low_list, axis=0)
    upp_cond_shallow = np.mean(upp_list, axis=0)

    avg_cond_deep = interpolate_profile(d12, c12, depth_deep)
    low_cond_deep = interpolate_profile(d12, l12, depth_deep)
    upp_cond_deep = interpolate_profile(d12, u12, depth_deep)

    avg_cond = np.concatenate([avg_cond_shallow, avg_cond_deep])
    low_cond = np.concatenate([low_cond_shallow, low_cond_deep])
    upp_cond = np.concatenate([upp_cond_shallow, upp_cond_deep])

# === Convert to resistivity and CI ===
avg_resistivity = 1 / avg_cond
avg_lower95 = 1 / upp_cond
avg_upper95 = 1 / low_cond

# === Temperature estimation aligned at 200 km ===
def estimate_temperature(resistivity, Ea_k, sigma_0):
    k = 8.617e-5
    conductivity = 1 / resistivity
    return 273 + Ea_k / (k * np.log(sigma_0 / conductivity))

fit_params = {
    "Lunar Sample 12053.47": (1.4e7, 1.6),
    "Basalt": (3e7, 1.5),
    "Lunar Sample 10024.22": (3.1e6, 1.2),
}

temperature_raw = {}
uncertainty_raw = {}

for mat, (sigma_0, Ea_k) in fit_params.items():
    temp = estimate_temperature(avg_resistivity, Ea_k, sigma_0)
    temp_low = estimate_temperature(avg_upper95, Ea_k, sigma_0)
    temp_high = estimate_temperature(avg_lower95, Ea_k, sigma_0)
    temperature_raw[mat] = temp
    uncertainty_raw[mat] = (temp_low, temp_high)

# Align all to Basalt at 200 km
depth_index_200km = np.argmin(np.abs(common_depth - 200))
T_ref = temperature_raw["Lunar Sample 12053.47"][depth_index_200km]

temperature_profiles = {}
temperature_uncertainties = {}

for mat in fit_params:
    temp = temperature_raw[mat]
    temp_low, temp_high = uncertainty_raw[mat]
    shift = T_ref - temp[depth_index_200km]
    temperature_profiles[mat] = temp + shift
    temperature_uncertainties[mat] = (temp_low + shift, temp_high + shift)

# === Seismic velocity profiles ===
vp_depths = np.array([200, 400, 600, 750, 800, 1000, 1200, 1250, 1350, 1400])
vp_lower = np.array([7.5, 7.5, 7.5, 7.8, 7.8, 7.8, 7.5, 6, 5.5, 4.5])
vp_upper = np.array([7.8, 7.8, 8, 8.2, 8.5, 8.5, 8.5, 8.5, 8.5, 7.5])
vs_depths = np.array([200, 400, 600, 800, 1000, 1150, 1200, 1250, 1300, 1350, 1400])
vs_lower = np.array([4.0, 4.1, 4.2, 4.2, 4.3, 3, 3, 2.5, 0.5, 0.5, 0.5])
vs_upper = np.array([4.5, 4.6, 4.7, 4.7, 4.8, 4.9, 5, 4.8, 4.8, 3.5, 3.5])
dmq_near_side = [800, 1200]

# === Plotting ===
fig, axes = plt.subplots(1, 3, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1, 2]})

# Panel 1: Resistivity
axes[0].step(avg_resistivity, common_depth, where='post', color='black', label="Averaged Resistivity")
axes[0].fill_betweenx(common_depth, avg_lower95, avg_upper95, step='pre', color='gray', alpha=0.3, label="95% CI")
axes[0].axhspan(*dmq_near_side, color='red', alpha=0.1)
axes[0].set_xscale("log")
axes[0].invert_yaxis()
axes[0].set_xlabel("Resistivity (Ω·m)")
axes[0].set_ylabel("Depth (km)")
axes[0].set_ylim(1400, 200)
axes[0].set_title("Resistivity Profile with 95% CI")
axes[0].legend()
axes[0].grid(True, which="major", linestyle="--", alpha=0.7)

# Panel 2: Temperature
for mat, temp in temperature_profiles.items():
    temp_low, temp_high = temperature_uncertainties[mat]
    axes[1].plot(temp, common_depth, linestyle="--", label=mat)
    axes[1].fill_betweenx(common_depth, temp_low, temp_high, alpha=0.2, label=f"{mat} ±95% CI")
axes[1].axhspan(*dmq_near_side, color='red', alpha=0.1)
axes[1].invert_yaxis()
axes[1].set_xlabel("Temperature (K)")
axes[1].set_ylabel("Depth (km)")
axes[1].set_ylim(1400, 200)
axes[1].set_title("Estimated Temperature Profiles (Aligned at 200 km)")
axes[1].legend()
axes[1].grid(True, which="both", linestyle="--", alpha=0.7)

# Panel 3: Vp and Vs
axes[2].fill_betweenx(vp_depths, vp_lower, vp_upper, step='pre', color='blue', alpha=0.3, label="Vp Range")
axes[2].fill_betweenx(vs_depths, vs_lower, vs_upper, step='pre', color='green', alpha=0.3, label="Vs Range")
axes[2].step(vp_lower, vp_depths, 'k--', where='post', label="Vp Lower Bound")
axes[2].step(vp_upper, vp_depths, 'k--', where='post', label="Vp Upper Bound")
axes[2].step(vs_lower, vs_depths, 'k--', where='post', label="Vs Lower Bound")
axes[2].step(vs_upper, vs_depths, 'k--', where='post', label="Vs Upper Bound")
axes[2].axhspan(*dmq_near_side, color='red', alpha=0.1, label="Deep Moonquake Zone")
axes[2].invert_yaxis()
axes[2].set_xlabel("Velocity (km/s)")
axes[2].set_ylabel("Depth (km)")
axes[2].set_ylim(1400, 200)
axes[2].set_title("Seismic Velocities (Vp and Vs)")
axes[2].legend()
axes[2].grid(True, which="both", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig('Res_T_VpVs.jpeg', dpi=300, format='jpg')
plt.show()
