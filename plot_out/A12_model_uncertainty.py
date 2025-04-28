# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:25:00 2025

@author: pingy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === User setting: choose to plot resistivity or conductivity ===
plot_resistivity = True  # Set to False to plot conductivity instead

# === Load data ===
file_path = r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo12_1D_cond.csv"
df = pd.read_csv(file_path)

# === Extract data ===
depth = df['depth'].values
avg = df['avg'].values
lower68 = df['lower68'].values
upper68 = df['upper68'].values
lower95 = df['lower95'].values
upper95 = df['upper95'].values

# === Convert to resistivity if needed ===
if plot_resistivity:
    avg = 1 / avg
    lower68 = 1 / lower68
    upper68 = 1 / upper68
    lower95 = 1 / lower95
    upper95 = 1 / upper95
    xlabel = "Resistivity (Ω·m)"
else:
    xlabel = "Conductivity (S/m)"

# === Create stepped model for shading ===
step_depth = np.repeat(depth, 2)[1:]
step_depth = np.append(step_depth, step_depth[-1] + (step_depth[-1] - step_depth[-2]))

avg_step = np.repeat(avg, 2)
lower68_step = np.repeat(lower68, 2)
upper68_step = np.repeat(upper68, 2)
lower95_step = np.repeat(lower95, 2)
upper95_step = np.repeat(upper95, 2)

# === Plot ===
plt.figure(figsize=(6, 8))

plt.plot(avg_step, step_depth, label='Average', color='red', linewidth=2)
plt.fill_betweenx(step_depth, lower95_step, upper95_step, color='mistyrose', alpha=0.6, label='95% CI')
plt.fill_betweenx(step_depth, lower68_step, upper68_step, color='lightcoral', alpha=0.6, label='68% CI')

plt.xscale('log')
plt.ylim(200, 1400)            # Depth range
plt.gca().invert_yaxis()     # Depth increases downward

plt.xlabel(xlabel, fontsize=16)
plt.ylabel("Depth (km)", fontsize=16)
#plt.title("Apollo 12 1D Model with Confidence Intervals", fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, which='major', linestyle='--', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# === Save as high-resolution JPEG ===
plt.savefig('A12_Res_model.jpeg', format='jpg', dpi=300)
plt.show()
