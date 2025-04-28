# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:00:44 2025

@author: pingy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# === Settings ===
plot_resistivity = True  # Set to False to show conductivity instead
depth_limit = 500        # Plot up to this depth (in km)

# === File paths for each Apollo mission ===
file_paths = {
    "Apollo 12": r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo12_1D_cond.csv",
    "Apollo 15": r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo15_1D_cond.csv",
    "Apollo 16": r"D:\Users\pingy.LAPTOP-9PCQEDK5\Downloads\apollo16_1D_cond.csv"
}

# === Colors for each mission (customizable) ===
colors = {
    "Apollo 12": "red",
    "Apollo 15": "royalblue",
    "Apollo 16": "darkgreen"
}

# === Create plot ===
plt.figure(figsize=(6, 8))

for mission, path in file_paths.items():
    df = pd.read_csv(path)

    # Filter by depth limit
    mask = df["depth"] <= depth_limit
    depth = df["depth"].values[mask]
    avg = df["avg"].values[mask]
    lower68 = df["lower68"].values[mask]
    upper68 = df["upper68"].values[mask]
    lower95 = df["lower95"].values[mask]
    upper95 = df["upper95"].values[mask]

    # Extend flat to 600 km if needed (especially for Apollo 12)
    if depth[-1] < depth_limit:
        depth = np.append(depth, depth_limit)
        avg = np.append(avg, avg[-1])
        lower68 = np.append(lower68, lower68[-1])
        upper68 = np.append(upper68, upper68[-1])
        lower95 = np.append(lower95, lower95[-1])
        upper95 = np.append(upper95, upper95[-1])

    # Convert to resistivity if needed
    if plot_resistivity:
        avg = 1 / avg
        lower68 = 1 / lower68
        upper68 = 1 / upper68
        lower95 = 1 / lower95
        upper95 = 1 / upper95
        xlabel = "Resistivity (Ω·m)"
        out_filename = "Apollo_12_15_16_Resistivity_CI_0_600km.jpg"
    else:
        xlabel = "Conductivity (S/m)"
        out_filename = "Apollo_12_15_16_Conductivity_CI_0_600km.jpg"

    # === Step model for plotting ===
    step_depth = np.repeat(depth, 2)[1:]
    step_depth = np.append(step_depth, step_depth[-1] + (step_depth[-1] - step_depth[-2]))

    avg_step = np.repeat(avg, 2)
    lower68_step = np.repeat(lower68, 2)
    upper68_step = np.repeat(upper68, 2)
    lower95_step = np.repeat(lower95, 2)
    upper95_step = np.repeat(upper95, 2)

    # === Plot CI bands ===
    plt.fill_betweenx(step_depth, lower95_step, upper95_step, color=colors[mission], alpha=0.2, label=f"{mission} 95% CI")
    plt.fill_betweenx(step_depth, lower68_step, upper68_step, color=colors[mission], alpha=0.4, label=f"{mission} 68% CI")

    # === Plot average stepped curve ===
    plt.plot(avg_step, step_depth, color=colors[mission], linewidth=2, label=f"{mission} Avg")

# === Axis and style settings ===
plt.xscale("log")
plt.ylim(200, depth_limit)
plt.gca().invert_yaxis()  # Depth increases downward
plt.xlabel(xlabel, fontsize=16)
plt.ylabel("Depth (km)", fontsize=16)
#plt.title("Apollo 12, 15, 16 – 1D Models with Confidence Intervals", fontsize=18)
plt.legend(fontsize=12)

# === Reduce grid lines ===
plt.grid(True, which='major', linestyle='--', alpha=0.4)
plt.gca().xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(100))

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

# === Save to file ===
plt.savefig('All_Res_model.jpeg', dpi=300, format='jpg')
plt.show()

