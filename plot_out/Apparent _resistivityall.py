# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:03:52 2025

@author: pingy
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import pandas as pd

# Load the image
#image_path = "D:/Sat_MV/image.png"
#image = io.imread(image_path)
#gray_image = color.rgb2gray(image)

# Manually digitized Apollo observed data and LP/KS observed data
# Data extracted based on visual approximation of the logarithmic scale

apollo_data = {
    "T (hours)": np.array([0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 12, 15, 20]),
    "ρa (Ωm)": np.array([1600, 1400, 1200, 1000, 800, 700, 600, 400, 300, 200, 150, 120, 100, 90, 70]),
    "Error": np.array([100, 90, 80, 70, 60, 50, 45, 35, 30, 25, 20, 15, 12, 10, 8])
}

lpks_data = {
    "T (hours)": np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22]),
    "ρa (Ωm)": np.array([200, 180, 160, 140, 130, 120, 110, 105, 100, 95, 90, 85, 80, 75, 70]),
    "Error": np.array([20, 18, 17, 15, 14, 12, 11, 10, 9, 8, 7, 6, 6, 5, 4])
}

# Load Apollo 12, 15, and 16 data
file_paths = {
    "Apollo 12": "D:/Sat_MV/A12_ZH_SlidingWindows_Results.csv",
    "Apollo 15": "D:/Sat_MV/A15_ZH0527_SlidingWindows_Results.csv",
    "Apollo 16": "D:/Sat_MV/A16_ZH0527_SlidingWindows_Results.csv"
}

apollo_mission_data = {}

for mission, path in file_paths.items():
    df = pd.read_csv(path)
    apollo_mission_data[mission] = {
        "T (hours)": df["Period (Hour)"].values,
        "ρa (Ωm)": df["ZH_Apparent_Resistivity"].values,
        "Error": df["ZH_Rho_Error"].values
    }

# Plot the digitized data and Apollo mission data
plt.figure(figsize=(8,6))
plt.errorbar(apollo_data["T (hours)"], apollo_data["ρa (Ωm)"], yerr=apollo_data["Error"], fmt='s', color='blue', label="Apollo12 OG", markersize=8)
plt.errorbar(lpks_data["T (hours)"], lpks_data["ρa (Ωm)"], yerr=lpks_data["Error"], fmt='^', color='black', label="LP/KS ", markersize=8)

colors = {"Apollo 12": "orange", "Apollo 15": "green", "Apollo 16": "red"}

for mission, data in apollo_mission_data.items():
    plt.errorbar(data["T (hours)"], data["ρa (Ωm)"], yerr=data["Error"], fmt='o', color=colors[mission], label=mission, markersize=8)

plt.xscale("log")
plt.yscale("linear")
plt.xlim(0.2, 24)
plt.ylim(0, 12000)
plt.xlabel("T (hours)",fontsize=14)
plt.ylabel("ρa (Ωm)",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
#plt.title("Digitized Data and Apollo Mission Resistivities")

plt.show()

# Display the extracted data in a table
apollo_df = pd.DataFrame(apollo_data)
lpks_df = pd.DataFrame(lpks_data)

print("Apollo Observed Data:")
print(apollo_df.head())
print("\nLP/KS Observed Data:")
print(lpks_df.head())

for mission, data in apollo_mission_data.items():
    df = pd.DataFrame(data)
    print(f"\n{mission} Resistivity Data:")
    print(df.head())
