# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:01:16 2025

@author: pingy
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

# =========================
# Original Code (Unchanged)
# =========================

# Read Apollo 12 data
kny = pd.read_csv("D:/Sat_MV/AP12_1969_1min.csv", sep=',').iloc[5520:6960] 
Decimate_factor = 1
kny = kny.iloc[::Decimate_factor]  # Downsample

# Rename problematic columns 'min' and 'sec' to avoid conflicts
kny.rename(columns={'min': 'minute', 'sec': 'second'}, inplace=True)

# Combine 'year', 'month', 'day', 'hour', 'minute', and 'second' into a single timestamp
kny['timestamp'] = pd.to_datetime(kny[['year', 'month', 'day', 'hour', 'minute', 'second']])

# Read three-axis magnetic field vectors X, Y, Z (adding offsets)
knyX = signal.medfilt(kny['BX'] + 25.8, 1)
knyY = signal.medfilt(kny['BY'] - 11.9, 1)
knyZ = signal.medfilt(kny['BZ'] + 25.8, 1)

# Assuming Hx, Hy, Hz are already present in kny, otherwise, define them:
Hx = kny['Hx'] if 'Hx' in kny.columns else np.nan
Hy = kny['Hy'] if 'Hy' in kny.columns else np.nan
Hz = kny['Hz'] if 'Hz' in kny.columns else np.nan

# Create a DataFrame to store the data
output_df = pd.DataFrame({
    'timestamp': kny['timestamp'],
    'Hx': knyX,
    'Hy': knyY,
    'Hz': knyZ
    })

# Save to a CSV file
output_filename = "D:/Sat_MV/AP12_Processed.csv"
output_df.to_csv(output_filename, index=False)

print(f"Data saved successfully to {output_filename}")
