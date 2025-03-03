# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:12:38 2025

@author: pingy
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_resistivity_model(file_path):
    """
    Reads a resistivity model CSV file and returns depth (in km) and resistivity values.
    """
    df = pd.read_csv(file_path)
    
    # Print column names to debug KeyError
    print(f"Columns in {file_path}: {df.columns.tolist()}")  
    
    # Ensure correct column names (strip spaces just in case)
    df.columns = df.columns.str.strip()
    
    # Verify the expected columns exist
    if "Depth (m)" not in df.columns or "Resistivity (Ω·m)" not in df.columns:
        raise KeyError(f"Expected columns 'Depth (m)' and 'Resistivity (Ω·m)' not found in {file_path}")

    depth_km = df["Depth (m)"] / 1000  # Convert depth to km
    resistivity = df["Resistivity (Ω·m)"]
    return depth_km, resistivity

def plot_resistivity_models(file_paths, labels):
    """
    Reads multiple resistivity models from CSV files and plots them.
    
    :param file_paths: List of CSV file paths.
    :param labels: List of labels corresponding to each file.
    """
    plt.figure(figsize=(8, 10))
    
    for file_path, label in zip(file_paths, labels):
        try:
            depth_km, resistivity = read_resistivity_model(file_path)
            plt.step(resistivity, depth_km, where='post', linestyle='-', label=label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.xscale("log")  # Log scale for resistivity
    plt.gca().invert_yaxis()  # Invert y-axis so depth increases downward
    
    plt.xlabel("Resistivity (Ω·m)")
    plt.ylabel("Depth (km)")
    plt.ylim(1400,0)
    plt.title("Comparison of Resistivity Models")
    plt.legend()
    #plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.show()

# List of file paths (use raw strings or forward slashes to avoid escape sequence issues)
file_paths = [
    r"D:\Sat_MV\inverted_model_A12_OG.csv",
    r"D:\Sat_MV\inverted_model_A12.csv",
    r"D:\Sat_MV\inverted_model_A15.csv",
    r"D:\Sat_MV\inverted_model_A16.csv"
]

# Labels for each model
labels = ["Apollo12_OG", "Apollo12", "Apollo15", "Apollo16"]

# Plot the resistivity models
plot_resistivity_models(file_paths, labels)

