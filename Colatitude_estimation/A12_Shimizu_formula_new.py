# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:49:49 2025

@author: pingy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Function to convert spherical to Cartesian coordinates
def spherical_to_cartesian(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# Function to convert Cartesian to spherical coordinates
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return theta, phi, r

# Function to calculate pole positions and exchange based on BX mean
def calculate_pole_positions(data, theta_p, phi_p, moon_radius):
    south_x = moon_radius * np.sin(theta_p) * np.cos(phi_p)
    south_y = moon_radius * np.sin(theta_p) * np.sin(phi_p)
    south_z = moon_radius * np.cos(theta_p)

    north_x = -1 * south_x
    north_y = -1 * south_y
    north_z = -1 * south_z

    # Check BX mean to decide if poles need swapping
    bx_mean = data['p'].mean()  # Use processed p (BX) to decide
    if bx_mean < 0:  # Swap poles if BX mean is negative
        south_x, north_x = north_x, south_x
        south_y, north_y = north_y, south_y
        south_z, north_z = north_z, south_z

    # Convert poles to lat/lon
    south_theta, south_phi, _ = cartesian_to_spherical(south_x, south_y, south_z)
    north_theta, north_phi, _ = cartesian_to_spherical(north_x, north_y, north_z)
    south_lat = 90 - np.degrees(south_theta)
    south_lon = np.degrees(south_phi)
    north_lat = 90 - np.degrees(north_theta)
    north_lon = np.degrees(north_phi)

    return (north_x, north_y, north_z, north_lat, north_lon), (south_x, south_y, south_z, south_lat, south_lon)

# Function to process Apollo data
def process_apollo_data(file_path, station_lat, station_lon, title, offset):
    data = pd.read_csv(file_path, parse_dates=True, index_col=0, sep=',').iloc[5520:6960]
    fs = 1/60
    dt = 1 / fs
    endtime = len(data) * dt
    tt = np.arange(0, endtime, dt)

    bx = sp.signal.medfilt(data['BX'] + offset[0], 1)
    by = sp.signal.medfilt(data['BY'] + offset[1], 1)
    bz = sp.signal.medfilt(data['BZ'] + offset[2], 1)

    p = -bx
    g = -by
    h = -bz
    r = np.sqrt(p**2 + g**2 + h**2)

    sin_theta_p = np.sqrt((p / r)**2 + (g / r)**2)
    theta_p = np.arctan2(sin_theta_p, h / r)
    phi_p = np.arctan2(g / r, p / r)

    data['p'] = p
    data['g'] = g
    data['h'] = h
    data['magnitude'] = r
    data['phi_p'] = np.degrees(phi_p)
    data['theta_p'] = np.degrees(theta_p)
    
    station_lat_rad = np.radians(90 - station_lat)
    station_lon_rad = np.radians(station_lon)
    theta_p_rad = np.radians(data['theta_p'].mean())
    phi_p_rad = np.radians(data['phi_p'].mean())

    # Calculate pole positions
    north_pole, south_pole = calculate_pole_positions(data, theta_p_rad, phi_p_rad, moon_radius)

    print(f"Station: {title}")
    print(f"North Pole Latitude: {north_pole[3]:.2f}°, Longitude: {north_pole[4]:.2f}°")
    print(f"South Pole Latitude: {south_pole[3]:.2f}°, Longitude: {south_pole[4]:.2f}°")

    return data, station_lat_rad, station_lon_rad, theta_p_rad, phi_p_rad, north_pole, south_pole

# Constants
apollo12_file = "D:/Sat_MV/AP12_1969_1min.csv"
moon_radius = 1737.4

# Process Apollo 12 data
apollo12_data, apollo12_lat_rad, apollo12_lon_rad, apollo12_theta_p, apollo12_phi_p, apollo12_north, apollo12_south = process_apollo_data(
    apollo12_file, -3.0128, -23.4219, "Apollo 12", [25.8, -11.9, 25.8]) # According to Dyal et al. (1973)

# Plot Apollo 12 station and pole positions
fig, ax = plt.subplots(figsize=(8, 8))

# Apollo 15 station position
station12_y = moon_radius * np.sin(apollo12_lat_rad) * np.sin(apollo12_lon_rad)
station12_z = moon_radius * np.cos(apollo12_lat_rad)

# Plot Apollo 12 station and poles
ax.scatter(station12_y, station12_z, color='blue', label='Apollo 12 Station')
ax.scatter(apollo12_north[1], apollo12_north[2], color='red', label='North Pole (AP12)')
ax.scatter(apollo12_south[1], apollo12_south[2], color='black', label='South Pole (AP12)')

# Draw the Moon's surface as a perfect circle in YZ
theta = np.linspace(0, 2 * np.pi, 500)
circle_y = moon_radius * np.sin(theta)
circle_z = moon_radius * np.cos(theta)
ax.plot(circle_y, circle_z, color='gray', alpha=0.7)

# Customize the plot
ax.set_aspect('equal', adjustable='datalim')
ax.set_title("Apollo 12 Station and Pole Positions on the Moon (YZ Plane)")
ax.set_xlabel("Y (km)")
ax.set_ylabel("Z (km)")
plt.grid()

# Move legend outside the circle
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# Calculate station positions in magnetic coordinates
def station_in_magnetic_coords(lat_rad, lon_rad, north_pole):
    north_theta, north_phi, _ = cartesian_to_spherical(north_pole[0], north_pole[1], north_pole[2])

    # Rotation matrix to align magnetic north pole with z-axis
    pole_cartesian = spherical_to_cartesian(north_theta, north_phi, 1)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(pole_cartesian, z_axis)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm != 0:
        rot_axis /= rot_axis_norm
    angle = np.arccos(np.dot(pole_cartesian, z_axis))
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    station_cartesian = spherical_to_cartesian(lat_rad, lon_rad, 1)
    rotated_cartesian = np.dot(R, station_cartesian)
    theta_new, phi_new, _ = cartesian_to_spherical(*rotated_cartesian)
    return np.degrees(theta_new), np.degrees(phi_new)

apollo12_colatitude, apollo12_longitude = station_in_magnetic_coords(apollo12_lat_rad, apollo12_lon_rad, apollo12_north)


print(f"Apollo 12 Magnetic Colatitude: {apollo12_colatitude:.2f}°, Magnetic Longitude: {apollo12_longitude:.2f}°")


