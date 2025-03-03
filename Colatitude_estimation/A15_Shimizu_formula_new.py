# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:21:47 2024

@author: pingy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Function to annotate averages in the blank area of the plot
def annotate_averages(ax, tt, y_min, y_max, average_phi_p, average_theta_p, average_alpha):
    """
    Add annotations for average azimuth, inclination, and angle to the North Pole in the blank area of the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The Axes object of the plot.
    tt (array): Time array.
    y_min (float): Minimum value of the y-axis.
    y_max (float): Maximum value of the y-axis.
    average_phi_p (float): Average azimuth.
    average_theta_p (float): Average inclination.
    average_alpha (float): Average angle to the North Pole.
    """
    x_mid = tt[len(tt) // 2]  # Midpoint of the time axis
    text_y_pos = y_min + (y_max - y_min) * 0.15  # Position text slightly above the minimum y-value
    ax.text(x_mid, text_y_pos, f'Avg Azimuth: {average_phi_p:.2f}°', color='red', fontsize=10)
    ax.text(x_mid, text_y_pos - (y_max - y_min) * 0.05, f'Avg Inclination: {average_theta_p:.2f}°', color='red', fontsize=10)
    ax.text(x_mid, text_y_pos - (y_max - y_min) * 0.1, f'Avg Angle to Pole: {average_alpha:.2f}°', color='red', fontsize=10)

# Read Apollo 15 data
#kny = pd.read_csv("D:/Sat_MV/APollo15_0427.csv", parse_dates=True, index_col=0, sep=',').iloc[74500:129600]
kny = pd.read_csv("D:/Sat_MV/APollo15_0527.csv", parse_dates=True, index_col=0, sep=',').iloc[242875:286202]
# Sampling frequency and time calculations
fs = 3  # Sampling frequency (default: 3 Hz)
dt = 1 / fs  # Sampling interval
endtime = len(kny) * dt  # Total time in seconds
tt = np.arange(0, endtime, dt)

# Read three-axis magnetic field vectors X, Y, Z (adding offsets)
knyX = sp.signal.medfilt(kny['BX'] -3.3, 1)
knyY = sp.signal.medfilt(kny['BY'] - 0.9, 1)
knyZ = sp.signal.medfilt(kny['BZ'] + 0.2, 1)  # According to Dyal et al. (1973) remanent field @A15 site
#remanent field 3.3 +/- 1.5 nT for BX, 0.9 +/- 2 nT for BY, -0.2 +/- 1.5 nT for BZ 

# Plot the rotated magnetic fields (X, Y only)
plt.figure(figsize=(12, 6))
plt.plot(tt/3600, knyZ, label='Magnetic Field Z')
plt.plot(tt/3600, knyY, label='Magnetic Field Y')
plt.plot(tt/3600, knyX, label='Magnetic Field X')
plt.title("Apollo15 Magnetic Field Variation (X, Y Components in ME coodinates)")
plt.xlabel("Time (hours)")
plt.ylabel("Magnetic Field (nT)")
plt.ylim(-40,40)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Transform the vectors as (x, y, z) = (-p, -g, -h)
p = -knyX
g = -knyY
h = -knyZ

# Calculate the magnitude of the vector (p, g, h)
r = np.sqrt(p**2 + g**2 + h**2)

# Compute spherical angles
sin_theta_p = np.sqrt((p / r)**2 + (g / r)**2)  # Compute sin(theta_p)
theta_p = np.arctan2(sin_theta_p, h / r)  # Inclination (in radians)
phi_p = np.arctan2(g / r, p / r)  # Azimuth (in radians)

# Compute the angle to the North Pole (alpha)
alpha = np.arccos(h / r)  # Angle in radians
kny['north_pole_angle'] = np.degrees(alpha)  # Convert to degrees

# Add calculated results to the DataFrame
kny['p'] = p
kny['g'] = g
kny['h'] = h
kny['magnitude'] = r
kny['phi_p'] = np.degrees(phi_p)  # Azimuth in degrees
kny['theta_p'] = np.degrees(theta_p)  # Inclination in degrees

# Original station location in degrees
station_lat = 26.13239  # North latitude is positive
station_lon = 3.63330  # East longitude is positive
moon_radius = 1737.4  # Moon's radius in kilometers

# North Pole latitude and longitude calculation
# Convert station coordinates and angles to radians
station_lat_rad = np.radians(90-station_lat)
station_lon_rad = np.radians(station_lon)
theta_p_rad = np.radians(kny['theta_p'].mean())  # Average colatitude in radians
phi_p_rad = np.radians(kny['phi_p'].mean())  # Average azimuth in radians
cos_delta =np.sin(theta_p_rad)*np.sin(station_lat_rad)*np.cos(station_lon_rad - phi_p_rad)+np.cos(theta_p_rad)*np.cos(station_lat_rad)
theta_delta = np.arccos(cos_delta)

# Convert back to degrees in ME coordinate
north_pole_lat_deg = 90-np.degrees(theta_p_rad)
north_pole_lon_deg = np.degrees(phi_p_rad)
colatitude_delta = np.degrees(theta_delta)

# Print the North Pole latitude and longitude in the original coordinate system
print(f"South Pole Latitude (ME coordinates): {north_pole_lat_deg:.2f}°")
print(f"South Pole Longitude (ME coordinates): {north_pole_lon_deg:.2f}°")
print(f"colatitude: {180-colatitude_delta:.2f}°")

def spherical_to_cartesian(theta, phi, r):
    """Convert spherical to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian to spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    phi = np.arctan2(y, x)
    return theta, phi, r

def rotation_matrix_from_pole(theta_pole, phi_pole):
    """Compute the rotation matrix to align the new pole with the z-axis."""
    # New pole in Cartesian coordinates
    pole = spherical_to_cartesian(theta_pole, phi_pole, 1.0)  # Unit vector

    # Rotation axis: cross product of pole and z-axis
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(pole, z_axis)
    rot_axis_norm = np.linalg.norm(rot_axis)

    # Handle the case where the new pole is already the z-axis
    if rot_axis_norm == 0:
        return np.eye(3)  # Identity matrix (no rotation needed)

    rot_axis /= rot_axis_norm  # Normalize rotation axis

    # Rotation angle
    cos_theta = np.dot(pole, z_axis)  # Dot product of pole and z-axis
    angle = np.arccos(cos_theta)

    # Compute rotation matrix using Rodrigues' formula
    K = np.array([
        [0, -rot_axis[2], rot_axis[1]],
        [rot_axis[2], 0, -rot_axis[0]],
        [-rot_axis[1], rot_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R

def transform_coordinates(theta_station, phi_station, r, theta_pole, phi_pole):
    """Transform the coordinates of a point under the new pole."""
    # Convert station to Cartesian coordinates
    station_cartesian = spherical_to_cartesian(theta_station, phi_station, r)

    # Compute rotation matrix to align new pole with z-axis
    R = rotation_matrix_from_pole(theta_pole, phi_pole)

    # Apply rotation
    rotated_cartesian = np.dot(R, station_cartesian)

    # Convert back to spherical coordinates
    theta_new, phi_new, r_new = cartesian_to_spherical(*rotated_cartesian)
    return theta_new, phi_new, r_new

# Use rotation matrix to find new station's colatitude and longitude to the magnetic North Pole
if __name__ == "__main__":
    # Original station coordinates in spherical: (theta_station, phi_station, r)
    theta_station = np.radians(90-station_lat)  
    phi_station = np.radians(station_lon)   
    r = moon_radius

    # New north pole coordinates: (theta_pole, phi_pole, r)
    #theta_pole = np.radians(90-north_pole_lat_deg)   # mag-field pointed out from the Moon @ moon's Southth pole  
    #phi_pole = np.radians(north_pole_lon_deg)
    theta_pole = np.pi - theta_p_rad    # mag-field pointed into the Moon @moon's North pole
    phi_pole = (phi_p_rad + np.pi) % (2 * np.pi)

    # Transform the station coordinates
    theta_new, phi_new, r_new = transform_coordinates(theta_station, phi_station, moon_radius, theta_pole, phi_pole)

    print("New spherical coordinates regarding the magnetic North pole:")
    print(f"Theta: {np.degrees(theta_new):.2f} degrees")
    print(f"Phi: {np.degrees(phi_new):.2f} degrees")
    print(f"R: {r_new:.2f} km")

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(tt[:len(kny['phi_p'])], kny['phi_p'], label='Azimuth (phi_p)', color='blue')
ax.plot(tt[:len(kny['theta_p'])], kny['theta_p'], label='Inclination (theta_p)', color='orange')
ax.plot(tt[:len(kny['north_pole_angle'])], kny['north_pole_angle'], label='Angle to North Pole (alpha)', color='green')

# Calculate averages
average_phi_p = kny['phi_p'].mean()
average_theta_p = kny['theta_p'].mean()
average_alpha = kny['north_pole_angle'].mean()

# Get dynamic y-axis range
y_min = min(kny['phi_p'].min(), kny['theta_p'].min(), kny['north_pole_angle'].min())
y_max = max(kny['phi_p'].max(), kny['theta_p'].max(), kny['north_pole_angle'].max())
ax.set_ylim(y_min - 5, y_max + 5)  # Add some padding to the y-axis range

# Annotate averages in red
annotate_averages(ax, tt, y_min, y_max, average_phi_p, average_theta_p, average_alpha)

# Add labels, legend, and grid
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Angle (degrees)')
ax.legend()
ax.set_title('Azimuth, Inclination, and Angle to South Pole')
ax.grid()
#plt.ylim(-200,200)
plt.tight_layout()
plt.show()

# Plot the station, north pole, and antipodal point on a spherical coordinate net
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Convert spherical to Cartesian coordinates for plotting
r1 = moon_radius
station_x = r1 * np.sin(station_lat_rad) * np.cos(station_lon_rad)
station_y = r1 * np.sin(station_lat_rad) * np.sin(station_lon_rad)
station_z = r1 * np.cos(station_lat_rad)

pole_x = r1 * np.sin(theta_p_rad) * np.cos(phi_p_rad)
pole_y = r1 * np.sin(theta_p_rad) * np.sin(phi_p_rad)
pole_z = r1 * np.cos(theta_p_rad)

# Plot the antipole as well
thetap_opposite = np.pi - theta_p_rad
phip_opposite = (phi_p_rad + np.pi) % (2 * np.pi)
pole_x_opposite = r1 * np.sin(thetap_opposite) * np.cos(phip_opposite)
pole_y_opposite = r1 * np.sin(thetap_opposite) * np.sin(phip_opposite)
pole_z_opposite = r1 * np.cos(thetap_opposite)

# Plot the station, north pole, and antipodal point
ax.scatter(station_x, station_y, station_z, color='blue', label='Station')
ax.scatter(pole_x, pole_y, pole_z, color='red', label='South Pole')
ax.scatter(pole_x_opposite, pole_y_opposite, pole_z_opposite, color='green', label='North Pole')

# Draw the Moon's surface as a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r1 * np.outer(np.cos(u), np.sin(v))
y = r1 * np.outer(np.sin(u), np.sin(v))
z = r1 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='gray', alpha=0.5, edgecolor='none')

# Customize the plot
ax.set_box_aspect([1, 1, 1])  # Ensures x, y, z axes are equally scaled
ax.set_title("Apollo15 Station, North Pole, and South Pole on the Moon")
ax.set_xlabel("X (km)")
ax.set_ylabel("Y (km)")
ax.set_zlabel("Z (km)")
ax.view_init(elev=0, azim=0)
ax.legend()

plt.show()