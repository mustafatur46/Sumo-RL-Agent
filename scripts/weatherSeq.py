import sys
import os
from datetime import datetime
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.weather import WeatherManager

# Set global plot style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.alpha': 0.3,
    'grid.color': '#cccccc',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.axisbelow': True
})


def create_sumo_weather_sequence(json_file):
    """
    Builds and returns the SUMO weather parameter sequence using the unified WeatherManager.
    """
    # Initialize the manager with the JSON file or directory
    wm = WeatherManager(json_file)
    # Directly use the precomputed sequence
    return wm.weather_sequence


def plot_max_speed_per_hour(weather_sequence):
    """
    Plots the maximum speed for each hour of the day in km/h.
    """
    # Extract max speeds and convert to km/h (SUMO uses m/s)
    hours = range(24)  # 0-23 hours
    speeds = []
    
    for hour in hours:
        # Convert from m/s to km/h
        speed_kmh = weather_sequence[hour]['maxSpeed_m_s'] * 3.6
        speeds.append(speed_kmh)

    plt.figure(figsize=(12, 6))
    plt.plot(hours, speeds, 'b-', marker='o')
    plt.title('Maximum Speed per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Maximum Speed (km/h)')
    plt.grid(True)
    plt.xticks(range(0, 24))
    
    # Save the plot
    plt.savefig('outputs/max_speed_per_hour.png')
    plt.close()


def plot_weather_influence(weather_sequence):
    """
    Plots the weather influence factors for each hour of the day.
    """
    hours = range(24)
    speed_factors = []
    weather_types = []
    
    for hour in hours:
        weather = weather_sequence[hour]
        speed_factors.append(weather['speedFactor'])
        weather_types.append(weather['precip_type'])

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot speed factors
    plt.plot(hours, speed_factors, 'b-', marker='o', label='Speed Factor')
    
    plt.title('Weather Influence per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Speed Factor (1.0 = no influence)')
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.ylim(0, 1.1)  # Speed factor is typically between 0 and 1
    plt.legend()
    
    # Save the plot
    plt.savefig('outputs/weather_influence.png')
    plt.close()


def calculate_weather_severity(weather_entry):
    """
    Calculates a weather severity index from 0 (best conditions) to 1 (worst conditions)
    """
    # Normalize visibility (higher vis = better conditions)
    vis_factor = 1 - min(weather_entry.get('vis_km', 1.0) / 10.0, 1.0)
    
    # Combine precipitation (rain and snow)
    rain_mm = weather_entry.get('rain_mm', 0) + weather_entry.get('showers_mm', 0)
    snow_cm = weather_entry.get('snowfall_cm', 0)
    snow_depth = weather_entry.get('snow_depth_m', 0) * 100  # convert to cm
    
    # Normalize precipitation (0-1 scale)
    rain_factor = min(rain_mm / 10.0, 1.0)  # assume 10mm is maximum impact
    snow_factor = min((snow_cm + snow_depth) / 5.0, 1.0)  # assume 5cm is maximum impact
    
    # Get speed factor (already 0-1, where 1 is best conditions)
    speed_factor = 1 - weather_entry.get('speedFactor', 1.0)
    
    # Combine factors with weights
    severity = (
        0.3 * vis_factor +      # 30% visibility
        0.2 * rain_factor +     # 20% rain
        0.3 * snow_factor +     # 30% snow
        0.2 * speed_factor      # 20% speed adjustment
    )
    
    return severity


def plot_weather_severity(weather_sequence):
    """
    Plots the weather severity index for each hour of the day with improved styling.
    """
    hours = range(24)
    severity_indices = []
    
    for hour in hours:
        weather = weather_sequence[hour]
        severity = calculate_weather_severity(weather)
        severity_indices.append(severity)

    # Create the plot with improved styling
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set background style
    ax.set_facecolor('white')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3, color='gray')
    
    # Plot severity index with beautiful purple color
    line = ax.plot(hours, severity_indices, color='#9b59b6', marker='o', 
                  linewidth=2.5, markersize=8, label='Severity Index')
    
    # Fill area under the curve with slight transparency
    ax.fill_between(hours, severity_indices, alpha=0.1, color='#9b59b6')
    
    # Customize axis
    ax.set_xlabel('Hour', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weather Severity Index', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 23.5)
    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Customize title
    plt.title('Weather Severity Index by Hour', fontsize=14, fontweight='bold', pad=20)
    
    # Add text annotations for severe weather with improved styling
    for i, severity in enumerate(severity_indices):
        if severity > 0.5:  # Only annotate more severe conditions
            ax.annotate(f'{severity:.2f}', 
                       (i, severity),
                       textcoords="offset points",
                       xytext=(0,10),
                       ha='center',
                       va='bottom',
                       fontsize=9,
                       fontweight='bold',
                       color='#34495e')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('outputs/weather_severity.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Example usage:
    js = 'oneJunction/weatherFiles/AlaskaSnow01_29.json'
    seq = create_sumo_weather_sequence(js)
    plot_max_speed_per_hour(seq)
    plot_weather_influence(seq)
    plot_weather_severity(seq)
    print("Plots saved as 'max_speed_per_hour.png', 'weather_influence.png', and 'weather_severity.png' in outputs directory")