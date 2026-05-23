import os
import subprocess
import random
import sys

try:
    seed = int(sys.argv[1])
except (IndexError, ValueError):
    print("No valid seed provided. Using default seed: 42")
    seed = 42

# Path to your SUMO configuration file
sumo_config_file = "oneJunction/oneJunction.sumocfg"

# Path to SUMO binary (adjust if SUMO is not in your PATH)
sumo_binary = "sumo-gui"  # Use "sumo" for command-line mode

# Run SUMO with the configuration file
try:
    subprocess.run([sumo_binary, "--scale", "1.0", "-c", sumo_config_file,"--seed", str(seed)], check=True)
except FileNotFoundError:
    print("SUMO binary not found. Ensure SUMO is installed and added to your PATH.")
except subprocess.CalledProcessError as e:
    print(f"Error while running SUMO: {e}")
