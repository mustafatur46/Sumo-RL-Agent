"""
This script applies the unified SUMO weather model parameters in a running SUMO simulation.
"""
import traci
import time
import os
import sys
from weatherSeq import create_sumo_weather_sequence


def apply_weather_params(hour_idx, weather):
    """
    Sets TraCI weather-related vehicle parameters for all vehicles at the given hour index.
    """
    accel = weather['accel']
    decel = weather['decel']
    sigma = weather['sigma']
    min_gap = weather['minGap']
    tau = weather['tau']
    speed_factor = weather['speedFactor']
    max_speed = weather['maxSpeed_m_s']

    print(f"[Hour {hour_idx}] Applying weather: type={weather['precip_type']}, vis={weather['vis_m']}m")
    print(f"  accel={accel}, decel={decel}, sigma={sigma}, minGap={min_gap}, tau={tau}, speedFactor={speed_factor}, maxSpeed={max_speed} m/s")

    for veh_id in traci.vehicle.getIDList():
        try:
            traci.vehicle.setLaneChangeMode(veh_id, 1621)
            traci.vehicle.setAccel(veh_id, accel)
            traci.vehicle.setDecel(veh_id, decel)
            traci.vehicle.setParameter(veh_id, 'has.lcmodel.sigma', str(sigma))
            traci.vehicle.setMinGap(veh_id, min_gap)
            traci.vehicle.setTau(veh_id, tau)
            # Setze speedFactor zuerst
            traci.vehicle.setSpeedFactor(veh_id, speed_factor)
            # Dann setze maxSpeed als finale Geschwindigkeitsbegrenzung
            traci.vehicle.setMaxSpeed(veh_id, max_speed)
        except traci.TraCIException:
            continue


def main():
    # JSON weather file or directory
    json_file = "oneJunction/weatherFiles/freibergRain.json"
    # Seed for reproducibility
    try:
        seed = int(sys.argv[1])
    except (IndexError, ValueError):
        seed = 42

    # Build sequence
    sequence = create_sumo_weather_sequence(json_file)

    # Start SUMO
    sumo_cmd = [
        'sumo-gui',
        '--scale', '1.0',
        '-c', 'oneJunction/oneJunctionWeather.sumocfg',
        '--seed', str(seed),
        '--start'
    ]
    traci.start(sumo_cmd)

    step = 0
    max_steps = len(sequence) * 3600  # simulate through all hours

    while step < max_steps:
        traci.simulationStep()
        # Apply on departure
        for veh_id in traci.simulation.getDepartedIDList():
            w = sequence[step // 3600]
            apply_weather_params(step // 3600, w)

        # Hourly update
        if step % 3600 == 0:
            hour_idx = step // 3600
            if hour_idx < len(sequence):
                apply_weather_params(hour_idx, sequence[hour_idx])
            else:
                print(f"Warning: hour_idx {hour_idx} out of range")

        # Optional debug every 600 steps
        if step % 600 == 0:
            for vid in traci.vehicle.getIDList():
                cur_max = traci.vehicle.getMaxSpeed(vid)
                cur_speed = traci.vehicle.getSpeed(vid)
                print(f"Vehicle {vid}: maxSpeed={cur_max*3.6:.1f}km/h, speed={cur_speed*3.6:.1f}km/h")

        step += 1

    traci.close()


if __name__ == '__main__':
    main()
