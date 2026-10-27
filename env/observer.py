"""
Beobachtung: ein flacher, auf [-1, 1] normierter Vektor.

Aufbau (Index-Reihenfolge):
    phase_one_hot      N_PHASES
    green_elapsed      1       (Anteil an MAX_GREEN)
    queue              N_PHASES (haltende Fahrzeuge je Phasengruppe)
    wait               N_PHASES (Wartezeit-Summe je Gruppe)
    count              N_PHASES (Fahrzeuge je Gruppe)
    eta                N_PHASES (kürzeste Ankunftszeit an der Haltelinie)
    hour_sin, hour_cos 2
    weather            N_WEATHER_FEATURES

Alle Werte kommen aus TraCI-Subscriptions (siehe environment.py), d. h. es
sind keine zusätzlichen Round-Trips zu SUMO nötig.
"""
import numpy as np
import traci.constants as tc

import config as C
from weather import N_WEATHER_FEATURES

OBS_SIZE = C.N_PHASES * 5 + 1 + 2 + N_WEATHER_FEATURES

LANE_VARS = [tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.LAST_STEP_VEHICLE_NUMBER,
             tc.VAR_WAITING_TIME, tc.LAST_STEP_VEHICLE_ID_LIST]
VEHICLE_VARS = [tc.VAR_SPEED, tc.VAR_LANEPOSITION]


class Observer:
    def __init__(self, tl, weather):
        self.tl = tl
        self.weather = weather
        self.group_capacity = {
            g: len(lanes) * C.LANE_CAPACITY for g, lanes in C.PHASE_LANES.items()
        }

    @staticmethod
    def _min_eta(vehicle_ids, veh_results) -> float:
        best = C.ETA_SCALE_S
        for vid in vehicle_ids:
            v = veh_results.get(vid)
            if v is None:
                continue
            dist = C.LANE_LENGTH_M - v[tc.VAR_LANEPOSITION]
            best = min(best, dist / max(v[tc.VAR_SPEED], 1.0))
        return best

    def observe(self, sim_time: float, lane_results: dict, veh_results: dict) -> np.ndarray:
        n = C.N_PHASES
        phase = np.zeros(n, dtype=np.float32)
        phase[self.tl.current_phase] = 1.0
        elapsed = np.array([min(1.0, self.tl.green_elapsed / C.MAX_GREEN)], dtype=np.float32)

        queue = np.zeros(n, dtype=np.float32)
        wait = np.zeros(n, dtype=np.float32)
        count = np.zeros(n, dtype=np.float32)
        eta = np.full(n, C.ETA_SCALE_S, dtype=np.float32)
        for g, lanes in C.PHASE_LANES.items():
            for lane in lanes:
                r = lane_results[lane]
                queue[g] += r[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]
                wait[g] += r[tc.VAR_WAITING_TIME]
                count[g] += r[tc.LAST_STEP_VEHICLE_NUMBER]
                eta[g] = min(eta[g], self._min_eta(r[tc.LAST_STEP_VEHICLE_ID_LIST], veh_results))
            cap = self.group_capacity[g]
            queue[g] = min(1.0, queue[g] / cap)
            count[g] = min(1.0, count[g] / cap)
            wait[g] = min(1.0, wait[g] / C.WAIT_SCALE_S)
            eta[g] = eta[g] / C.ETA_SCALE_S

        hour = (sim_time % 86400) / 3600.0
        rad = 2 * np.pi * hour / 24.0
        clock = np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)

        obs = np.concatenate([phase, elapsed, queue, wait, count, eta, clock,
                              self.weather.features()]).astype(np.float32)
        return np.clip(obs, -1.0, 1.0)
