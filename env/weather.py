"""
Wettermodell: Open-Meteo-Stundenwerte -> SUMO-Fahrzeugparameter.

    Geschwindigkeit  <- Niederschlagsart/-menge und Sichtweite (FHWA-kalibriert)
    Bremsvermögen    <- Reibwert (mu sinkt mit sqrt(Intensität))
    Beschleunigung   <- moderat mit der Geschwindigkeit skaliert
    Reaktionszeit,   <- steigen bei Sicht unter 300 m
    Mindestabstand

Die Parameter werden auf JEDES Fahrzeug angewendet – auch auf die, die erst
nach dem Stundenwechsel in die Simulation einfahren.
"""
import json
import logging
import math

import numpy as np
import traci

import config as C

log = logging.getLogger(__name__)

PRECIP_TYPES = ["clear", "rain", "snow", "mixed", "fog"]
N_WEATHER_FEATURES = len(PRECIP_TYPES) + 3     # one-hot + Intensität, Sicht, Reibung


class WeatherManager:
    # SUMO-Standardwerte für den vType "car"
    DEFAULT_ACCEL = 2.6
    DEFAULT_DECEL = 4.5
    DEFAULT_MIN_GAP = 2.0
    DEFAULT_TAU = 1.6
    REFERENCE_SPEED = 50.0            # km/h
    V_REF = REFERENCE_SPEED / 3.6     # m/s

    # ── Reibwert (für Bremsvermögen) ────────────────────────────────────────
    MU_DRY = 0.8
    K_RAIN = 0.1                      # mu -= K * sqrt(Intensität [mm/h])
    K_SNOW = 0.2
    DECEL_SCALE_MIN = 0.5             # nie unter 50 % des trockenen Bremsvermögens planen

    # ── Geschwindigkeit (kalibriert auf FHWA-Messwerte, freie Fahrt) ────────
    # Regen:  -5 % leicht  ... -12 % stark (>= 5 mm/h)
    # Schnee: -15 % leicht ... -30 % stark (>= 1 cm/h = 10 mm Wasseräquivalent)
    # Nebel:  keine Wirkung ab 300 m Sicht, bis -25 % bei 50 m
    RAIN_LIGHT, RAIN_HEAVY, RAIN_HEAVY_MMH = 0.05, 0.12, 5.0
    SNOW_LIGHT, SNOW_HEAVY, SNOW_HEAVY_MMH = 0.15, 0.30, 10.0
    FOG_VIS_FREE, FOG_VIS_MIN, FOG_MAX_REDUCTION = 300.0, 50.0, 0.25
    VISIBILITY_MAX = 1000.0           # m – darunter gilt die Stunde als "fog"
    VISIBILITY_REACTION = 300.0       # m – darunter steigen Reaktionszeit und Abstand

    CLEAR = {
        "precip_type": "clear", "vis_m": 10_000.0, "intensity": 0.0,
        "speedFactor": 1.0, "maxSpeed_m_s": V_REF,
        "accel": DEFAULT_ACCEL, "decel": DEFAULT_DECEL,
        "minGap": DEFAULT_MIN_GAP, "tau": DEFAULT_TAU,
    }

    def __init__(self, json_file=None):
        """json_file=None -> immer klares Wetter."""
        self.enabled = json_file is not None
        self.sequence = self._build_sequence(json_file) if self.enabled else []
        self.current = self.CLEAR
        self._hour_idx = None

    # ------------------------------------------------------------- Modell
    def _friction_ratio(self, precip_type: str, intensity: float) -> float:
        """Reibwert relativ zu trocken (1.0 = trocken). Skaliert das Bremsvermögen."""
        mu = self.MU_DRY
        if precip_type == "rain":
            mu -= self.K_RAIN * math.sqrt(intensity)
        elif precip_type in ("snow", "mixed"):
            mu -= self.K_SNOW * math.sqrt(intensity)
        return max(self.DECEL_SCALE_MIN, mu / self.MU_DRY)

    def _precip_speed_factor(self, precip_type: str, intensity: float) -> float:
        if precip_type == "rain":
            frac = min(1.0, intensity / self.RAIN_HEAVY_MMH)
            return 1.0 - (self.RAIN_LIGHT + (self.RAIN_HEAVY - self.RAIN_LIGHT) * frac)
        if precip_type in ("snow", "mixed"):
            frac = min(1.0, intensity / self.SNOW_HEAVY_MMH)
            return 1.0 - (self.SNOW_LIGHT + (self.SNOW_HEAVY - self.SNOW_LIGHT) * frac)
        return 1.0

    def _fog_speed_factor(self, vis_m: float) -> float:
        if vis_m >= self.FOG_VIS_FREE:
            return 1.0
        frac = (self.FOG_VIS_FREE - max(vis_m, self.FOG_VIS_MIN)) / (self.FOG_VIS_FREE - self.FOG_VIS_MIN)
        return 1.0 - self.FOG_MAX_REDUCTION * frac

    def _visibility_ratio(self, vis_m: float) -> float:
        return min(1.0, vis_m / self.VISIBILITY_REACTION)

    def _reaction_time(self, vis_m: float) -> float:
        return self.DEFAULT_TAU * (2.0 - self._visibility_ratio(vis_m))

    def _min_gap(self, vis_m: float) -> float:
        return self.DEFAULT_MIN_GAP * (2.0 - self._visibility_ratio(vis_m))

    def _build_sequence(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            hourly = json.load(f).get("hourly", {})
        n = len(hourly.get("time", []))
        vis = hourly.get("visibility", [self.VISIBILITY_MAX * 10] * n)
        # "precipitation" enthält bei Open-Meteo bereits das Schnee-Wasseräquivalent,
        # deshalb Regen und Schnee getrennt aus "rain"/"showers" und "snowfall" lesen.
        rain = [r + s for r, s in zip(hourly.get("rain", [0.0] * n), hourly.get("showers", [0.0] * n))]
        snow = hourly.get("snowfall", [0.0] * n)

        seq = []
        for i in range(n):
            vis_m = float(vis[i])
            has_rain, has_snow = rain[i] > 0.0, snow[i] > 0.0
            if has_rain and has_snow:
                ptype, intensity = "mixed", rain[i] + snow[i] * 10.0
            elif has_snow:
                ptype, intensity = "snow", snow[i] * 10.0
            elif has_rain:
                ptype, intensity = "rain", rain[i]
            elif vis_m < self.VISIBILITY_MAX:
                ptype, intensity = "fog", 0.0
            else:
                ptype, intensity = "clear", 0.0

            speed_factor = min(self._precip_speed_factor(ptype, intensity),
                               self._fog_speed_factor(vis_m))
            speed_factor = max(C.SPEED_FACTOR_MIN, speed_factor)
            # Fahrer beschleunigen bei Nässe/Schnee vorsichtiger, aber nicht wie ein Lkw
            accel_scale = 0.5 + 0.5 * speed_factor
            decel_scale = self._friction_ratio(ptype, intensity)

            seq.append({
                "precip_type": ptype,
                "vis_m": vis_m,
                "intensity": float(intensity),
                "speedFactor": round(speed_factor, 3),
                "maxSpeed_m_s": round(speed_factor * self.V_REF, 3),
                "accel": round(self.DEFAULT_ACCEL * accel_scale, 3),
                "decel": round(self.DEFAULT_DECEL * decel_scale, 3),
                "minGap": round(self._min_gap(vis_m), 3),
                "tau": round(self._reaction_time(vis_m), 3),
            })
        return seq

    # ------------------------------------------------------------- Laufzeit
    def select_hour(self, sim_time: float) -> bool:
        """Wählt den Stundeneintrag; True, wenn sich das Wetter geändert hat."""
        idx = int((sim_time % 86400) // 3600)
        if idx == self._hour_idx:
            return False
        self._hour_idx = idx
        self.current = self.sequence[idx] if idx < len(self.sequence) else self.CLEAR
        return True

    def apply(self, conn, vehicle_ids) -> None:
        if not self.enabled:
            return
        w = self.current
        for vid in vehicle_ids:
            try:
                conn.vehicle.setAccel(vid, w["accel"])
                conn.vehicle.setDecel(vid, w["decel"])
                conn.vehicle.setMinGap(vid, w["minGap"])
                conn.vehicle.setTau(vid, w["tau"])
                conn.vehicle.setSpeedFactor(vid, w["speedFactor"])
                conn.vehicle.setMaxSpeed(vid, w["maxSpeed_m_s"])
            except traci.TraCIException as exc:   # Fahrzeug evtl. gerade angekommen
                log.debug("weather apply %s: %s", vid, exc)

    def features(self) -> np.ndarray:
        """Normierter Merkmalsvektor (Länge N_WEATHER_FEATURES, Werte in [0, 1])."""
        w = self.current
        one_hot = np.zeros(len(PRECIP_TYPES), dtype=np.float32)
        one_hot[PRECIP_TYPES.index(w["precip_type"])] = 1.0
        extra = np.array([
            min(1.0, w["intensity"] / C.PRECIP_SCALE),
            min(1.0, w["vis_m"] / C.VISIBILITY_SCALE_M),
            w["speedFactor"],
        ], dtype=np.float32)
        return np.concatenate([one_hot, extra])
