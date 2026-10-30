"""
Gymnasium-Umgebung für die Ampel an Kreuzung J13.

Ein Env-Schritt = eine Entscheidung des Agenten:
    * gleiche Phase  -> DECISION_INTERVAL Sekunden Grün
    * andere Phase   -> Gelb + Rot-Gelb, dann DECISION_INTERVAL Sekunden Grün

Unzulässige Aktionen (Mindestgrün nicht erreicht / Höchstgrün überschritten)
werden per Action-Mask ausgeschlossen und – falls doch eine ankommt – auf
eine zulässige Aktion abgebildet. Die Episode endet ausschließlich durch
Zeitablauf (truncated=True), niemals durch "Fehlverhalten".

Alle Messwerte kommen über TraCI-Subscriptions: SUMO liefert sie mit jedem
simulationStep() automatisch mit, es sind keine Einzelabfragen nötig.
"""
import logging
import os
import random

import gymnasium as gym
import numpy as np
import traci
import traci.constants as tc
from gymnasium import spaces

import config as C
from observer import LANE_VARS, OBS_SIZE, VEHICLE_VARS, Observer
from reward import RewardCalculator
from sumo_runner import close_sumo, start_sumo
from trafficlight import TrafficLightController
from weather import WeatherManager

log = logging.getLogger(__name__)

SIM_VARS = [tc.VAR_TIME, tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_NUMBER,
            tc.VAR_TELEPORT_STARTING_VEHICLES_NUMBER]

# Attribute, die eine offene TraCI-Verbindung halten und nicht gepickelt
# werden dürfen (SubprocVecEnv.get_attr pickelt gebundene Methoden samt self).
_UNPICKLABLE = ("conn", "tl", "observer", "reward", "weather")


class SumoEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_id: int = 0, gui: bool = False, level: int = 0,
                 scenarios: list | None = None):
        """
        scenarios: feste Szenarien (dict mit seed, start_hour, duration_s,
                   weather_file, optional scale und extra_route_files), die bei
                   jedem reset() der Reihe nach durchlaufen werden.
                   None -> zufällig aus dem Curriculum-Level.
        """
        super().__init__()
        self.env_id = env_id
        self.gui = gui
        self.level = level
        self.scenarios = scenarios
        self._scenario_idx = 0

        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(C.N_PHASES)

        for name in _UNPICKLABLE:
            setattr(self, name, None)
        self.episode_count = 0
        self._last_obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._last_reward = 0.0
        self._label_base = f"env{env_id}_pid{os.getpid()}_{id(self):x}"

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in _UNPICKLABLE:
            state[name] = None
        return state

    # ------------------------------------------------------------ curriculum
    def set_level(self, level: int) -> None:
        self.level = int(np.clip(level, 0, len(C.CURRICULUM) - 1))

    def _next_scenario(self) -> dict:
        if self.scenarios:
            sc = dict(self.scenarios[self._scenario_idx % len(self.scenarios)])
            self._scenario_idx += 1
            return sc
        cfg = C.CURRICULUM[self.level]
        weather_file = None
        if random.random() < cfg["weather_prob"]:
            files = sorted(p.name for p in C.WEATHER_DIR.glob("*.json"))
            weather_file = random.choice(files) if files else None
        return dict(
            seed=int(self.np_random.integers(0, 2**31 - 1)),
            start_hour=random.choice(cfg["hours"]),
            duration_s=cfg["duration_s"],
            weather_file=weather_file,
        )

    # ------------------------------------------------------------------ gym
    def action_masks(self) -> np.ndarray:
        if self.tl is None:
            return np.ones(C.N_PHASES, dtype=bool)
        return np.array(self.tl.action_mask(), dtype=bool)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed + self.env_id)
        self.close()

        sc = self._next_scenario()
        self.begin = int(sc["start_hour"]) * 3600
        self.end = self.begin + int(sc["duration_s"])
        weather_path = C.WEATHER_DIR / sc["weather_file"] if sc.get("weather_file") else None
        self.scenario = sc
        self.episode_count += 1
        self._stats = dict(halting=0.0, arrived=0, teleports=0, sim_steps=0, forced=0)

        # Jede Episode bekommt ein neues TraCI-Label: falls ein close() nach einem
        # SUMO-Absturz fehlschlägt, bleibt das alte Label in traci registriert.
        label = f"{self._label_base}_{self.episode_count}"
        last_exc = None
        for attempt in range(3):
            try:
                conn = start_sumo(C.SUMO_CFG, self.begin, self.end, sc["seed"] + attempt, self.gui, label,
                                  scale=float(sc.get("scale", 1.0)),
                                  extra_route_files=sc.get("extra_route_files"))
                self.conn = conn
                conn.simulation.subscribe(SIM_VARS)
                for lane in C.INCOMING_LANES:
                    conn.lane.subscribe(lane, LANE_VARS)

                self.tl = TrafficLightController(conn)
                self.tl.install_program()
                self.weather = WeatherManager(weather_path)
                self.weather.select_hour(self.begin)
                self.observer = Observer(self.tl, self.weather)
                self.reward = RewardCalculator()

                self._simulate(1)              # ein Schritt, damit Subscriptions gefüllt sind
                self.tl.green_elapsed = 0
                self.reward.reset()
                self._last_obs = self._observe()
                return self._last_obs, self._info()
            except (traci.FatalTraCIError, traci.TraCIException, OSError) as exc:
                last_exc = exc
                log.error("env %d: SUMO-Start fehlgeschlagen (Versuch %d/3): %s", self.env_id, attempt + 1, exc)
                self.close()
                label = f"{label}r"
        raise RuntimeError(f"SUMO konnte nicht gestartet werden: {last_exc}")

    def step(self, action):
        action = int(action)
        mask = self.tl.action_mask()
        forced = False
        if not mask[action]:
            forced = True
            action = self.tl.current_phase if mask[self.tl.current_phase] \
                else (self.tl.current_phase + 1) % C.N_PHASES
        self._stats["forced"] += int(forced)

        try:
            if action != self.tl.current_phase:
                for sumo_phase, duration in self.tl.switch_sequence(action):
                    self.tl.set_sumo_phase(sumo_phase)
                    self._simulate(duration)
                self.tl.set_green(action)
            self._simulate(C.DECISION_INTERVAL)
            reward, r_info = self.reward.compute()
            self._last_obs = self._observe()
            sim_time = self._sim_time
        except (traci.FatalTraCIError, traci.TraCIException, OSError) as exc:
            # SUMO ist abgestürzt: Episode sauber beenden, nächstes reset() startet neu.
            # Reward = letzter normaler Reward, damit ein Absturz weder belohnt noch bestraft wird.
            log.error("env %d: SUMO-Fehler, Episode wird abgebrochen: %s", self.env_id, exc)
            self.close()
            info = self._info()
            info["sumo_error"] = str(exc)
            return self._last_obs, self._last_reward, False, True, info

        self._last_reward = reward
        self._stats["teleports"] += r_info["teleports"]
        truncated = sim_time >= self.end
        info = self._info()
        info.update(queue=r_info["queue"], forced_action=forced)
        if truncated:
            self.close()
        return self._last_obs, reward, False, truncated, info

    def close(self) -> None:
        if self.conn is not None:
            close_sumo(self.conn)
        for name in _UNPICKLABLE:
            setattr(self, name, None)

    # -------------------------------------------------------------- helpers
    def _simulate(self, seconds: int) -> None:
        conn = self.conn
        for _ in range(seconds):
            conn.simulationStep()
            self.tl.tick(1)
            sim = conn.simulation.getSubscriptionResults()
            self._sim_time = sim[tc.VAR_TIME]
            departed = sim[tc.VAR_DEPARTED_VEHICLES_IDS]
            for vid in departed:
                conn.vehicle.subscribe(vid, VEHICLE_VARS)
            if self.weather.select_hour(self._sim_time):
                self.weather.apply(conn, conn.vehicle.getIDList())
            else:
                self.weather.apply(conn, departed)

            lanes = conn.lane.getAllSubscriptionResults()
            self.reward.observe_step(lanes, sim[tc.VAR_TELEPORT_STARTING_VEHICLES_NUMBER])
            self._stats["halting"] += self.reward.last_halting
            self._stats["arrived"] += sim[tc.VAR_ARRIVED_VEHICLES_NUMBER]
            self._stats["sim_steps"] += 1

    def _observe(self) -> np.ndarray:
        return self.observer.observe(
            self._sim_time,
            self.conn.lane.getAllSubscriptionResults(),
            self.conn.vehicle.getAllSubscriptionResults(),
        )

    def _info(self) -> dict:
        s = self._stats
        steps = max(s["sim_steps"], 1)
        return {
            "level": self.level,
            "episode_idx": self.episode_count,        # "episode" ist für SB3-Monitor reserviert
            "env_id": self.env_id,
            "mean_queue": s["halting"] / steps,                        # haltende Fahrzeuge im Mittel
            "wait_per_vehicle": s["halting"] / max(s["arrived"], 1),  # s Wartezeit je Fahrzeug
            "arrived": s["arrived"],
            "teleports": s["teleports"],
            "forced": s["forced"],
            "sim_steps": s["sim_steps"],
        }
