# trainAgentparallel.py

from datetime import datetime
import math
from typing import Tuple
import gym
import numpy as np
import traci
from gym import spaces
from weatherSeq import create_sumo_weather_sequence
from weatherSumo import apply_weather_params
import os
import time
import random
import glob
import re
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure,Logger
from stable_baselines3.common.callbacks import BaseCallback,CallbackList
from stable_baselines3.common.utils import get_linear_fn
from enum import Enum

class PhaseState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    REDYELLOW = "redyellow"

class WeatherTrafficEnv(gym.Env):
    def __init__(self, config):
        super(WeatherTrafficEnv, self).__init__()

        self.current_phase = 0
        self.phase_timer = 0
        self.in_redyellow = False
        self.sumo_config = config.get("sumo_config", "oneJunction/oneJunctionWeather.sumocfg")
        self.sumocfg_path = self.sumo_config  
        self.weather_dir = config.get("weather_dir", "oneJunction/weatherFiles/")
        self.traffic_light_id = config.get("traffic_light_id", "J13")
        self.traci_port = config.get("traci_port", random.randint(8813, 9999))
        self.phase_state = PhaseState.GREEN  # or "yellow", "redyellow"
        self.last_phase_switch_step = 0
        self.green_phase_map = {
            0: 1,
            1: 4,
            2: 7,
            3: 10
        }

        self.yellow_phase_map = {
            0: 2,
            1: 5,
            2: 8,
            3: 11
        }

        self.redyellow_phase_map = {
            0: 0,
            1: 3,
            2: 6,
            3: 9
        }

        self.phase_lane_map = {
            0: ["-E8_0", "-E8_1", "-E6_0", "-E6_1"],
            1: ["-E8_2", "-E6_2"],
            2: ["-E7_0", "-E7_1", "-E9_0", "-E9_1"],
            3: ["-E7_2", "-E9_2"]
        }

        self.observation_lanes = [
            "-E8_0", "-E8_1", "-E8_2",
            "-E6_0", "-E6_1", "-E6_2",
            "-E7_0", "-E7_1", "-E7_2",
            "-E9_0", "-E9_1", "-E9_2"
        ]

        self.phase_duration_map = {0: 10, 1: 10, 2: 10, 3: 10}
        self.phase_switched_this_step = False
        self.weather_keys = [
            "accel", "decel", "sigma", "minGap", "tau", "maxSpeed_m_s",
            "temperature", "vis_km", "rain_mm", "showers_mm", "snowfall_cm", "snow_depth_m"
        ]
        self.weather_type_keys = ["rain", "snow", "clear"]
        self.recent_rewards = []
        self.sequence = []
        self.current_weather = {}
        self.last_weather_hour = -1
        self.num_rl_phases = 4
        self.max_phase_duration = 30
        self.phase_durations = list(range(5, 121))

        self.action_space = spaces.MultiDiscrete([self.num_rl_phases, len(self.phase_durations)])
        num_lanes = len(self.observation_lanes)
        num_weather = 13
        num_weather_type = 3
        time_features = 2
        extra_features = 5

        self.obs_dim = (
            num_lanes +
            num_weather +
            num_weather_type +
            time_features +
            extra_features
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.step_counter = 0
        self.max_steps = 3600 * 24
        self.collision_count = 0
        self.in_yellow = False
        self.next_redyellow_phase = None
        self.next_rl_phase = None
        self.YELLOW_HOLD_STEPS = 3
        self.REDYELLOW_HOLD_STEPS = 1
        self.next_green_phase = None

    # ---- Curriculum Helper ----
    def get_smoothed_reward(self, window: int = 50) -> float:
        recent = self.recent_rewards[-window:] if hasattr(self, 'recent_rewards') else []
        return sum(recent) / len(recent) if recent else 0.0
    def _ensure_active_sim(self):
        try:
            if traci.getLabel() != str(self.traci_port):
                self._ensure_active_sim()

        except traci.TraCIException:
            print(f"[ERROR] Failed to switch to port {self.traci_port}. TraCI may not be loaded yet.")

    @staticmethod
    def curriculum_level(mean_reward: float, episode: int) -> int:
        if episode < 15:
            return 0
        elif mean_reward < 2.0:
            return 0
        elif mean_reward < 2.7:
            return 1
        elif mean_reward < 3.5:
            return 2
        elif mean_reward < 4.2 and episode >= 100:
            return 3
        elif mean_reward < 4.8 and episode >= 200:
            return 4
        elif episode >= 300:
            return 5
        else:
            return 4


    def init_traffic_light(self):
        if self.traffic_light_id not in traci.trafficlight.getIDList():
            raise RuntimeError(f"[ERROR] Traffic light ID '{self.traffic_light_id}' not found in SUMO network!")

        print(f"[OK] Agent controls traffic light: '{self.traffic_light_id}'")

        # Force override SUMO's default program logic
        traci.trafficlight.setProgram(self.traffic_light_id, "custom")

        self.current_phase = 0
        self.in_yellow = False
        self.in_redyellow = False
        self.phase_timer = 0
        self.step_counter = 0
        self.next_redyellow_phase = None
        self.next_rl_phase = self.current_phase

        traci.trafficlight.setPhase(self.traffic_light_id, self.green_phase_map[self.current_phase])        
        traci.simulationStep()  # ✅ warm-up step for GUI sync
        
    def reset(self,episode=0):
        if traci.isLoaded():
            traci.close()
            
        if not hasattr(self, "episode_counter"):
            self.episode_counter = 0
        self.episode = self.episode_counter
        self.episode_counter += 1
        # Get current mean reward + curriculum level
        mean_reward = self.get_smoothed_reward(window=50)
        level = 0 if self.episode < 10 else WeatherTrafficEnv.curriculum_level(mean_reward, self.episode)

        # Lock-in minimum reached level to avoid regressions
        # self.min_level = max(getattr(self, "min_level", 0), level)
        # level = max(level, self.min_level)

        if level == 5 and (mean_reward < 1.0 or self.episode < 200):
            level = 4
        # Curriculum-driven start_hour & duration
        if level == 0:
            start_hour = random.choice([2, 3, 4])
            duration_hours = 1
        elif level == 1:
            start_hour = random.choice([5, 6])
            duration_hours = 1
        elif level == 2:
            start_hour = random.choice([7, 8, 9])
            duration_hours = 2
        elif level == 3:
            start_hour = random.choice([10, 11, 12, 13, 14])
            duration_hours = 3
        elif level == 4:
            start_hour = random.choice([15, 16, 17, 18, 19])
            duration_hours = 4
        elif level == 5:
            start_hour = random.choice([7, 8, 9, 15, 16, 17])
            duration_hours = random.choice([5, 6])

        # occasional challenge spike
        if random.random() < 0.1:
            start_hour = random.randint(0, 23)
            duration_hours = random.choice([2, 3, 4, 6])


        self.begin_time = start_hour * 3600
        self.end_time = self.begin_time + duration_hours * 3600
        self.max_steps = self.end_time - self.begin_time

        weather_files = [f for f in os.listdir(self.weather_dir) if f.endswith(".json")]
        chosen_weather = os.path.join(self.weather_dir, random.choice(weather_files))
        random_seed = random.randint(0, 99999)

        self.last_chosen_weather = chosen_weather
        self.last_random_seed = random_seed
        episode_id = getattr(self, "episode", 0)
        begin_time = start_hour * 3600
        end_time = begin_time + duration_hours * 3600
        
        try:
            os.makedirs("logs/episode_seeds", exist_ok=True)
            with open(f"logs/episode_seeds/episode_{episode_id}.txt", "w") as f:
                f.write(f"seed={random_seed}\n")
                f.write(f"weather={chosen_weather}\n")
                f.write(f"start_hour={start_hour}, end_hour={start_hour + duration_hours}\n")
                f.write(f"sim_seconds={begin_time} to {end_time}\n")
        except Exception as e:
            print(f"[WARNING] Failed to write seed log: {e}")


        print(f"[Reset] Weather file: {chosen_weather}, seed: {random_seed}, port: {self.traci_port}")

        self.sequence = create_sumo_weather_sequence(chosen_weather)
        if len(self.sequence) < 24:
            raise ValueError("Weather file does not contain a full 24-hour sequence.")
        self.sequence = self.sequence[start_hour:] + self.sequence[:start_hour]
        self.current_weather = self.sequence[0] if self.sequence else {}

        traci.start([
            "sumo-gui", "-c", self.sumocfg_path,
            "--start", "--no-step-log", "true",
            "--begin", str(0),
            "--end", str(86400),
            "--seed", str(random_seed)
        ], label=str(self.traci_port))
        print(f"[DEBUG] Starting SUMO on port {self.traci_port}")
        print("[DEBUG] SUMO started successfully")
        self._ensure_active_sim()

        self.init_traffic_light()

        apply_weather_params(start_hour, self.current_weather)

        
        self.phase_timer = 0
        self.in_yellow = False
        self.in_redyellow = False

        obs = self._get_obs()
        return obs
    

    def _debug_step_log(self, rl_phase, duration):
        sumo_phase = traci.trafficlight.getPhase(self.traffic_light_id)
        # print("\n[DEBUG STEP]")
        # print(f"Step #{self.step_counter}")
        # print(f"Agent State Phase       : {self.phase_state}")
        # print(f"SUMO TL Phase           : {sumo_phase}")
        # print(f"Current RL Phase        : {self.current_phase}")
        # print(f"Agent Action            : phase={rl_phase}, duration={duration}")
        # print(f"Phase Timer             : {self.phase_timer}")
        # print(f"Last Phase Switch Step  : {self.last_phase_switch_step}")
        # print(f"Cooldown Active         : {self.step_counter - self.last_phase_switch_step < 5}")

    def get_valid_actions(self):
        """
        Returns the valid actions (green phases) the agent can choose.
        The agent should only choose green phases: 0, 1, 2, or 3.
        """
        return [0, 1, 2, 3]
    
    def step(self, action):
        try:
            if not traci.isLoaded():
                return self._get_obs(), 0.0, True, {"reason": "traci_closed"}

            traci.switch(str(self.traci_port))
            self.phase_switched_this_step = False

            rl_phase, duration_idx = action
            duration = self.phase_durations[min(duration_idx, len(self.phase_durations) - 1)]

            if self.phase_state == PhaseState.GREEN:
                if rl_phase not in self.get_valid_actions():
                    print(f"[Penalty] Invalid action: {rl_phase}")
                    self.invalid_phase_count += 1
                    done = self.step_counter >= self.max_steps  # ✅ Fix: check episode end
                    return self._get_obs(), -2.0, done, {"reason": "invalid_action"}
                self._execute_phase_logic(rl_phase, duration)

            self._advance_phase_fsm()

            obs, reward, done, info = self._return_step("safe_transition")
            return obs, reward, done, info

        except Exception as e:
            import traceback
            print(f"[ERROR] in step(): {e}")
            self.close()
            traceback.print_exc()
            return self._get_obs(), 0.0, True, {"reason": "exception"}

    def _build_additional_metrics(self):
        current_phase = self.current_phase / len(self.green_phase_map)
        timer_scaled = self.phase_timer / 60.0

        total_wait = sum(traci.lane.getWaitingTime(l) for l in self.observation_lanes)
        vehicle_count = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.observation_lanes)
        flow = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.phase_lane_map.get(self.current_phase, []))

        mean_wait = total_wait / max(vehicle_count, 1)
        flow_norm = flow / (vehicle_count + 1)

        norm_wait = min(mean_wait / 60.0, 1.0)
        norm_vehicle = min(vehicle_count / 40.0, 1.0)
        norm_flow = min(flow_norm, 1.0)

        return [
            current_phase,
            timer_scaled,
            norm_wait,
            norm_vehicle,
            norm_flow
        ]

    def _execute_phase_logic(self, rl_phase, duration):

        if duration not in self.phase_durations:
            print(f"[INVALID] Duration {duration} not in allowed set.")
            self.invalid_phase_count += 1
            return

        if self.step_counter - self.last_phase_switch_step < 5:
            print(f"[COOLDOWN] Cannot switch phases before 5 steps.")
            return

        self.phase_state = PhaseState.YELLOW
        self.phase_timer = 0
        self.next_rl_phase = rl_phase
        self.next_duration = duration
        self.next_green_phase = self.green_phase_map[rl_phase]
        self.next_redyellow_phase = self.redyellow_phase_map[rl_phase]
        self.last_phase_switch_step = self.step_counter
        print(f"[PHASE] Switching: GREEN {self.current_phase} → YELLOW")
        traci.trafficlight.setPhase(self.traffic_light_id, self.yellow_phase_map[self.current_phase])


    def _advance_phase_fsm(self):
        if self.phase_state == PhaseState.YELLOW:
            self.phase_timer += 1
            if self.phase_timer >= self.YELLOW_HOLD_STEPS:
                self.phase_timer = 0
                self.phase_state = PhaseState.REDYELLOW
                print(f"[PHASE] YELLOW done → REDYELLOW")
                traci.trafficlight.setPhase(self.traffic_light_id, self.next_redyellow_phase)

        elif self.phase_state == PhaseState.REDYELLOW:
            self.phase_timer += 1
            if self.phase_timer >= self.REDYELLOW_HOLD_STEPS:
                self.phase_timer = self.next_duration
                self.phase_state = PhaseState.GREEN
                self.last_phase_switch_step = self.step_counter
                self.current_phase = self.next_rl_phase
                self.phase_switched_this_step = True
                print(f"[PHASE] REDYELLOW done → GREEN {self.current_phase}")
                traci.trafficlight.setPhase(self.traffic_light_id, self.next_green_phase)


        elif self.phase_state == PhaseState.GREEN:
            if self.phase_timer > 0:
                self.phase_timer -= 1

        self.in_yellow = self.phase_state == PhaseState.YELLOW
        self.in_redyellow = self.phase_state == PhaseState.REDYELLOW

        traci.simulationStep()
        self._post_step_updates()

    def _post_step_updates(self):
        self.step_counter += 1
        self.collision_count += traci.simulation.getCollidingVehiclesNumber()

        if self.step_counter % 3600 == 0:
            current_hour = self.step_counter // 3600
            if current_hour < len(self.sequence):
                self.current_weather = self.sequence[current_hour]
                apply_weather_params(current_hour, self.current_weather)

        for veh_id in traci.simulation.getDepartedIDList():
            try:
                w = self.current_weather
                traci.vehicle.setLaneChangeMode(veh_id, 1621)
                traci.vehicle.setAccel(veh_id, w["accel"])
                traci.vehicle.setDecel(veh_id, w["decel"])
                traci.vehicle.setMaxSpeed(veh_id, w["maxSpeed_m_s"])
                traci.vehicle.setSpeedFactor(veh_id, w["final_speedFactor"])
                traci.vehicle.setParameter(veh_id, "has.lcmodel.sigma", str(w["sigma"]))
                traci.vehicle.setMinGap(veh_id, w["minGap"])
                traci.vehicle.setTau(veh_id, w["tau"])
            except traci.TraCIException:
                continue


    def _return_step(self, reason: str) -> Tuple[np.ndarray, float, bool, dict]:
        obs = self._get_obs()
        reward = self.compute_reward()
        
        done = self.step_counter >= self.max_steps or getattr(self, "teleport_termination", False)
        info = self.build_info_dict(reward)
        info["teleports"] = traci.simulation.getEndingTeleportNumber()

        info["reason"] = reason if not getattr(self, "teleport_termination", False) else "teleport"
        return obs, reward, done, info



    def build_info_dict(self, reward: float) -> dict:
        self._ensure_active_sim()

        rewards = getattr(self, "recent_rewards", [])
        mean_reward = sum(rewards) / len(rewards) if rewards else reward
        smoothed_reward = self.get_smoothed_reward(window=50)
        level = WeatherTrafficEnv.curriculum_level(mean_reward, self.episode)

        return {
            "left_turn_pressure": sum(traci.lane.getLastStepHaltingNumber(l) for l in ["-E8_2", "-E6_2", "-E7_2", "-E9_2"]) / 4,
            "straight_pressure": sum(traci.lane.getLastStepHaltingNumber(l) for l in ["-E8_0", "-E8_1", "-E6_0", "-E6_1", "-E7_0", "-E7_1", "-E9_0", "-E9_1"]) / 8,
            "weather_factor": self.current_weather.get("final_speedFactor", 1.0),
            "mean_reward": mean_reward,
            "collisions": traci.simulation.getCollidingVehiclesNumber(),
            "flow": sum(traci.lane.getLastStepVehicleNumber(l) for l in self.observation_lanes),
            "vehicle_count": traci.vehicle.getIDCount(),
            "phase_timer": self.phase_timer,
            "current_rl": self.current_phase,
            "reward": reward,
            "step_counter": self.step_counter,
            "curriculum_level": level,
            "mean_reward_smooth": smoothed_reward,
            "in_yellow": self.in_yellow,
            "in_redyellow": self.in_redyellow
        }




    def compute_reward(self):
        self._ensure_active_sim()

        flow = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.phase_lane_map.get(self.current_phase, []))
        total_waiting = sum(traci.lane.getWaitingTime(l) for l in self.observation_lanes)
        num_vehicles = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.observation_lanes)
        avg_lane_pressure = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in self.observation_lanes])
        left_pressure = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in ["-E8_2", "-E6_2", "-E7_2", "-E9_2"]])
        straight_pressure = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in ["-E8_0", "-E8_1", "-E6_0", "-E6_1", "-E7_0", "-E7_1", "-E9_0", "-E9_1"]])
        weather_factor = self.current_weather.get("final_speedFactor", 1.0)

        mean_wait_time = total_waiting / max(num_vehicles, 1)
        prev_wait_time = getattr(self, "prev_wait_time", mean_wait_time)
        wait_reduction = prev_wait_time - mean_wait_time
        self.prev_wait_time = mean_wait_time

        prev_flow = getattr(self, "prev_flow", flow)
        flow_delta = flow - prev_flow
        self.prev_flow = flow

        mean_reward = self.get_smoothed_reward(window=50)
        level = WeatherTrafficEnv.curriculum_level(mean_reward, self.episode)

        reward = 0.0

        # --- Baseline ---
        reward += (1.5 + 0.5 * level) * (flow / (num_vehicles + 1))
        reward += (2.0 + 0.4 * level) * wait_reduction

        # --- Curriculum: Level 0 Logic ---
        if level == 0:
            if num_vehicles <= 2:
                # Penalize phase switching, but with a milder penalty to allow exploration.
                if self.phase_switched_this_step:
                    reward -= 0.5

                # Compute a relative wait metric.
                # Adding a small constant to avoid division by zero.
                relative_improvement = (prev_wait_time - mean_wait_time) / (prev_wait_time + 1e-5)

                # Amplify the reward for relative improvement.
                # For example, if the wait time reduces by 10%, add a bonus proportional to that improvement.
                reward += 2.0 * relative_improvement

                # Fixed bonus for exceptionally low wait times can remain, but might be reduced.
                if mean_wait_time == 0:
                    reward += 1.0  # Reduced from 2.0
                elif mean_wait_time < 3:
                    reward += 0.5  # Reduced from 1.0

                # Bonus for not switching phases.
                if not self.phase_switched_this_step:
                    reward += 0.3

        # --- Curriculum Level 1–5 logic stays as-is ---
        if level == 1:
            if self.phase_switched_this_step:
                reward += 0.5 if avg_lane_pressure > 0.2 else -0.5

        elif level == 2:
            reward += 1.0 * max(0, 1.0 - avg_lane_pressure)
            if self.phase_switched_this_step and wait_reduction < 0:
                reward -= 0.5

        elif level == 3:
            reward += 1.0 - np.std([left_pressure, straight_pressure])

        elif level == 4:
            reward += 3.0 * wait_reduction
            if self.phase_switched_this_step and flow_delta < -2:
                reward -= 1.0

        elif level == 5:
            traffic_scale = min(num_vehicles / 40.0, 1.0)
            reward += (flow / (num_vehicles + 1)) * weather_factor
            reward += wait_reduction * traffic_scale

        # --- Universal penalties and bonuses ---
        if num_vehicles > 3 and flow < 1:
            reward -= 1.0

        teleports = traci.simulation.getEndingTeleportNumber()
        if teleports > 0:
            reward -= 2.0 * teleports
            print(f"[TELEPORT] {teleports} vehicles teleported.")
            self.teleport_termination = True

        if not self.phase_switched_this_step and avg_lane_pressure < 2.0:
            reward += 0.3

        if weather_factor < 0.7 and not self.phase_switched_this_step:
            reward += 0.5

        reward += 0.1 if flow > 0 else 0.0

        reward = np.clip(reward, -3.0, 6.0)
        self.recent_rewards.append(reward)
        return reward



    def set_logger(self, logger: Logger):
        self.logger = logger

# Refactored _get_obs into modular parts for clarity

    def _get_obs(self):
        if not traci.isLoaded():
            return np.zeros(self.obs_dim, dtype=np.float32)

        try:
            return np.array(
                self._build_lane_pressure_obs() +
                self._build_weather_obs() +
                self._build_time_features() +
                self._build_additional_metrics(),
                dtype=np.float32
            )
        except:
            print("[ERROR] Failed to get observation from SUMO.")
            return np.zeros(self.obs_dim, dtype=np.float32)

    def _build_lane_pressure_obs(self):
        return [
            min(traci.lane.getLastStepHaltingNumber(l) / 20.0, 1.0)
            for l in self.observation_lanes
        ]

    def _build_weather_obs(self):
        w = self.current_weather
        return [
            w.get("temperature", 15.0) / 40.0,
            w.get("vis_km", 10.0) / 20.0,
            w.get("rain_mm", 0.0) / 10.0,
            w.get("showers_mm", 0.0) / 10.0,
            w.get("snowfall_cm", 0.0) / 10.0,
            w.get("snow_depth_m", 0.0),
            w.get("accel", 2.6) / 5.0,
            w.get("decel", 4.5) / 8.0,
            w.get("sigma", 0.5) / 2.0,
            w.get("minGap", 2.0) / 5.0,
            w.get("tau", 1.6) / 3.0,
            w.get("maxSpeed_m_s", 13.9) / 20.0,
            w.get("final_speedFactor", 1.0)
        ] + w.get("weather_type_onehot", [0.0, 0.0, 1.0])

    def _build_time_features(self):
        sim_hour = ((self.begin_time + self.step_counter) / 3600.0) % 24
        hour_sin = math.sin(2 * math.pi * sim_hour / 24.0)
        hour_cos = math.cos(2 * math.pi * sim_hour / 24.0)
        return [hour_sin, hour_cos]

    def _build_additional_metrics(self):
        current_phase = self.current_phase / len(self.green_phase_map)
        timer_scaled = self.phase_timer / 60.0

        total_wait = sum(traci.lane.getWaitingTime(l) for l in self.observation_lanes)
        vehicle_count = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.observation_lanes)
        flow = sum(traci.lane.getLastStepVehicleNumber(l) for l in self.phase_lane_map.get(self.current_phase, []))

        mean_wait = total_wait / max(vehicle_count, 1)
        flow_norm = flow / (vehicle_count + 1)

        norm_wait = min(mean_wait / 60.0, 1.0)
        norm_vehicle = min(vehicle_count / 40.0, 1.0)
        norm_flow = min(flow_norm, 1.0)

        return [
            current_phase,
            timer_scaled,
            norm_wait,
            norm_vehicle,
            norm_flow
        ]

    def init_traffic_light(self):
        if self.traffic_light_id not in traci.trafficlight.getIDList():
            raise RuntimeError(f"[ERROR] Traffic light ID '{self.traffic_light_id}' not found in SUMO network!")
        else:
            print(f"[OK] Agent controls traffic light: '{self.traffic_light_id}'")

        self.current_phase = 0
        self.in_yellow = False
        self.in_redyellow = False
        self.phase_timer = 0
        self.step_counter = 0
        self.next_redyellow_phase = None
        self.next_rl_phase = self.current_phase

        traci.trafficlight.setPhase(self.traffic_light_id, self.green_phase_map[self.current_phase])


  
    def close(self):
        try:
            traci.close()
        except traci.TraCIException:
            print("[WARN] traci.close() skipped – already closed.")




class TensorboardMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [0] * len(infos))
        if not infos or not isinstance(infos[0], dict):
            print("[WARNING] TensorBoard callback received empty or invalid infos")
            return True
        def avg(key, default=0.0):
            try:
                return sum(info.get(key, default) for info in infos) / len(infos)
            except Exception as e:
                print(f"[ERROR] avg() failed for key={key}: {e}")
                return default
        if self.n_calls % 100 == 0:
            self.logger.record("custom/step_reward", sum(rewards) / len(rewards))
            self.logger.record("custom/mean_reward", avg("mean_reward"))
            self.logger.record("custom/mean_reward_smooth", avg("mean_reward_smooth"))
            self.logger.record("custom/curriculum_level", avg("curriculum_level"))
            self.logger.record("custom/flow", avg("flow"))
            self.logger.record("custom/vehicle_count", avg("vehicle_count"))
            self.logger.record("custom/left_turn_pressure", avg("left_turn_pressure"))
            self.logger.record("custom/straight_pressure", avg("straight_pressure"))
            self.logger.record("custom/weather_factor", avg("weather_factor"))
            self.logger.record("custom/collisions", avg("collisions"))
            if "teleports" in infos[0]:
                self.logger.record("custom/teleports", avg("teleports"))
            self.logger.record("custom/flow_reward_ratio", sum(rewards) / max(avg("flow", 1), 1))
            fsm_counts = {"green": 0, "yellow": 0, "redyellow": 0}
            for info in infos:
                if info.get("in_yellow"): fsm_counts["yellow"] += 1
                elif info.get("in_redyellow"): fsm_counts["redyellow"] += 1
                else: fsm_counts["green"] += 1
            self.logger.dump(self.n_calls)
        return True
# ---- Training Setup ----
if __name__ == "__main__":

    # --- Config ---
    global_config = {
        "sumo_config": "oneJunction/oneJunctionRL.sumocfg",
        "weather_dir": "oneJunction/weatherFiles/",
        "traffic_light_id": "J13"
    }

    dt = datetime.now().strftime("%d-%m-%Y_%H-%M")
    log_dir = f"logs/ppo_weather/run_{dt}"
    os.makedirs("./checkpoints", exist_ok=True)

    # --- Vectorized Env ---
    def make_env(rank):
        def _init():
            config = {
                "sumo_config": global_config["sumo_config"],
                "weather_dir": global_config["weather_dir"],
                "traffic_light_id": global_config["traffic_light_id"],
                "traci_port": 8800 + rank
            }
            return WeatherTrafficEnv(config)
        return _init

    num_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # --- Checkpoints ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix="ppo_weather",
        verbose=2
    )

    # --- Policy / Optimizer ---
    policy_kwargs = dict(
        net_arch=[256, 256],
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=dict(
            betas=(0.9, 0.999),
            eps=1e-8
        )
    )

    # --- Learning Rate Schedule ---
    lr_schedule = get_linear_fn(1e-4, 5e-5, 4_000_000)

    # --- Resume from Checkpoint ---
    def extract_step_num(path):
        match = re.search(r"_(\d+)_steps", path)
        return int(match.group(1)) if match else 0

    all_ckpts = glob.glob("checkpoints/ppo_weather_*_steps.zip")

    if all_ckpts:
        latest_ckpt = max(all_ckpts, key=extract_step_num)
        print(f"[Resume] Loading latest checkpoint: {latest_ckpt}")
        model = PPO.load(latest_ckpt, env=env)
    else:
        print(f"[Start] Training from scratch... [{dt}]")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir
        )
        print(model.policy.optimizer)

    # --- Callbacks ---
    callbacks = CallbackList([
        checkpoint_callback,
        TensorboardMetricsCallback()
    ])

    # --- Train ---
    _ = env.reset()
    model.learn(total_timesteps=4_000_000, callback=callbacks)

    # --- Save Final Model ---
    model.save("agents/ppo_weather_agent")

    # --- Optional: Flush final logs ---
    model.logger.dump(step=model.num_timesteps)
