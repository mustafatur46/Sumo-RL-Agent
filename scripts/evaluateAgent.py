from stable_baselines3 import PPO
from datetime import datetime
import os
import random
from weatherSeq import create_sumo_weather_sequence
from weatherSumo import apply_weather_params
from EnvForEval import WeatherTrafficEnv
import traci


def evaluate_trained_agent(model_path, sumo_config, weather_dir, traffic_light_id, timesteps=86400):
    config = {
        "sumo_config": sumo_config,
        "weather_dir": weather_dir,
        "traffic_light_id": traffic_light_id,
        "traci_port": random.randint(9900, 9999)
    }

    env = WeatherTrafficEnv(config)
    obs = env.reset(episode=999)

    model = PPO.load(model_path)

    done = False
    total_reward = 0.0
    step_count = 0

    while not done and step_count < timesteps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1

        if step_count % 600 == 0:
            print(f"[Eval] Step {step_count} / {timesteps}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

    print("\n--- Evaluation Completed ---")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final info: {info}")

    env.close()


if __name__ == "__main__":
    model_path = "checkpoints\ppo_weather_400000_steps.zip"
    sumo_config = "oneJunction/oneJunctionRL.sumocfg"
    weather_dir = "oneJunction/weatherFiles/"
    traffic_light_id = "J13"

    evaluate_trained_agent(
        model_path=model_path,
        sumo_config=sumo_config,
        weather_dir=weather_dir,
        traffic_light_id=traffic_light_id,
        timesteps=86400
    )
