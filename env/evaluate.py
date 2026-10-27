"""
Evaluation: trainierter Agent gegen Festzeit-Ampel auf identischen Szenarien
(gleicher Seed, gleiche Uhrzeit, gleiches Wetter).

    python env/evaluate.py                                  # best_model vs. Festzeit
    python env/evaluate.py --model logs/final_model.zip
    python env/evaluate.py --gui --policy agent             # zuschauen
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

import config as C
from environment import SumoEnvironment

# Grünzeiten der Festzeit-Ampel (aus dem statischen SUMO-Programm: 33/10/33/10 s,
# auf das 5-s-Entscheidungsraster gerundet)
FIXED_GREEN = {0: 35, 1: 10, 2: 35, 3: 10}


def fixed_time_policy(env: SumoEnvironment, obs, mask) -> int:
    cur = env.tl.current_phase
    if env.tl.green_elapsed >= FIXED_GREEN[cur] and env.tl.can_switch():
        return (cur + 1) % C.N_PHASES
    return cur


def random_policy(env, obs, mask) -> int:
    return int(np.random.choice(np.flatnonzero(mask)))


def agent_policy(model):
    def _policy(env, obs, mask) -> int:
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return _policy


def run_episode(env: SumoEnvironment, policy) -> dict:
    obs, info = env.reset()
    ep_return = 0.0
    while True:
        action = policy(env, obs, env.action_masks())
        obs, reward, _, truncated, info = env.step(action)
        ep_return += reward
        if truncated:
            info["return"] = ep_return
            return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=C.LOG_DIR / "best_model" / "best_model.zip")
    parser.add_argument("--policy", choices=["all", "agent", "fixed", "random"], default="all")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    policies = {}
    if args.policy in ("all", "agent"):
        if not args.model.is_file():
            raise SystemExit(f"Modell nicht gefunden: {args.model}")
        policies["agent"] = agent_policy(MaskablePPO.load(args.model, device="cpu"))
    if args.policy in ("all", "fixed"):
        policies["fixed"] = fixed_time_policy
    if args.policy in ("all", "random"):
        policies["random"] = random_policy

    results = {name: [] for name in policies}
    for sc in C.EVAL_SCENARIOS:
        for name, policy in policies.items():
            env = SumoEnvironment(env_id=200, gui=args.gui, scenarios=[sc])
            info = run_episode(env, policy)
            env.close()
            results[name].append(info)
            print(f"{name:7s} | {sc['start_hour']:02d}:00 {sc['duration_s'] // 3600}h "
                  f"{(sc['weather_file'] or 'klar'):24s} | wait/veh {info['wait_per_vehicle']:6.1f} s | "
                  f"queue {info['mean_queue']:5.1f} | arrived {info['arrived']:4d} | "
                  f"teleports {info['teleports']:2d} | return {info['return']:8.2f}")

    print("\nMittelwert aller Szenarien")
    print(f"{'policy':8s} {'wait/veh [s]':>13s} {'queue':>7s} {'arrived':>8s} {'teleports':>10s} {'return':>9s}")
    summary = {}
    for name, infos in results.items():
        m = lambda k: float(np.mean([i[k] for i in infos]))
        summary[name] = dict(wait_per_vehicle=m("wait_per_vehicle"), mean_queue=m("mean_queue"),
                             arrived=m("arrived"), teleports=m("teleports"), ret=m("return"))
        print(f"{name:8s} {m('wait_per_vehicle'):13.1f} {m('mean_queue'):7.1f} "
              f"{m('arrived'):8.0f} {m('teleports'):10.1f} {m('return'):9.2f}")

    # Mittelwerte merken – plot_training.py zeichnet daraus die Referenzlinien
    out = C.LOG_DIR / "eval_results.json"
    stored = json.loads(out.read_text()) if out.is_file() else {}
    stored.update(summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stored, indent=2))


if __name__ == "__main__":
    main()
