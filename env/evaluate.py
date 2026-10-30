"""
Evaluation: trainierter Agent gegen Festzeit-Ampel auf identischen Szenarien
(gleicher Seed, gleiche Uhrzeit, gleiches Wetter).

    python env/evaluate.py                                  # best_model vs. Festzeit
    python env/evaluate.py --model logs/final_model.zip
    python env/evaluate.py --gui --policy agent             # zuschauen
    python env/evaluate.py --stress                         # Verkehrsfaktor 1.5/2.0, einseitige Last
    python env/evaluate.py --ablation                       # Uhrzeit/Wetter in der Beobachtung ausblenden
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
from observer import OBS_SIZE

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


# ── Ablation: einzelne Teile der Beobachtung vor dem Netz "ausschalten" ─────
# Indexlayout siehe observer.py: phase(4) elapsed(1) queue(4) wait(4) count(4) eta(4) clock(2) weather(8)
TIME_IDX = slice(C.N_PHASES * 5 + 1, C.N_PHASES * 5 + 3)
WEATHER_IDX = slice(C.N_PHASES * 5 + 3, OBS_SIZE)
CLEAR_WEATHER = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)   # Features bei klarem Wetter


def _shift_clock(clock: np.ndarray, hours: float) -> np.ndarray:
    rad = np.arctan2(clock[0], clock[1]) + 2 * np.pi * hours / 24.0
    return np.array([np.sin(rad), np.cos(rad)], dtype=np.float32)


ABLATIONS = {
    "voll":                  lambda o: o,
    "ohne Uhrzeit":          lambda o: _patch(o, TIME_IDX, np.zeros(2, dtype=np.float32)),
    "Uhrzeit +12 h":         lambda o: _patch(o, TIME_IDX, _shift_clock(o[TIME_IDX], 12)),
    "Wetter = klar":         lambda o: _patch(o, WEATHER_IDX, CLEAR_WEATHER),
    "ohne Uhrzeit + Wetter": lambda o: _patch(_patch(o, TIME_IDX, np.zeros(2, dtype=np.float32)),
                                              WEATHER_IDX, CLEAR_WEATHER),
}


def _patch(obs: np.ndarray, idx: slice, values: np.ndarray) -> np.ndarray:
    out = obs.copy()
    out[idx] = values
    return out


def ablated_policy(model, transform):
    def _policy(env, obs, mask) -> int:
        action, _ = model.predict(transform(obs), action_masks=mask, deterministic=True)
        return int(action)
    return _policy


def ablation_test(model, gui: bool) -> None:
    """Agent auf den Testszenarien, jeweils mit verändertem Beobachtungsvektor."""
    table = {}
    for label, transform in ABLATIONS.items():
        print(f"\n=== Ablation: {label} ===")
        table[label] = evaluate({"agent": ablated_policy(model, transform)}, C.EVAL_SCENARIOS, gui)["agent"]

    base = table["voll"]["wait_per_vehicle"]
    print("\nZusammenfassung Ablation (Mittel über die drei Szenarien)")
    print(f"{'Beobachtung':24s} {'wait/veh [s]':>13s} {'vs. voll':>9s} {'queue':>7s} {'teleports':>10s} {'return':>9s}")
    for label, s in table.items():
        print(f"{label:24s} {s['wait_per_vehicle']:13.1f} {s['wait_per_vehicle'] - base:+8.1f}s "
              f"{s['mean_queue']:7.1f} {s['teleports']:10.1f} {s['ret']:9.2f}")

    out = C.LOG_DIR / "ablation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    print(f"\ngespeichert: {out}")


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


def evaluate(policies: dict, scenarios: list, gui: bool) -> dict:
    """Fährt jedes Szenario mit jeder Policy und gibt die Mittelwerte je Policy zurück."""
    results = {name: [] for name in policies}
    for sc in scenarios:
        for name, policy in policies.items():
            env = SumoEnvironment(env_id=200, gui=gui, scenarios=[sc])
            info = run_episode(env, policy)
            env.close()
            results[name].append(info)
            print(f"{name:7s} | {sc['start_hour']:02d}:00 {sc['duration_s'] // 3600}h "
                  f"{(sc['weather_file'] or 'klar'):24s} | wait/veh {info['wait_per_vehicle']:6.1f} s | "
                  f"queue {info['mean_queue']:5.1f} | arrived {info['arrived']:4d} | "
                  f"teleports {info['teleports']:2d} | return {info['return']:8.2f}", flush=True)

    print("\nMittelwert aller Szenarien")
    print(f"{'policy':8s} {'wait/veh [s]':>13s} {'queue':>7s} {'arrived':>8s} {'teleports':>10s} {'return':>9s}")
    summary = {}
    for name, infos in results.items():
        m = lambda k: float(np.mean([i[k] for i in infos]))
        summary[name] = dict(wait_per_vehicle=m("wait_per_vehicle"), mean_queue=m("mean_queue"),
                             arrived=m("arrived"), teleports=m("teleports"), ret=m("return"))
        print(f"{name:8s} {m('wait_per_vehicle'):13.1f} {m('mean_queue'):7.1f} "
              f"{m('arrived'):8.0f} {m('teleports'):10.1f} {m('return'):9.2f}", flush=True)
    return summary


def stress_test(policies: dict, gui: bool) -> None:
    """Dieselben Szenarien unter Bedingungen, die im Training nicht vorkamen."""
    table = {}
    for label, variant in C.STRESS_VARIANTS.items():
        print(f"\n=== Stresstest: {label} ===")
        scenarios = [dict(sc, **variant) for sc in C.EVAL_SCENARIOS]
        table[label] = evaluate(policies, scenarios, gui)

    print("\nZusammenfassung Stresstest (Mittel über die drei Szenarien)")
    head = f"{'Variante':20s}"
    for name in policies:
        head += f" | {name + ' wait':>12s} {name + ' queue':>12s} {name + ' telep.':>13s}"
    print(head)
    for label, summary in table.items():
        row = f"{label:20s}"
        for name in policies:
            s = summary[name]
            row += f" | {s['wait_per_vehicle']:10.1f} s {s['mean_queue']:12.1f} {s['teleports']:13.1f}"
        print(row)

    out = C.LOG_DIR / "stress_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2))
    print(f"\ngespeichert: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=C.LOG_DIR / "best_model" / "best_model.zip")
    parser.add_argument("--policy", choices=["all", "agent", "fixed", "random"], default="all")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--stress", action="store_true",
                        help="Szenarien mit Verkehrsfaktor 1.5/2.0 und einseitiger Last (config.STRESS_VARIANTS)")
    parser.add_argument("--ablation", action="store_true",
                        help="Agent mit ausgeblendeter Uhrzeit / Wetter in der Beobachtung")
    args = parser.parse_args()

    if args.ablation:
        if not args.model.is_file():
            raise SystemExit(f"Modell nicht gefunden: {args.model}")
        ablation_test(MaskablePPO.load(args.model, device="cpu"), args.gui)
        return

    policies = {}
    if args.policy in ("all", "agent"):
        if not args.model.is_file():
            raise SystemExit(f"Modell nicht gefunden: {args.model}")
        policies["agent"] = agent_policy(MaskablePPO.load(args.model, device="cpu"))
    if args.policy in ("all", "fixed"):
        policies["fixed"] = fixed_time_policy
    if args.policy in ("all", "random") and not args.stress:
        policies["random"] = random_policy

    if args.stress:
        stress_test(policies, args.gui)
        return

    summary = evaluate(policies, C.EVAL_SCENARIOS, args.gui)

    # Mittelwerte merken – plot_training.py zeichnet daraus die Referenzlinien
    out = C.LOG_DIR / "eval_results.json"
    stored = json.loads(out.read_text()) if out.is_file() else {}
    stored.update(summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stored, indent=2))


if __name__ == "__main__":
    main()
