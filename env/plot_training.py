"""
Lernkurven als PNG: Evaluation auf den festen Szenarien, Wartezeit und
Warteschlange im Training, Curriculum-Level – jeweils mit Festzeit-Ampel und
Zufall als Referenzlinien (aus logs/eval_results.json, erzeugt von evaluate.py).

    python env/plot_training.py                 # -> logs/training_plot.png
    python env/plot_training.py --show          # zusätzlich anzeigen
    python env/plot_training.py --log-dir logs_testlauf_150k
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import config as C

TB_TAGS = {
    "wait": "traffic/wait_per_vehicle",
    "queue": "traffic/mean_queue",
    "level": "curriculum/level",
    "ep_rew": "rollout/ep_rew_mean",
}


def load_tb_scalars(tb_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Alle run_*-Ordner einlesen und je Tag nach Schritt sortiert zusammenführen."""
    series = {k: ([], []) for k in TB_TAGS}
    for run in sorted(tb_dir.glob("run*")):
        ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
        ea.Reload()
        tags = set(ea.Tags()["scalars"])
        for key, tag in TB_TAGS.items():
            if tag in tags:
                for ev in ea.Scalars(tag):
                    series[key][0].append(ev.step)
                    series[key][1].append(ev.value)
    out = {}
    for key, (steps, vals) in series.items():
        if steps:
            order = np.argsort(steps)
            out[key] = (np.asarray(steps)[order], np.asarray(vals)[order])
    return out


def smooth(y: np.ndarray, window: int = 20) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    pad = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=C.LOG_DIR)
    parser.add_argument("--out", type=Path, default=None, help="PNG-Pfad (Default: <log-dir>/training_plot.png)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    out = args.out or args.log_dir / "training_plot.png"
    if not args.show:
        matplotlib.use("Agg")

    baselines = {}
    bl_file = C.LOG_DIR / "eval_results.json"
    if bl_file.is_file():
        baselines = json.loads(bl_file.read_text())

    tb = load_tb_scalars(args.log_dir / "tb_logs")
    npz = args.log_dir / "eval_logs" / "evaluations.npz"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"SumoRLAgent – training curves ({args.log_dir.name})", fontsize=14)

    # 1) Evaluation auf festen Szenarien
    ax = axes[0, 0]
    if npz.is_file():
        d = np.load(npz)
        steps = d["timesteps"]
        mean = d["results"].mean(axis=1)
        std = d["results"].std(axis=1)
        ax.plot(steps, mean, marker="o", ms=3, label="RL agent (best so far: %.1f)" % mean.max())
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
    for name, color in (("fixed", "tab:red"), ("random", "tab:gray")):
        if name in baselines:
            ax.axhline(baselines[name]["ret"], color=color, ls="--",
                       label=f"{'fixed-time signal' if name == 'fixed' else 'random'} ({baselines[name]['ret']:.0f})")
    ax.set_title("Evaluation: return on the fixed scenarios (higher is better)")
    ax.set_xlabel("training steps")
    ax.set_ylabel("episode return")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Wartezeit je Fahrzeug im Training
    ax = axes[0, 1]
    if "wait" in tb:
        s, v = tb["wait"]
        ax.plot(s, v, alpha=0.25, color="tab:blue")
        ax.plot(s, smooth(v), color="tab:blue", label="training (smoothed)")
    if "fixed" in baselines:
        ax.axhline(baselines["fixed"]["wait_per_vehicle"], color="tab:red", ls="--",
                   label=f"fixed-time signal ({baselines['fixed']['wait_per_vehicle']:.1f} s)")
    ax.set_title("Waiting time per vehicle (lower is better)")
    ax.set_xlabel("training steps")
    ax.set_ylabel("seconds")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) Warteschlange im Training
    ax = axes[1, 0]
    if "queue" in tb:
        s, v = tb["queue"]
        ax.plot(s, v, alpha=0.25, color="tab:green")
        ax.plot(s, smooth(v), color="tab:green", label="training (smoothed)")
    if "fixed" in baselines:
        ax.axhline(baselines["fixed"]["mean_queue"], color="tab:red", ls="--",
                   label=f"fixed-time signal ({baselines['fixed']['mean_queue']:.1f})")
    ax.set_title("Mean number of halting vehicles (lower is better)")
    ax.set_xlabel("training steps")
    ax.set_ylabel("vehicles")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4) Episoden-Return und Curriculum-Level
    ax = axes[1, 1]
    if "ep_rew" in tb:
        s, v = tb["ep_rew"]
        ax.plot(s, v, color="tab:purple", label="episode return (training)")
        ax.set_ylabel("return", color="tab:purple")
    if "level" in tb:
        ax2 = ax.twinx()
        s, v = tb["level"]
        ax2.step(s, v, where="post", color="tab:orange", label="curriculum level")
        ax2.set_ylabel("level", color="tab:orange")
        ax2.set_yticks(range(len(C.CURRICULUM)))
    ax.set_title("Training return and curriculum level")
    ax.set_xlabel("training steps")
    ax.grid(alpha=0.3)
    lines, labels = ax.get_legend_handles_labels()
    if "level" in tb:
        l2, lb2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lb2
    ax.legend(lines, labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"gespeichert: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
