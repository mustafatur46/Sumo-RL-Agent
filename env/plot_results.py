"""
Ergebnis-Plot als PNG aus den JSON-Dateien von evaluate.py:
Evaluation (Agent / Festzeit / Zufall), Stresstest (--stress) und
Ablation der Beobachtung (--ablation).

    python env/plot_results.py                 # -> logs/results_plot.png
    python env/plot_results.py --show
"""
import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import config as C

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"

ABLATION_LABELS = {
    "voll":                  "full observation",
    "ohne Uhrzeit":          "no time of day",
    "Uhrzeit +12 h":         "time shifted +12 h",
    "Wetter = klar":         "weather masked (clear)",
    "ohne Uhrzeit + Wetter": "no time, no weather",
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _style(ax, title: str, xlabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9dce3")
    ax.grid(axis="x", color="#e6e8ee", lw=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()


def _label(ax, bars, fmt="%.1f s") -> None:
    for b in bars:
        ax.text(b.get_width() + 0.3, b.get_y() + b.get_height() / 2, fmt % b.get_width(),
                va="center", ha="left", fontsize=9, color=INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=C.LOG_DIR)
    parser.add_argument("--out", type=Path, default=None, help="PNG-Pfad (Default: <log-dir>/results_plot.png)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    out = args.out or args.log_dir / "results_plot.png"
    if not args.show:
        matplotlib.use("Agg")

    ev = _load(args.log_dir / "eval_results.json")
    st = _load(args.log_dir / "stress_results.json")
    ab = _load(args.log_dir / "ablation_results.json")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
    fig.suptitle("SumoRLAgent – results on the fixed evaluation scenarios "
                 "(rain 08:00, snow 17:00, clear 12:00; 1 h each, mean)", fontsize=12, x=0.01, ha="left")

    # 1) Evaluation: Wartezeit je Fahrzeug je Policy
    ax = axes[0]
    if ev:
        names = [n for n in ("agent", "fixed", "random") if n in ev]
        labels = {"agent": "RL agent", "fixed": "fixed-time signal", "random": "random"}
        colors = {"agent": BLUE, "fixed": ORANGE, "random": GRAY}
        vals = [ev[n]["wait_per_vehicle"] for n in names]
        bars = ax.barh([labels[n] for n in names], vals, color=[colors[n] for n in names], height=0.55)
        _label(ax, bars)
        ax.set_xlim(0, max(vals) * 1.25)
    _style(ax, "Waiting time per vehicle", "seconds (lower is better)")

    # 2) Stresstest: Agent vs. Festzeit unter Last, die im Training nicht vorkam
    ax = axes[1]
    if st:
        variants = list(st)
        y = np.arange(len(variants))
        h = 0.36
        a = [st[v]["agent"]["wait_per_vehicle"] for v in variants]
        f = [st[v]["fixed"]["wait_per_vehicle"] for v in variants]
        b1 = ax.barh(y - h / 2, a, height=h - 0.04, color=BLUE, label="RL agent")
        b2 = ax.barh(y + h / 2, f, height=h - 0.04, color=ORANGE, label="fixed-time signal")
        _label(ax, b1)
        _label(ax, b2)
        ax.set_yticks(y)
        ax.set_yticklabels([v.replace("x1.0 (Training)", "x1.0 (as trained)")
                             .replace("asym West +350/h", "west arm +350 veh/h") for v in variants])
        ax.set_xlim(0, max(f) * 1.2)
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9)
    _style(ax, "Stress test: unseen traffic load", "seconds per vehicle (lower is better)")

    # 3) Ablation: Teile der Beobachtung ausblenden
    ax = axes[2]
    if ab:
        keys = list(ab)
        vals = [ab[k]["wait_per_vehicle"] for k in keys]
        bars = ax.barh([ABLATION_LABELS.get(k, k) for k in keys], vals,
                       color=[BLUE if k == "voll" else "#9ec5f4" for k in keys], height=0.55)
        _label(ax, bars)
        ax.set_xlim(0, max(vals) * 1.25)
    _style(ax, "Ablation: what the agent sees", "seconds per vehicle (lower is better)")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"gespeichert: {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
