"""
Training des Ampel-Agenten mit MaskablePPO.

Beispiele (aus dem Projekt-Root, SUMO_HOME muss gesetzt sein):
    python env/training.py                       # 4 Envs, 2 Mio. Schritte
    python env/training.py --num-envs 1 --gui    # zum Zuschauen
    python env/training.py --resume              # letzten Checkpoint fortsetzen
    python env/training.py --level 2             # Curriculum-Level festnageln
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config as C
from callbacks import CurriculumCallback
from curriculum import CurriculumManager
from environment import SumoEnvironment
from mask_utils import mask_fn

CKPT_PREFIX = "maskable_ppo"
INFO_KEYS = ("wait_per_vehicle", "mean_queue", "teleports", "forced", "level")


def make_env(env_id: int, gui: bool = False, scenarios: list | None = None, level: int = 0):
    def _init():
        env = SumoEnvironment(env_id=env_id, gui=gui, scenarios=scenarios, level=level)
        env = Monitor(env, info_keywords=INFO_KEYS)
        return ActionMasker(env, mask_fn)
    return _init


def build_vec_env(num_envs: int, gui: bool, level: int):
    fns = [make_env(i, gui, level=level) for i in range(num_envs)]
    if num_envs == 1:
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def latest_model(log_dir: Path) -> Path | None:
    """Neuester Stand: periodischer Checkpoint oder final_model.zip (bei Abbruch/Ende)."""
    candidates = list((log_dir / "checkpoints").glob(f"{CKPT_PREFIX}_*_steps.zip"))
    final = log_dir / "final_model.zip"
    if final.is_file():
        candidates.append(final)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def restore_eval_state(eval_cb: MaskableEvalCallback, log_dir: Path) -> None:
    """Bisherige Evaluationen weiterführen, statt best_model beim ersten Eval zu überschreiben."""
    npz = log_dir / "eval_logs" / "evaluations.npz"
    if not npz.is_file():
        return
    data = np.load(npz)
    eval_cb.evaluations_timesteps = list(data["timesteps"])
    eval_cb.evaluations_results = list(data["results"])
    eval_cb.evaluations_length = list(data["ep_lengths"])
    eval_cb.best_mean_reward = float(max(np.mean(r) for r in data["results"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=C.LOG_DIR)
    parser.add_argument("--device", default="cpu", help="cpu ist bei diesem kleinen Netz meist schneller als cuda")
    parser.add_argument("--resume", action="store_true", help="letzten Checkpoint aus --log-dir fortsetzen")
    parser.add_argument("--level", type=int, default=None, help="Curriculum-Level fest vorgeben (0..%d)" % (len(C.CURRICULUM) - 1))
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=25_000, help="Evaluations-Intervall in Env-Schritten")
    parser.add_argument("--ckpt-freq", type=int, default=50_000)
    args = parser.parse_args()

    log_dir: Path = args.log_dir
    ckpt_dir = log_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_dir / "training.log"), logging.StreamHandler(sys.stdout)],
    )
    if not C.SUMO_CFG.is_file():
        sys.exit(f"SUMO-Config nicht gefunden: {C.SUMO_CFG}")

    curriculum = CurriculumManager(log_dir / "curriculum_state.json", level=args.level)
    ckpt = latest_model(log_dir) if args.resume else None
    if ckpt is not None and curriculum.load():
        logging.info("Curriculum geladen: Level %d, %d Episoden", curriculum.level, curriculum.total_episodes)

    # Level schon beim Bau übergeben: SB3 resettet die Envs VOR on_training_start()
    train_env = build_vec_env(args.num_envs, args.gui, curriculum.level)
    eval_env = DummyVecEnv([make_env(100, False, C.EVAL_SCENARIOS)])

    # n_steps je Env so wählen, dass ein Rollout ~2048 Schritte hat
    n_steps = max(2048 // args.num_envs, 64)
    hyper = dict(
        n_steps=n_steps, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    if ckpt is not None:
        logging.info("Setze fort ab %s", ckpt)
        model = MaskablePPO.load(ckpt, env=train_env, device=args.device)
    else:
        if args.resume:
            logging.warning("Kein Checkpoint in %s – starte neu.", log_dir)
        model = MaskablePPO(
            "MlpPolicy", train_env, device=args.device, seed=args.seed, verbose=1,
            tensorboard_log=str(log_dir / "tb_logs"), **hyper,
        )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(log_dir / "best_model"),
        log_path=str(log_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=len(C.EVAL_SCENARIOS),
        deterministic=True,
    )
    if ckpt is not None:
        restore_eval_state(eval_cb, log_dir)
    callbacks = [
        CurriculumCallback(curriculum),
        CheckpointCallback(
            save_freq=max(args.ckpt_freq // args.num_envs, 1),
            save_path=str(ckpt_dir), name_prefix=CKPT_PREFIX,
        ),
        eval_cb,
    ]

    logging.info("Training: envs=%d, n_steps=%d, timesteps=%d, device=%s",
                 args.num_envs, n_steps, args.timesteps, args.device)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=ckpt is None, tb_log_name="run", progress_bar=True)
    except KeyboardInterrupt:
        logging.info("Abbruch durch Benutzer – speichere Zwischenstand.")
    finally:
        model.save(str(log_dir / "final_model"))
        curriculum.save()
        train_env.close()
        eval_env.close()
    logging.info("Modell gespeichert: %s", log_dir / "final_model.zip")


if __name__ == "__main__":
    main()
