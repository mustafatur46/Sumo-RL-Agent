"""
Curriculum: eine einzige Instanz im Trainings-Hauptprozess entscheidet über
das Level und schiebt es per env_method("set_level") in alle Envs.

Level-Up, wenn
    * die Mindestzahl Episoden im Level erreicht ist UND der gleitende
      Mittelwert des Rewards seit CURRICULUM_PATIENCE Episoden nicht mehr
      besser wurde (Plateau)  – oder –
    * die Höchstzahl Episoden erreicht ist.

Das Kriterium ist skalenfrei: es braucht keine handgewählten Reward-Schwellen.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np

import config as C


class CurriculumManager:
    WINDOW = 20

    def __init__(self, state_path: Path, level: int | None = None):
        self.state_path = Path(state_path)
        self.level = 0
        self.episodes_at_level = 0
        self.best_rolling = -np.inf
        self.since_best = 0
        self.total_episodes = 0
        self._recent = deque(maxlen=self.WINDOW)
        if level is not None:
            self.level = int(np.clip(level, 0, self.max_level))
            self.fixed = True
        else:
            self.fixed = False

    @property
    def max_level(self) -> int:
        return len(C.CURRICULUM) - 1

    @property
    def rolling_reward(self) -> float:
        return float(np.mean(self._recent)) if self._recent else float("nan")

    def record_episode(self, ep_return: float, ep_len: int) -> bool:
        """Episode eintragen. Gibt True zurück, wenn ein Level-Up erfolgt ist."""
        self._recent.append(ep_return / max(ep_len, 1))   # Reward je Entscheidung
        self.episodes_at_level += 1
        self.total_episodes += 1

        if len(self._recent) == self.WINDOW:
            if self.rolling_reward > self.best_rolling + 1e-3:
                self.best_rolling = self.rolling_reward
                self.since_best = 0
            else:
                self.since_best += 1

        if self.fixed or self.level >= self.max_level or not self._should_level_up():
            return False
        self.level += 1
        self.episodes_at_level = 0
        self.best_rolling = -np.inf
        self.since_best = 0
        self._recent.clear()
        return True

    def _should_level_up(self) -> bool:
        cfg = C.CURRICULUM[self.level]
        if self.episodes_at_level >= cfg["max_episodes"]:
            return True
        if self.episodes_at_level < cfg["min_episodes"]:
            return False
        return self.since_best >= C.CURRICULUM_PATIENCE

    # ------------------------------------------------------------- persist
    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(level=self.level, episodes_at_level=self.episodes_at_level,
                     best_rolling=float(self.best_rolling), since_best=self.since_best,
                     total_episodes=self.total_episodes, recent=list(self._recent))
        self.state_path.write_text(json.dumps(state, indent=2))

    def load(self) -> bool:
        if not self.state_path.is_file():
            return False
        state = json.loads(self.state_path.read_text())
        if not self.fixed:
            self.level = int(state["level"])
        self.episodes_at_level = state["episodes_at_level"]
        self.best_rolling = float(state["best_rolling"])
        self.since_best = state["since_best"]
        self.total_episodes = state["total_episodes"]
        self._recent = deque(state["recent"], maxlen=self.WINDOW)
        return True
