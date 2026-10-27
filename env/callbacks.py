"""
Trainings-Callbacks: Curriculum-Steuerung und Verkehrsmetriken in TensorBoard.
"""
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from curriculum import CurriculumManager


class CurriculumCallback(BaseCallback):
    """
    Liest Episodenenden aus den Monitor-Infos, führt das Curriculum und
    schreibt die wichtigsten "lernt der Agent?"-Kurven:

        curriculum/level             aktuelles Level
        curriculum/rolling_reward    Reward je Entscheidung (gleitend, 20 Episoden)
        traffic/wait_per_vehicle     Sekunden Wartezeit je Fahrzeug (gleitend)
        traffic/mean_queue           haltende Fahrzeuge im Mittel (gleitend)
        traffic/teleports            Teleports je Episode (gleitend)
    """

    def __init__(self, curriculum: CurriculumManager, verbose: int = 1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self._wait = deque(maxlen=20)
        self._queue = deque(maxlen=20)
        self._tele = deque(maxlen=20)
        self._forced = deque(maxlen=20)

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_level", self.curriculum.level)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is None:
                continue
            self._wait.append(info.get("wait_per_vehicle", 0.0))
            self._queue.append(info.get("mean_queue", 0.0))
            self._tele.append(info.get("teleports", 0))
            self._forced.append(info.get("forced", 0))

            leveled_up = self.curriculum.record_episode(ep["r"], ep["l"])
            if leveled_up:
                self.training_env.env_method("set_level", self.curriculum.level)
                if self.verbose:
                    print(f"[CURRICULUM] Level-Up -> {self.curriculum.level} "
                          f"nach {self.curriculum.total_episodes} Episoden")
            self.curriculum.save()

            self.logger.record("curriculum/level", self.curriculum.level)
            self.logger.record("curriculum/episodes_at_level", self.curriculum.episodes_at_level)
            if not np.isnan(self.curriculum.rolling_reward):
                self.logger.record("curriculum/rolling_reward", self.curriculum.rolling_reward)
            self.logger.record("traffic/wait_per_vehicle", float(np.mean(self._wait)))
            self.logger.record("traffic/mean_queue", float(np.mean(self._queue)))
            self.logger.record("traffic/teleports", float(np.mean(self._tele)))
            self.logger.record("traffic/forced_actions", float(np.mean(self._forced)))
        return True
