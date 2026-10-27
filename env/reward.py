"""
Reward: beschränkt, dicht und direkt am Ziel orientiert.

    r = -(Summe haltender Fahrzeuge je Sim-Sekunde) / DECISION_INTERVAL / QUEUE_SCALE
        - TELEPORT_PENALTY * Teleports

Ein "Phase halten" dauert 5 s, ein Wechsel 9 s (Gelb + Rot-Gelb + Grün); die
Summe stellt sicher, dass die Wartezeit während Gelb nicht unter den Tisch
fällt. Der Wert liegt in [REWARD_MIN, 0]: freier Verkehr -> ~0, Stau -> negativ.
"""
import traci.constants as tc

import config as C


class RewardCalculator:
    def __init__(self):
        self.last_halting = 0
        self.reset()

    def reset(self) -> None:
        self._halting_sum = 0.0
        self._n = 0
        self._teleports = 0

    def observe_step(self, lane_results: dict, teleports: int) -> None:
        """Nach jedem Simulationsschritt aufrufen (auch während Gelb)."""
        halting = sum(lane_results[l][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for l in C.INCOMING_LANES)
        self.last_halting = halting
        self._halting_sum += halting
        self._n += 1
        self._teleports += teleports

    def compute(self) -> tuple[float, dict]:
        """Reward für das abgeschlossene Entscheidungsintervall."""
        mean_halting = self._halting_sum / max(self._n, 1)
        teleports = self._teleports

        reward = (-self._halting_sum / C.DECISION_INTERVAL / C.QUEUE_SCALE
                  - C.TELEPORT_PENALTY * teleports)
        reward = max(C.REWARD_MIN, reward)

        info = {"queue": mean_halting, "teleports": teleports}
        self.reset()
        return float(reward), info
