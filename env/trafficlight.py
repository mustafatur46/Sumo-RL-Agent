"""
Ampelsteuerung: setzt die Signalbilder in SUMO und erzwingt Gelb / Rot-Gelb
sowie Mindest- und Höchstgrünzeiten.

Die Umgebung ruft `switch_sequence()` auf und fährt die zurückgegebenen
Phasen selbst durch die Simulation, damit jeder Simulationsschritt für den
Reward sichtbar bleibt.
"""
import traci
from traci import trafficlight as tl

import config as C


class TrafficLightController:
    def __init__(self, conn, tl_id: str = C.TL_ID):
        self.conn = conn
        self.tl_id = tl_id
        self.current_phase = 0        # Agent-Phase (0..N_PHASES-1)
        self.green_elapsed = 0        # Sekunden seit Beginn der aktuellen Grünphase

    # ------------------------------------------------------------------ setup
    def install_program(self) -> None:
        """
        Installiert das 12-Phasen-Programm mit sehr langen Phasendauern, damit
        SUMO niemals von allein weiterschaltet – nur der Agent schaltet.
        """
        logic = self.conn.trafficlight.getAllProgramLogics(self.tl_id)[0]
        phases = [tl.Phase(10_000, p.state) for p in logic.phases]
        new_logic = tl.Logic("rl", 0, 0, phases)
        self.conn.trafficlight.setProgramLogic(self.tl_id, new_logic)
        self.conn.trafficlight.setProgram(self.tl_id, "rl")
        self.set_green(0)

    def set_green(self, phase: int) -> None:
        self.conn.trafficlight.setPhase(self.tl_id, C.GREEN_PHASES[phase])
        self.current_phase = phase
        self.green_elapsed = 0

    # --------------------------------------------------------------- queries
    def can_switch(self) -> bool:
        return self.green_elapsed >= C.MIN_GREEN

    def must_switch(self) -> bool:
        return self.green_elapsed >= C.MAX_GREEN

    def action_mask(self):
        """True = Aktion erlaubt. Index = Agent-Phase."""
        mask = [True] * C.N_PHASES
        if not self.can_switch():
            mask = [False] * C.N_PHASES
            mask[self.current_phase] = True
        elif self.must_switch():
            mask[self.current_phase] = False
        return mask

    # --------------------------------------------------------------- control
    def switch_sequence(self, target: int):
        """
        Liefert die Zwischenphasen (SUMO-Phasenindex, Dauer) für den Wechsel
        von der aktuellen Grünphase auf `target`.
        """
        return [
            (C.YELLOW_PHASES[self.current_phase], C.YELLOW_TIME),
            (C.REDYELLOW_PHASES[target], C.REDYELLOW_TIME),
        ]

    def set_sumo_phase(self, sumo_phase: int) -> None:
        self.conn.trafficlight.setPhase(self.tl_id, sumo_phase)

    def tick(self, seconds: int = 1) -> None:
        self.green_elapsed += seconds
