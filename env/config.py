"""
Zentrale Konfiguration: Pfade, Kreuzungs-Layout und Lern-Konstanten.

Alles, was Training und Evaluation gemeinsam brauchen, steht hier – damit
beide garantiert dieselbe Kreuzung, dieselben Phasen und dieselben Skalen
benutzen.
"""
from pathlib import Path

# ── Pfade (relativ zum Projekt-Root, unabhängig vom Arbeitsverzeichnis) ──────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_CFG     = PROJECT_ROOT / "oneJunction" / "oneJunctionLearn.sumocfg"
WEATHER_DIR  = PROJECT_ROOT / "oneJunction" / "weatherFiles"
LOG_DIR      = PROJECT_ROOT / "logs"

# ── Kreuzung J13 ─────────────────────────────────────────────────────────────
TL_ID = "J13"

# Agent-Phase -> SUMO-Phasenindex (Programm "custom" in oneJunction.net.xml)
GREEN_PHASES     = {0: 0,  1: 3, 2: 6, 3: 9}    # N/S gerade, N/S links, E/W gerade, E/W links
YELLOW_PHASES    = {0: 1,  1: 4, 2: 7, 3: 10}   # Gelb nach der jeweiligen Grünphase
REDYELLOW_PHASES = {0: 11, 1: 2, 2: 5, 3: 8}    # Rot-Gelb vor der jeweiligen Grünphase
N_PHASES = len(GREEN_PHASES)

# Welche Zufahrtsspuren bekommen in welcher Agent-Phase Grün?
PHASE_LANES = {
    0: ["-E8_0", "-E8_1", "-E6_0", "-E6_1"],
    1: ["-E8_2", "-E6_2"],
    2: ["-E7_0", "-E7_1", "-E9_0", "-E9_1"],
    3: ["-E7_2", "-E9_2"],
}
INCOMING_LANES = [lane for lanes in PHASE_LANES.values() for lane in lanes]

# ── Signalzeiten (Sekunden) ──────────────────────────────────────────────────
DECISION_INTERVAL = 5     # Agent entscheidet alle 5 s
MIN_GREEN         = 10    # Grün darf frühestens nach 10 s beendet werden
MAX_GREEN         = 60    # Grün muss spätestens nach 60 s beendet werden
YELLOW_TIME       = 3
REDYELLOW_TIME    = 1

# ── Beobachtungs-Skalen (für Normierung auf [0, 1]) ─────────────────────────
LANE_LENGTH_M      = 186.4
VEH_SPACE_M        = 7.5                         # ~5 m Fahrzeug + 2.5 m Lücke
LANE_CAPACITY      = LANE_LENGTH_M / VEH_SPACE_M # ≈ 25 Fahrzeuge je Spur
WAIT_SCALE_S       = 300.0                       # Wartezeit-Summe je Gruppe
ETA_SCALE_S        = 60.0
PRECIP_SCALE       = 10.0                        # mm/h
VISIBILITY_SCALE_M = 2000.0

# ── Reward ───────────────────────────────────────────────────────────────────
# r = -(Summe haltender Fahrzeuge über alle Sim-Sekunden der Entscheidung)
#     / DECISION_INTERVAL / QUEUE_SCALE  - TELEPORT_PENALTY * Teleports
# Die SUMME (nicht der Mittelwert) sorgt dafür, dass ein Wechsel (9 s inkl.
# Gelb/Rot-Gelb) seine verlorene Zeit auch bezahlt.
QUEUE_SCALE      = 40.0   # 40 haltende Fahrzeuge über 5 s -> Reward -1
TELEPORT_PENALTY = 0.5    # je teleportiertem Fahrzeug (Stau-Kollaps)
REWARD_MIN       = -4.0

# Untergrenze für den Wetter-Geschwindigkeitsfaktor (Sicherheitsnetz)
SPEED_FACTOR_MIN = 0.5

# ── Curriculum (Episodenlänge/Verkehr wachsen mit dem Level) ────────────────
# min/max_episodes gelten global über alle parallelen Envs.
CURRICULUM = [
    dict(duration_s=3600, hours=[7, 8, 9, 16, 17, 18], weather_prob=0.0,
         min_episodes=40,  max_episodes=120),
    dict(duration_s=7200, hours=[6, 7, 8, 9, 10, 11, 15, 16, 17, 18], weather_prob=0.5,
         min_episodes=40,  max_episodes=120),
    dict(duration_s=10800, hours=list(range(0, 24)), weather_prob=1.0,
         min_episodes=60,  max_episodes=10**9),
]
CURRICULUM_PATIENCE = 30   # Episoden ohne Verbesserung des gleitenden Mittels -> Level-Up

# Feste Evaluations-Szenarien: gleicher Seed/Stunde/Wetter bei jeder Auswertung,
# damit die Lernkurve vergleichbar ist.
EVAL_SCENARIOS = [
    dict(seed=1001, start_hour=8,  duration_s=3600, weather_file="freibergRain.json"),
    dict(seed=1002, start_hour=17, duration_s=3600, weather_file="AlaskaSnow01_20.json"),
    dict(seed=1003, start_hour=12, duration_s=3600, weather_file=None),
]
