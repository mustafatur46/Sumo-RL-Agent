"""
Start/Stop einer SUMO-Instanz über TraCI.

Jede Umgebung bekommt eine eigene TraCI-Verbindung (eigenes Label, eigener
Port). Dadurch können mehrere Envs im selben Prozess (DummyVecEnv) oder in
Unterprozessen (SubprocVecEnv) parallel laufen, ohne sich zu stören.
"""
import os
import logging
import traci

log = logging.getLogger(__name__)


def sumo_binary(gui: bool) -> str:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise EnvironmentError(
            "SUMO_HOME ist nicht gesetzt. Beispiel (PowerShell): "
            "$env:SUMO_HOME = 'C:\\Program Files (x86)\\Eclipse\\Sumo'"
        )
    name = "sumo-gui" if gui else "sumo"
    binary = os.path.join(sumo_home, "bin", name)
    if os.name == "nt":
        binary += ".exe"
    if not os.path.isfile(binary):
        raise FileNotFoundError(f"SUMO-Binary nicht gefunden: {binary}")
    return binary


def start_sumo(cfg_path: str, begin: int, end: int, seed: int, gui: bool, label: str):
    """Startet SUMO und gibt das TraCI-Connection-Objekt zurück."""
    cmd = [
        sumo_binary(gui),
        "-c", str(cfg_path),
        "--begin", str(int(begin)),
        "--end", str(int(end)),
        "--seed", str(int(seed)),
        "--time-to-teleport", "300",     # Stau-Kollaps wird als Teleport sichtbar (und bestraft)
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--duration-log.disable", "true",
        "--waiting-time-memory", "10000",
        "--start", "true",
        "--quit-on-end", "true",
    ]
    # port=None -> TraCI sucht selbst einen freien Port
    traci.start(cmd, port=None, label=label, doSwitch=False)
    return traci.getConnection(label)


def close_sumo(conn) -> None:
    """Schließt die Verbindung; SUMO beendet sich dann von selbst."""
    if conn is None:
        return
    label = None
    try:
        label = conn.getLabel()
        conn.close()
    except Exception as exc:           # Verbindung evtl. schon tot
        log.debug("close_sumo: %s", exc)
        # traci entfernt das Label nur bei erfolgreichem close() aus seiner Registry
        try:
            traci.connection._connections.pop(label, None)
        except Exception:
            pass
