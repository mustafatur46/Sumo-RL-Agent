# SumoRLAgent — Weather-Aware Traffic Light Control with Reinforcement Learning

A curriculum-learning RL agent that controls a single-junction traffic light in **SUMO** (Simulation of Urban MObility).  
The agent learns to adapt phase timing to live traffic demand *and* real-world weather data (rain, snow, fog) that affect driver behaviour.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [SUMO Installation](#sumo-installation)
   - [Windows](#windows)
   - [Linux (Ubuntu / Debian)](#linux-ubuntu--debian)
   - [macOS](#macos)
4. [Setting up SUMO\_HOME](#setting-up-sumo_home)
   - [Windows (permanent)](#windows-permanent)
   - [Linux / macOS (permanent)](#linux--macos-permanent)
5. [Python Environment Setup](#python-environment-setup)
6. [Project Setup](#project-setup)
7. [Route-Dateien generieren](#route-dateien-generieren)
8. [SUMO-Dateien und ihr Zusammenspiel](#sumo-dateien-und-ihr-zusammenspiel)
9. [Running the Project](#running-the-project)
   - [Training](#training)
   - [Resuming from Checkpoint](#resuming-from-checkpoint)
   - [Evaluation](#evaluation)
   - [Hyperparameter Tuning (Optuna)](#hyperparameter-tuning-optuna)
   - [TensorBoard](#tensorboard)
8. [Project Structure](#project-structure)
9. [Architecture Overview](#architecture-overview)
   - [Observation Space](#observation-space)
   - [Action Space](#action-space)
   - [Reward Function](#reward-function)
   - [Curriculum Learning](#curriculum-learning)
   - [Weather Model](#weather-model)
10. [Configuration Reference](#configuration-reference)
11. [Logs and Outputs](#logs-and-outputs)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

| Component | Technology |
|-----------|-----------|
| Traffic simulator | SUMO ≥ 1.19 |
| RL algorithm | MaskablePPO (sb3-contrib) |
| Curriculum | 4 levels, HDF5-backed performance tracking |
| Weather | Open-Meteo JSON → empirical friction/visibility model |
| Framework | PyTorch + Stable-Baselines3 |

The agent controls a 4-phase traffic light at a single cross-junction.  
It selects one of 4 green phases per simulation step; the `TrafficLightManager` enforces the yellow / red-yellow inter-green period and the minimum green time automatically.

---

## Prerequisites

| Tool | Minimum version | Notes |
|------|----------------|-------|
| SUMO | 1.19.0 | `sumo`, `sumo-gui`, `traci` |
| Python | 3.10 | 3.11 / 3.12 also work |
| CUDA (optional) | 11.8 | For GPU training |
| Git | any | |

---

## SUMO Installation

### Windows

**Option A — Official installer (recommended)**

1. Go to <https://sumo.dlr.de/docs/Downloads.php#windows>
2. Download the latest **64-bit Windows installer** (e.g. `sumo-win64-1.21.0.msi`).
3. Run the installer; accept the default install path  
   (`C:\Program Files (x86)\Eclipse\Sumo`).
4. The installer adds `SUMO_HOME` to your system environment variables automatically.  
   Verify:
   ```powershell
   echo $env:SUMO_HOME
   # Expected output: C:\Program Files (x86)\Eclipse\Sumo
   ```

**Option B — Chocolatey**

```powershell
# Run PowerShell as Administrator
choco install sumo
```

**Option C — winget**

```powershell
winget install Eclipse.SUMO
```

---

### Linux (Ubuntu / Debian)

```bash
# Add the SUMO PPA
sudo add-apt-repository ppa:sumo/stable
sudo apt update
sudo apt install sumo sumo-tools sumo-gui

# Verify
sumo --version
```

For other distributions or the latest nightly build see  
<https://sumo.dlr.de/docs/Downloads.php#linux>

---

### macOS

```bash
# Using Homebrew
brew install sumo

# Verify
sumo --version
```

---

## Setting up SUMO\_HOME

`traci` (the Python API) **requires** the `SUMO_HOME` environment variable to point to the SUMO installation root (the folder that contains `bin/` and `tools/`).

### Windows (permanent)

Run **PowerShell as Administrator**:

```powershell
# Adjust the path if you chose a different install location
[System.Environment]::SetEnvironmentVariable(
    "SUMO_HOME",
    "C:\Program Files (x86)\Eclipse\Sumo",
    [System.EnvironmentVariableTarget]::Machine
)
# Add sumo/bin to PATH so you can call 'sumo' from any terminal
$oldPath = [System.Environment]::GetEnvironmentVariable("PATH","Machine")
[System.Environment]::SetEnvironmentVariable(
    "PATH",
    "$oldPath;C:\Program Files (x86)\Eclipse\Sumo\bin",
    [System.EnvironmentVariableTarget]::Machine
)
```

Close and reopen PowerShell, then verify:

```powershell
echo $env:SUMO_HOME        # → C:\Program Files (x86)\Eclipse\Sumo
sumo --version             # → SUMO 1.x.x
python -c "import traci; print('traci OK')"
```

### Linux / macOS (permanent)

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, …):

```bash
export SUMO_HOME="/usr/share/sumo"          # adjust for your distro/brew path
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

Apply without reopening the terminal:

```bash
source ~/.bashrc   # or ~/.zshrc
```

Verify:

```bash
echo $SUMO_HOME
sumo --version
python3 -c "import traci; print('traci OK')"
```

---

## Python Environment Setup

```bash
# 1) Create a dedicated virtual environment
python -m venv .venv

# 2) Activate it
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3) Upgrade pip
pip install --upgrade pip

# 4) Install all dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# (CPU-only: `pip install torch torchvision torchaudio`)

pip install stable-baselines3==2.3.2
pip install sb3-contrib==2.3.2
pip install gymnasium==0.29.1
pip install traci                # bundled with SUMO; also available on PyPI
pip install numpy pandas h5py
pip install optuna joblib
pip install tensorboard
pip install psutil
```

> **Note:** `traci` is also shipped inside `$SUMO_HOME/tools/`.  
> If the PyPI version conflicts, add that path to `PYTHONPATH`:
> ```bash
> export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
> ```

---

## Project Setup

```bash
# Clone (or unzip) the project
git clone <repo-url> SumoRLAgent
cd SumoRLAgent

# Activate your venv (see above)

# Confirm SUMO is reachable from Python
python -c "
import os, traci
print('SUMO_HOME:', os.environ.get('SUMO_HOME'))
print('traci version:', traci.version)
"
```

No further build steps are needed — the project is pure Python.

---

## Route-Dateien generieren

> Die folgenden Dateien sind in `.gitignore` und **nicht** im Repository enthalten.  
> Sie müssen einmalig erzeugt werden bevor das Training gestartet werden kann.

| Datei | Erzeugt von | Zweck |
|-------|------------|-------|
| `routeFiles/randtrips.rou.xml` | `randomTrips.py` | Zufällige Fahrten als Input für duarouter |
| `routeFiles/oneJunction.trips.xml` | `randomTrips.py` | Alternative Trip-Datei |
| `routeFiles/baseroutes.rou.xml` | `duarouter` | Fertig geroutete Basisrouten (von `oneJunctionLearn.sumocfg` genutzt) |
| `routeFiles/baseroutes.rou.alt.xml` | `duarouter` | Alternative Routenoptionen |
| `routeFiles/merged.rou.xml` | `duarouter` | Zusammengeführte Route-Datei |

> **`routeFiles/flows.rou.xml`** ist manuell erstellt (24-Stunden-Verkehrsmodell mit stündlichen  
> Fahrzeugströmen) und **liegt bereits im Repository**. Diese Datei nicht löschen.

### Schritt 1 – Zufällige Trips erzeugen

```powershell
# Sicherstellen dass SUMO_HOME gesetzt ist
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"

python "$env:SUMO_HOME\tools\randomTrips.py" `
    -n oneJunction\oneJunction.net.xml `
    -o oneJunction\routeFiles\randtrips.rou.xml `
    -b 0 -e 86400 `
    --seed 42 `
    --trip-attributes "depart=`"random`" departLane=`"best`" departSpeed=`"random`""
```

Unter **Linux / macOS**:
```bash
python $SUMO_HOME/tools/randomTrips.py \
    -n oneJunction/oneJunction.net.xml \
    -o oneJunction/routeFiles/randtrips.rou.xml \
    -b 0 -e 86400 --seed 42 \
    --trip-attributes 'depart="random" departLane="best" departSpeed="random"'
```

### Schritt 2 – Routen berechnen (duarouter)

```powershell
# Windows PowerShell
duarouter `
    -n oneJunction\oneJunction.net.xml `
    -r oneJunction\routeFiles\randtrips.rou.xml `
    -o oneJunction\routeFiles\baseroutes.rou.xml `
    --ignore-errors true `
    --no-warnings true
```

```bash
# Linux / macOS
duarouter \
    -n oneJunction/oneJunction.net.xml \
    -r oneJunction/routeFiles/randtrips.rou.xml \
    -o oneJunction/routeFiles/baseroutes.rou.xml \
    --ignore-errors true --no-warnings true
```

`duarouter` erzeugt automatisch auch `baseroutes.rou.alt.xml`.

### Schritt 3 – Merged-Datei erstellen (optional)

```powershell
duarouter `
    -n oneJunction\oneJunction.net.xml `
    -r "oneJunction\routeFiles\baseroutes.rou.xml,oneJunction\routeFiles\flows.rou.xml" `
    -o oneJunction\routeFiles\merged.rou.xml `
    --ignore-errors true
```

> Nach diesen drei Schritten sind alle benötigten Dateien vorhanden und das Training kann gestartet werden.

---

## SUMO-Dateien und ihr Zusammenspiel

### Übersicht: Welche Datei macht was?

```
oneJunction/
├── oneJunction.net.xml          ← Das Straßennetz  (Kanten, Spuren, Kreuzung J13)
├── oneJunction.add.xml          ← Zusatzdatei       (Detektoren, TL-Overrides)
├── oneJunctionLearn.sumocfg     ← Startkonfiguration für Training
├── oneJunctionWeather.sumocfg   ← Startkonfiguration für Evaluation (mit Output)
├── oneJunctionRL.sumocfg        ← Startkonfiguration mit RL-spezifischen Outputs
│
├── routeFiles/
│   ├── flows.rou.xml            ← 24h-Verkehrsmodell   (manuell, stündliche Flows)
│   ├── baseroutes.rou.xml       ← Einzelfahrzeug-Routen (generiert mit duarouter)
│   ├── baseroutes.rou.alt.xml   ← Alternative Routen    (generiert mit duarouter)
│   ├── randtrips.rou.xml        ← Zufällige Trips       (generiert mit randomTrips.py)
│   ├── oneJunction.trips.xml    ← Trip-Datei            (generiert mit randomTrips.py)
│   └── merged.rou.xml           ← Zusammengeführt       (generiert mit duarouter)
│
├── trafficlightFiles/
│   └── staticTrafficlight.tll.xml  ← Statisches Ampel-Programm (Referenz)
│
└── weatherFiles/
    ├── freibergRain.json        ← Echter Wetterdatensatz (Open-Meteo API)
    ├── AlaskaSnow01_20.json
    └── …                        (11 Szenarien gesamt)
```

---

### 1. Das Netz: `oneJunction.net.xml`

Das Netz beschreibt die physische Infrastruktur der 4-Arm-Kreuzung.  
SUMO liest daraus:

- **Kanten** (edges): `-E6`, `-E7`, `-E8`, `-E9` (eingehend) / `E6`, `E7`, `E8`, `E9` (ausgehend)
- **Spuren** (lanes): je 3 Spuren pro Kante → z.B. `-E8_0`, `-E8_1`, `-E8_2`  
  (Spur 0+1 = Geradeaus/Rechts, Spur 2 = Linksabbieger)
- **Kreuzung J13**: Die einzige Ampelkreuzung; der Agent steuert ausschließlich sie
- **Verbindungen** (connections): welche Spur auf welche Ausfahrt führt

Der RL-Agent referenziert diese IDs direkt in seiner Konfiguration:

```python
config = {
    "traffic_light_id": "J13",
    "observation_lanes": ["-E8_0", "-E8_1", "-E8_2", "-E6_0", ...],
    "green_lane_map": {
        0: ["-E8_0","-E8_1","-E6_0","-E6_1"],   # N↔S Geradeaus
        1: ["-E6_2","-E8_2"],                    # N↔S Linksabbieger
        2: ["-E7_0","-E7_1","-E9_0","-E9_1"],   # O↔W Geradeaus
        3: ["-E9_2","-E7_2"],                    # O↔W Linksabbieger
    }
}
```

---

### 2. Die Ampel-Zusatzdatei: `oneJunction.add.xml`

Diese Datei erweitert die Simulation ohne das Netz zu verändern.  
Aktuell ist sie bewusst leer (nur `<!-- Detectors -->`).  
Hier könnten zukünftig hinzukommen:

- **E2-Schleifen-Detektoren** zum Messen von Staulängen
- **Ampel-Programme** (falls nicht per TraCI gesteuert)
- **Induction-Loop-Detektoren** für Fahrzeugzählungen

Das Netz hat ein eingebautes 12-Phasen-Programm (in `oneJunction.net.xml` definiert).  
Das `TrafficLightManager` überschreibt dieses Programm bei jedem Reset via TraCI:

```python
# environment.py – reset()
traci.trafficlight.setProgram("J13", "custom")   # SUMO-Programm deaktivieren
traci.trafficlight.setPhase("J13", 0)             # Agent übernimmt Kontrolle
```

Das statische Referenzprogramm in `trafficlightFiles/staticTrafficlight.tll.xml`  
(12-Phasen-Zyklus: 33 s Grün → 3 s Gelb → 6 s Grün links → 3 s Gelb …)  
kann für Baseline-Vergleiche genutzt werden, ist aber kein Teil des RL-Trainings.

---

### 3. Die `.sumocfg`-Dateien: Startkonfigurationen

Eine `.sumocfg`-Datei fasst alle Eingaben für eine Simulation zusammen.  
Beim Start via `traci.start([sumo_bin, "-c", path_to_sumocfg])` liest SUMO sie aus.

| Datei | Verwendet für | Besonderheit |
|-------|--------------|-------------|
| `oneJunctionLearn.sumocfg` | Training | Kein Output (Geschwindigkeit) |
| `oneJunctionWeather.sumocfg` | Evaluation | Schreibt `tripinfo`, `emission`, `fcd`, `queue` XML-Outputs |
| `oneJunctionRL.sumocfg` | Post-hoc-Analyse | Schreibt RL-spezifische Outputs |

Alle drei nutzen dieselben Eingaben:

```xml
<net-file   value="oneJunction.net.xml"/>
<route-files value="routeFiles/flows.rou.xml,routeFiles/baseroutes.rou.xml"/>
<additional-files value="oneJunction.add.xml"/>
<begin value="0"/>
<end   value="86400"/>
```

Im Training überschreibt `environment.py` die `--begin`- und `--end`-Zeit per  
CLI-Argument, um Episoden kürzer zu machen (Curriculum Level 0 = nur 2–4 Stunden).

---

### 4. Die Route-Dateien: Fahrzeugaufkommen

Zwei Dateien definieren gemeinsam alle Fahrzeuge in der Simulation:

#### `flows.rou.xml` – Das 24h-Verkehrsmodell

Definiert stündliche Fahrzeugströme für alle 12 Quell-Ziel-Relationen (West→Ost, West→Nord, …).  
Jede Stunde hat eigene `<flow>`-Elemente mit `vehsPerHour` und Routenwahrscheinlichkeiten:

```xml
<!-- 8 Uhr morgens, Westen → Osten: 193 Fahrzeuge/Stunde -->
<flow id="flow_west_h8" type="car" begin="28800" end="32400" vehsPerHour="193"
      depart="random" departLane="best" departSpeed="random">
  <routeDistribution>
    <route refId="W_to_E" probability="0.8"/>
    <route refId="W_to_S" probability="0.1"/>
    <route refId="W_to_N" probability="0.1"/>
  </routeDistribution>
</flow>
```

Die 12 Routen (W→E, W→S, W→N, E→W, …) sind am Anfang der Datei definiert.  
Diese Datei bildet das realistische Tagesverkehrsprofil nach (Nacht: 7–17 Fzg/h, Rush-Hour: bis 255 Fzg/h).

#### `baseroutes.rou.xml` – Einzelfahrzeug-Basisrouten

Enthält individuelle Fahrzeuge mit fixen Abfahrtszeiten und Routen,  
generiert von `duarouter` aus zufälligen Trips (`randtrips.rou.xml`).  
Dient als Grundrauschen / Grundbelastung außerhalb der Flow-Definitionen.

```xml
<vehicle id="base_0" depart="0.00">
    <route edges="-E8 E6"/>
</vehicle>
```

---

### 5. Das Wettersystem: JSON → SUMO-Fahrzeugparameter

Das Wettersystem besteht aus drei Schichten:

#### Schicht 1: Wetterdaten (Open-Meteo JSON)

Jede JSON-Datei enthält reale Stundenwerte für einen Tag:

```json
{
  "hourly": {
    "time":        ["2025-01-28T00:00", "2025-01-28T01:00", ...],
    "temperature_2m": [8.4, 7.1, ...],
    "rain":        [0.0, 0.0, ..., 1.8, 1.6, ...],
    "snowfall":    [0.0, 0.0, ...],
    "visibility":  [51320.0, 37660.0, ..., 6000.0, ...]
  }
}
```

Die 11 Szenarien umfassen z.B.:
- Freiberg Regen (Januar), Freiberg Schneefall, Freiberg Schauer  
- Jakarta Starkregen (tropisch)  
- Alaska Schnee und Regen  
- Brocken Dichtebel (Sichtweite < 50 m)

#### Schicht 2: Empirisches Reibungsmodell (`WeatherManager`)

Beim Laden der JSON-Datei berechnet `WeatherManager._create_sumo_weather_sequence()`  
für jede der 24 Stunden SUMO-Fahrzeugparameter:

```
Niederschlagstyp bestimmen:
  rain > 0  AND snow > 0  → "mixed"
  snow > 0               → "snow"
  rain > 0               → "rain"
  visibility < 1000 m    → "fog"
  sonst                  → "none"

Reibungskoeffizient:
  μ = μ_trocken (0.8)
  bei Regen:  μ -= 0.1 × √(intensität_mm)
  bei Schnee: μ -= 0.2 × √(intensität_cm×10)
  μ = max(0.1, μ × Reifeneffizienz (0.9))

Sichtfaktor:
  fog_factor = min(1.0, sichtweite_m / 1000)

Geschwindigkeitsfaktor:
  speed_factor = min(μ, fog_factor)
  → 0.1 bei dichtestem Schnee/Nebel, 1.0 bei trockenem Wetter

SUMO-Parameter:
  maxSpeed  = 13.89 m/s × speed_factor
  accel     = 2.6 m/s²  × speed_factor
  decel     = 4.5 m/s²  × speed_factor
  tau       = 1.6 s × (1 + (1 − vis_ratio))   → bis 3.2 s bei Nebel
  minGap    = 2.0 m × (1 + (1 − vis_ratio))   → bis 4.0 m bei Nebel
```

#### Schicht 3: Anwendung in der Simulation

In jedem Simulationsschritt prüft `environment.step()` ob sich die Stunde geändert hat:

```python
# environment.py – step()
hour = int((wrapped_time // 3600) % 24)
if hour != self._last_weather_hour:
    w = self.weather_manager.get_weather_for_time(wrapped_time)
    self.weather_manager.apply_weather_params(w)   # alle Fahrzeuge aktualisieren
```

`apply_weather_params()` ruft `traci.vehicle.setMaxSpeed()`, `setAccel()`,  
`setDecel()`, `setTau()`, `setMinGap()` für **jedes** Fahrzeug in der Simulation auf.  
Neu einfahrende Fahrzeuge bekommen die Parameter sofort beim Einfahren gesetzt.

#### Was der Agent beobachtet

Das Wetter fließt in 4 Beobachtungsfelder des Agenten ein:

| Feld | Dimension | Inhalt |
|------|-----------|--------|
| `precip_type` | `(5,)` | One-Hot: clear / rain / snow / mixed / fog |
| `precip_intensity` | `(1,)` | Summe aus rain + showers + snowfall×10 + snow_depth×100 |
| `visibility` | `(1,)` | Sichtweite in km |
| `friction_factor` | `(1,)` | `speed_factor` (0.1–1.0) |

Der Agent kann damit lernen, bei schlechter Sicht (längere Bremsweg) oder  
Glätte (niedrigerer Durchfluss) die Grünphasen anders zu verteilen.

---

### Zusammenfassung: Dateifluss beim Training

```
SUMO-Start (training.py)
  │
  ├─► oneJunctionLearn.sumocfg
  │     ├─► oneJunction.net.xml        (Netz: Kanten, Spuren, Kreuzung J13)
  │     ├─► oneJunction.add.xml        (Zusatzdatei: aktuell leer)
  │     └─► routeFiles/
  │           ├─ flows.rou.xml         (24h Flows: stündl. Fahrzeugströme)
  │           └─ baseroutes.rou.xml    (Einzelfahrzeuge: Basisbelastung)
  │
  ├─► TraCI-Kontrolle durch environment.py
  │     ├─ traci.trafficlight.setPhase()   (Agent wählt Grünphase)
  │     ├─ traci.vehicle.setMaxSpeed()     (Wetter wird angewendet)
  │     └─ traci.simulationStep()          (1 Sekunde simuliert)
  │
  └─► WeatherManager (pro Episode)
        ├─ zufällige JSON aus weatherFiles/ laden
        ├─ 24h Reibungsmodell berechnen
        └─ stündlich alle Fahrzeuge updaten
```

---

## Running the Project

All commands assume you are in the **project root** (the folder that contains `env/` and `oneJunction/`).

### Training

```bash
# Single environment, 5 million steps (default)
python env/training.py

# Multiple parallel environments (SubprocVecEnv)
python env/training.py --num-envs 4

# Full option reference
python env/training.py --help
```

Key CLI arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--sumocfg` | `oneJunction/oneJunctionLearn.sumocfg` | SUMO config file |
| `--weather-dir` | `oneJunction/weatherFiles` | Folder with JSON weather files |
| `--num-envs` | `1` | Number of parallel SUMO instances |
| `--timesteps` | `5_000_000` | Total training steps |
| `--seed` | `None` | Global random seed |
| `--log-dir` | `./logs` | Where to save checkpoints and logs |
| `--tensorboard` | `./logs/tb_logs` | TensorBoard log directory |
| `--gui` | off | Launch `sumo-gui` instead of headless `sumo` |

### Resuming from Checkpoint

Training resumes automatically from the latest checkpoint found in  
`logs/checkpoints/ppo_ckpt_*_steps.zip`.  
Simply re-run the same command — the script detects existing checkpoints:

```bash
python env/training.py --num-envs 4 --timesteps 10_000_000
```

### Evaluation

```bash
python env/evaluate_trained_agent.py \
    --model logs/best_model/best_model.zip \
    --weather_file oneJunction/weatherFiles/freibergRain.json \
    --gui
```

| Flag | Description |
|------|-------------|
| `--model` | Path to the trained `.zip` model (required) |
| `--weather_file` | Specific weather JSON (optional; random if omitted) |
| `--gui` | Open SUMO GUI for visual inspection |

### Hyperparameter Tuning (Optuna)

```bash
python env/hyperparam_tuning.py
```

Results are saved to `optuna_study.pkl` and can be resumed automatically.  
TensorBoard logs go to `logs/optuna/trial_<N>/`.

### TensorBoard

```bash
tensorboard --logdir logs/tb_logs
# Then open http://localhost:6006 in your browser
```

---

## Project Structure

```
SumoRLAgent/
│
├── env/                            # Core RL + simulation code
│   ├── environment.py              # Main Gymnasium environment (SumoEnvironment)
│   ├── training.py                 # Entry point: builds VecEnv, trains MaskablePPO
│   ├── evaluate_trained_agent.py   # Loads model, runs evaluation loop
│   ├── evaluation_environment.py   # Lightweight eval-only environment
│   ├── reward.py                   # RewardCalculator + VehicleStateTracker
│   ├── observer.py                 # SumoObserver – builds observation dicts
│   ├── trafficlight.py             # TrafficLightManager – phase FSM
│   ├── curriculum.py               # CurriculumManager – 4-level progression
│   ├── weather.py                  # WeatherManager – JSON → SUMO vehicle params
│   ├── maskwrapper.py              # MaskedSumoEnv – action-masking wrapper
│   ├── callbacks.py                # CurriculumTensorboardCallback
│   ├── performance_tracker.py      # HDF5-backed per-level metrics store
│   ├── debug_logger.py             # CSV observation logger
│   ├── hyperparam_tuning.py        # Optuna hyperparameter search
│   ├── utils.py                    # Port helpers, path utilities
│   └── hyperparams.json / optunaparams.json
│
├── oneJunction/                    # SUMO scenario files
│   ├── oneJunction.net.xml         # Road network
│   ├── oneJunctionLearn.sumocfg    # Config used during training
│   ├── oneJunctionWeather.sumocfg  # Config used during evaluation
│   ├── routeFiles/                 # Vehicle demand definitions
│   │   ├── flows.rou.xml
│   │   ├── merged.rou.xml
│   │   └── …
│   ├── trafficlightFiles/
│   │   └── staticTrafficlight.tll.xml
│   └── weatherFiles/               # Real-world weather JSON profiles
│       ├── freibergRain.json
│       ├── AlaskaSnow01_20.json
│       ├── JakartaRain.json
│       └── …
│
├── logs/                           # Auto-generated during training
│   ├── training.log
│   ├── episode_log.csv
│   ├── curriculum_state.json
│   ├── checkpoints/                # ppo_ckpt_<step>_steps.zip
│   ├── best_model/                 # best_model.zip (saved by EvalCallback)
│   ├── env_<N>/                    # Per-env HDF5 performance data
│   ├── eval_env/
│   ├── debug_obs/                  # Per-step CSV observation logs
│   ├── episode_seeds/              # Reproducibility – seed + weather per episode
│   ├── tb_logs/                    # TensorBoard events
│   └── optuna/                     # Optuna trial TensorBoard logs
│
├── scripts/                        # Legacy / utility scripts
│   ├── trainAgent.py               # Older training environment (reference only)
│   ├── evaluateAgent.py
│   ├── StartSumoVanilla.py
│   ├── weatherSeq.py               # Weather sequence builder
│   ├── weatherSumo.py
│   └── plotScripts/                # Post-hoc plotting tools
│
├── optuna_study.pkl                # Persisted Optuna study
└── README.md
```

---

## Architecture Overview

### Observation Space

The agent receives a `Dict` observation with 13 keys:

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `phase_one_hot` | `(4,)` | float32 | Current green phase (one-hot) |
| `phase_timer` | `(1,)` | float32 | Seconds since phase start (0–120) |
| `can_switch_phase` | scalar | int | 1 if min-green elapsed |
| `queue_length` | `(4,)` | float32 | Halting vehicles per movement group |
| `avg_wait_time` | `(4,)` | float32 | Mean lane wait time per group (s) |
| `eta` | `(4,)` | float32 | Min time-to-junction per group (s) |
| `vehicle_count` | `(4,)` | int32 | Total vehicles per group |
| `hour_sin` | `(1,)` | float32 | Sine encoding of simulation hour |
| `hour_cos` | `(1,)` | float32 | Cosine encoding of simulation hour |
| `precip_type` | `(5,)` | float32 | One-hot: clear/rain/snow/mixed/fog |
| `precip_intensity` | `(1,)` | float32 | Combined precipitation intensity |
| `visibility` | `(1,)` | float32 | Visibility in km |
| `friction_factor` | `(1,)` | float32 | Road friction (0–1) |

The 4 movement groups correspond to:

| Group | Lanes | Direction |
|-------|-------|-----------|
| 0 | `-E8_0`, `-E8_1`, `-E6_0`, `-E6_1` | N–S straight |
| 1 | `-E6_2`, `-E8_2` | N–S left-turn |
| 2 | `-E7_0`, `-E7_1`, `-E9_0`, `-E9_1` | E–W straight |
| 3 | `-E9_2`, `-E7_2` | E–W left-turn |

### Action Space

`Discrete(4)` — choose one of 4 green phases.  
The `TrafficLightManager` enforces:

- Minimum green time: **5 s** (action to switch same phase is ignored)
- Yellow inter-green: **3 s**
- Red-yellow inter-green: **1 s**
- Maximum green time: **112 s** (violation terminates episode in training)

`MaskablePPO` + `ActionMasker` mask out unavailable actions (e.g. phase-hold during yellow transition).

### Reward Function

```
r = γ_wait · (−Σ wait_i)          # absolute wait penalty
  + γ_wait · Δ(Σ wait_i)          # wait-reduction shaping
  − λ_queue · Σ queue_i           # queue-length penalty
  − λ_fair  · std(queue_i)        # fairness penalty (balance across groups)
  − β_tele  · n_teleports         # teleport penalty
  − γ_viol  · n_violations        # phase-timeout violation penalty
```

Weights scale with curriculum level:

| Level | γ_wait | λ_queue | λ_fair | β_tele | γ_viol |
|-------|--------|---------|--------|--------|--------|
| 0 | 1.0 | 1.0 | 1.0 | 1.0 | 10 |
| 1 | 2.0 | 1.5 | 1.5 | 1.5 | 20 |
| 2 | 3.0 | 2.0 | 2.0 | 2.0 | 50 |
| 3 | 4.0 | 2.5 | 2.5 | 2.5 | 50 |

### Curriculum Learning

Four progressive levels stored in `logs/curriculum_state.json`:

| Level | Episode range | Duration | Peak hours focused |
|-------|--------------|----------|--------------------|
| 0 | 10–20 episodes | 2–4 h | 06:00–12:00, 16:00–19:00 |
| 1 | 15–30 episodes | 2–4 h | same |
| 2 | 20–50 episodes | 4–6 h | same |
| 3 | 30–80 episodes | 6–8 h | same |

Level-up criteria (all must be met simultaneously after `min_episodes`):

- Normalised episode reward > `min_reward`
- Mean waiting time < `max_wait`
- Positive reward slope over last 100 episodes
- Violation rate < `max_violations`
- Teleport rate < 0.1

### Weather Model

Each weather JSON file (Open-Meteo format, hourly) is converted to SUMO vehicle parameters via an empirical friction model:

```
friction = max(0.1, μ_dry · (1 − k · √intensity) · efficiency)
fog_factor = min(1, visibility / 1000 m)
speed_factor = min(friction, fog_factor)
```

Applied parameters per vehicle:
- `accel`, `decel` — scaled by speed_factor
- `maxSpeed` — `50 km/h × speed_factor`
- `tau` (reaction time) — increased linearly with reduced visibility
- `minGap` — increased linearly with reduced visibility

---

## Configuration Reference

The `config` dict (defined in `env/training.py`) wires the SUMO network to the agent:

```python
config = {
    # SUMO traffic light ID as it appears in the .net.xml
    "traffic_light_id": "J13",

    # Path to the .sumocfg file
    "sumocfg_path": "oneJunction/oneJunctionLearn.sumocfg",

    # Maps agent phase index → SUMO phase index (green phases)
    "green_phase_map":    {0: 0, 1: 3, 2: 6, 3: 9},

    # Maps agent phase index → SUMO yellow phase index
    "yellow_phase_map":   {0: 1, 1: 4, 2: 7, 3: 10},

    # Maps agent phase index → SUMO red-yellow phase index
    "redyellow_phase_map":{0: 11, 1: 2, 2: 5, 3: 8},

    # Maps group index → list of lane IDs that get green in that phase
    "green_lane_map": {
        0: ["-E8_0","-E8_1","-E6_0","-E6_1"],
        1: ["-E6_2","-E8_2"],
        2: ["-E7_0","-E7_1","-E9_0","-E9_1"],
        3: ["-E9_2","-E7_2"],
    },

    # All incoming lanes that the observer monitors
    "observation_lanes": [
        "-E8_0","-E8_1","-E8_2",
        "-E6_0","-E6_1","-E6_2",
        "-E7_0","-E7_1","-E7_2",
        "-E9_0","-E9_1","-E9_2",
    ],
}
```

If you adapt this to a different SUMO network, update every key to match the new network's phase indices and lane IDs.

---

## Logs and Outputs

| Path | Contents |
|------|----------|
| `logs/training.log` | Timestamped Python logging output |
| `logs/episode_log.csv` | Per-episode: avg_wait, max_wait, violations, teleports, level |
| `logs/episode_seeds/episode_<N>.txt` | Seed, weather file and start hour for reproduction |
| `logs/checkpoints/ppo_ckpt_*_steps.zip` | Model snapshots every 5 000 steps |
| `logs/best_model/best_model.zip` | Best model from evaluation |
| `logs/env_<N>/performance_env<N>.h5` | HDF5: per-level rewards, waits, throughput, violations |
| `logs/debug_obs/observation_run_<N>.csv` | Raw per-step observation values (up to 1 M rows/file) |
| `logs/tb_logs/` | TensorBoard event files |
| `optuna_study.pkl` | Serialised Optuna study (joblib) |

---

## Troubleshooting

### `SUMO_HOME` not set / `FileNotFoundError` on sumo binary

```
KeyError: 'SUMO_HOME'
```

The environment variable is not set in the current shell.  
→ Follow [Setting up SUMO\_HOME](#setting-up-sumo_home).

---

### `traci.exceptions.FatalTraCIError: connection closed`

SUMO exited unexpectedly (e.g. wrong `.sumocfg` path, port conflict).  
Check:
1. The path passed via `--sumocfg` is correct and the file exists.
2. No other SUMO process is using the same TraCI port (default 8813).  
   Use `--num-envs 1` to reduce concurrent connections while debugging.

---

### `OSError: [Errno 98] Address already in use` (Linux)

A previous SUMO process is still holding the port:

```bash
fuser -k 8813/tcp   # replace 8813 with the actual port
```

The training script attempts automatic port reassignment via `find_free_port()`.

---

### `VecCheckNan` raises NaN warning during training

Common causes:
- `VecNormalize` with an extremely large or unbounded reward.  
  → Reduce `clip_reward` or adjust reward weights.
- Weather observation returns `inf` visibility.  
  → The observation space allows `np.inf` for `visibility`; set a finite upper bound if needed.

---

### CUDA / GPU not used

```
CUDA available: False
```

Verify your PyTorch installation matches your CUDA driver:

```bash
python -c "import torch; print(torch.version.cuda)"
nvidia-smi
```

Re-install PyTorch with the correct CUDA version from <https://pytorch.org/get-started/locally/>.  
CPU training works but is significantly slower.

---

### `KeyError: 4` in `CurriculumManager.get_curriculum_config`

The curriculum level exceeded the defined maximum (3).  
This bug has been fixed — `level` is now capped at `max(level_config.keys())`.  
Delete `logs/curriculum_state.json` if you started training with the old version and the saved level is > 3.

---
