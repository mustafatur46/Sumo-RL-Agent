# file: plot_max_waiting_over_steps.py

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from xml.etree.ElementTree import fromstring, parse, ElementTree


def get_waiting_times(filepath: str) -> list:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        start = content.find("<tripinfos")
        if start == -1:
            return []
        xml_content = content[start:]
        root = fromstring(xml_content)
        return [
            float(trip.get("waitingTime", 0.0))
            for trip in root.findall("tripinfo")
        ]
    except Exception as e:
        print(f"Fehler in {filepath}: {e}")
        return []


def get_avg_queue_length(filepath: str) -> float:
    try:
        tree = parse(filepath)
        root = tree.getroot()
        values = []
        for timestep in root.findall("data"):
            lanes = timestep.find("lanes")
            if lanes is not None:
                for lane in lanes.findall("lane"):
                    q = lane.get("queueing_length")
                    if q is not None:
                        values.append(float(q))
        return sum(values) / len(values) if values else 0.0
    except Exception as e:
        print(f"Fehler beim Parsen der Queue-Datei {filepath}: {e}")
        return 0.0


def parse_emissions(xml_path: str, pollutant="CO2") -> pd.DataFrame:
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            content = f.read().lstrip("\ufeff\n\r ")
        root = fromstring(content)
        data = []
        for ts in root.findall("timestep"):
            time = float(ts.get("time"))
            total = 0.0
            for veh in ts.findall("vehicle"):
                val = float(veh.get(pollutant, 0.0))
                total += val
            data.append({"time": time, pollutant: total})
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Fehler beim CO2-Parsing in {xml_path}: {e}")
        return pd.DataFrame()


def get_avg_co2(filepath: str) -> float:
    df = parse_emissions(filepath, pollutant="CO2")
    if not df.empty:
        return df["CO2"].mean()
    return 0.0


def collect_waiting_stats(base_dir: str):
    max_results = []
    avg_results = []
    queue_results = []
    co2_results = []
    for folder in sorted(os.listdir(base_dir)):
        if re.fullmatch(r"\d{1,3}(?:_\d{3})+", folder):
            step = int(folder.replace("_", ""))
            tripinfo_path = os.path.join(base_dir, folder, "tripinfoRL.xml")
            queue_path = os.path.join(base_dir, folder, "queueRL.xml")
            emission_path = os.path.join(base_dir, folder, "emissionsRL.xml")
            if os.path.isfile(tripinfo_path):
                wait_times = get_waiting_times(tripinfo_path)
                if wait_times:
                    max_results.append((step, max(wait_times)))
                    avg_results.append((step, sum(wait_times) / len(wait_times)))
            if os.path.isfile(queue_path):
                avg_queue = get_avg_queue_length(queue_path)
                queue_results.append((step, avg_queue))
            if os.path.isfile(emission_path):
                avg_co2 = get_avg_co2(emission_path)
                if avg_co2 > 0:
                    co2_results.append((step, avg_co2))
    return sorted(max_results), sorted(avg_results), sorted(queue_results), sorted(co2_results)


def plot_waiting_time_trend(results, ylabel, title, filename):
    steps, values = zip(*results)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, linestyle='-', color='purple', linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.1)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{filename}", dpi=300)
    plt.show()


if __name__ == '__main__':
    base_dir = r"C:/Users/Mustafa/Desktop/Scenario1/outputs"
    max_results, avg_results, queue_results, co2_results = collect_waiting_stats(base_dir)

    if max_results:
        plot_waiting_time_trend(
            max_results,
            ylabel="Max Waiting Time (s)",
            title="Max Waiting Time vs Training Steps",
            filename="max_waiting_vs_steps.png"
        )

    if avg_results:
        plot_waiting_time_trend(
            avg_results,
            ylabel="Avg Waiting Time (s)",
            title="Average Waiting Time vs Training Steps",
            filename="avg_waiting_vs_steps.png"
        )

    if queue_results:
        plot_waiting_time_trend(
            queue_results,
            ylabel="Avg Queue Length (m)",
            title="Average Queue Length vs Training Steps",
            filename="avg_queue_vs_steps.png"
        )

    if co2_results:
        plot_waiting_time_trend(
            co2_results,
            ylabel="Avg CO2 Emission (mg)",
            title="Average CO2 Emissions vs Training Steps",
            filename="avg_co2_vs_steps.png"
        )
    else:
        print("Keine validen CO2-Daten gefunden.")