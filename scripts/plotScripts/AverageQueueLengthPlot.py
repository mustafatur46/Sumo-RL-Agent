import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def parse_queue(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for timestep in root.findall("data"):
        time_val = float(timestep.get("timestep"))
        lanes = timestep.find("lanes")
        if lanes is not None and len(lanes.findall("lane")) > 0:
            for lane in lanes.findall("lane"):
                q_len = float(lane.get("queueing_length", 0))
                data.append({"time": time_val, "queue_length": q_len})
        else:
            # Kein Fahrzeug: trotzdem eintragen mit 0.0
            data.append({"time": time_val, "queue_length": 0.0})
    return pd.DataFrame(data)


def aggregate_queue(df):
    """Berechnet pro Stunde den Durchschnitt der queue_length."""
    df["hour"] = (df["time"] / 3600).astype(int)
    agg = df.groupby("hour")["queue_length"].mean().reset_index()
    agg.rename(columns={"queue_length": "avg_queue_length"}, inplace=True)
    return agg

def get_traffic_flow_from_tripinfo(xml_path, time_interval=3600, simulation_duration=86400):
    """Berechnet den Traffic Flow (Anzahl Trips) pro Stunde aus der tripinfo.xml (Normal)"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for trip in root.findall("tripinfo"):
        depart = float(trip.get("depart"))
        data.append({"depart": depart})
    df = pd.DataFrame(data)
    bins = np.arange(0, simulation_duration + time_interval, time_interval)
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    flow = df.groupby("hour").size().reset_index(name="trip_count")
    return flow

def SaveQueueLengthPlot(file_name):
    # Pfade zu den Queue XML-Dateien für Normal, Wetter, RL
    queue_normal = r"outputs\queue.xml"
    queue_weather = r"outputs\queueWeather.xml"
    queue_rl      = r"outputs\queueRL.xml"
    # Pfad zur tripinfo Normal, für Traffic Flow
    tripinfo_normal = r"outputs\tripinfo.xml"
    
    # Daten einlesen und aggregieren (Queue)
    df_q_normal = parse_queue(queue_normal)
    df_q_weather = parse_queue(queue_weather)
    df_q_rl = parse_queue(queue_rl)
    
    agg_normal = aggregate_queue(df_q_normal)
    agg_weather = aggregate_queue(df_q_weather)
    agg_rl = aggregate_queue(df_q_rl)
    
    # Traffic Flow aus tripinfo Normal
    flow = get_traffic_flow_from_tripinfo(tripinfo_normal)
    
    # Plot erstellen
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(12, 6), constrained_layout=True)
    
    # Linke y-Achse: Traffic Flow als Barplot
    color_flow = "#0e4eab"
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Traffic Flow (Trips)", color=color_flow)
    bars = ax1.bar(flow["hour"], flow["trip_count"], color=color_flow, alpha=0.4, label="Traffic Flow")
    ax1.tick_params(axis="y", labelcolor=color_flow)
    ax1.set_xticks(range(1, 25))
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    upper_limit = flow["trip_count"].max() * 1.1
    lower_limit = flow["trip_count"].min() * 0.95
    ax1.set_ylim(lower_limit, upper_limit)
    
    # Rechte y-Achse: Average Queue Length als Linienplot (für alle drei Runs)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Queue Length (m)", color="black")
    line_normal, = ax2.plot(agg_normal["hour"], agg_normal["avg_queue_length"], marker=None, color="orange", label="Avg Queue (Normal)")
    line_weather, = ax2.plot(agg_weather["hour"], agg_weather["avg_queue_length"], marker=None, color="red", label="Avg Queue (Weather)")
    line_rl, = ax2.plot(agg_rl["hour"], agg_rl["avg_queue_length"], marker=None, color="green", label="Avg Queue (RL)")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(6))
    
    # Legende außerhalb platzieren
    lines =[bars, line_rl, line_normal, line_weather]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.title("Comparison of Traffic Flow and Average Queue Length per Hour")
    plt.tight_layout()
    plt.show()
    save_path = "outputs/" + file_name + ".png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    SaveQueueLengthPlot("AverageQueueLengthPlot_Comparison_Final")
