import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_tripinfo(xml_path):
    """Liest eine tripinfo.xml ein und gibt einen DataFrame mit depart und waiting_time zurück."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for trip in root.findall("tripinfo"):
        depart = float(trip.get("depart"))
        waiting_time = float(trip.get("waitingTime"))
        data.append({"depart": depart, "waiting_time": waiting_time})
    return pd.DataFrame(data)

def aggregate_waiting_time(df, bins):
    """Ordnet jedem Trip eine Stunde zu und berechnet Median und 90th Percentile der waiting_time pro Stunde."""
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    stats = df.groupby("hour")["waiting_time"].agg(
        median="median",
        p90=lambda x: np.percentile(x, 90)
    ).reset_index()
    return stats

def get_traffic_flow(df, bins):
    """Berechnet den Traffic Flow (Anzahl Trips) pro Stunde aus dem DataFrame."""
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    flow = df.groupby("hour").size().reset_index(name="trip_count")
    return flow

def SaveMeanWaitingTimePlot(file_name):
    # Pfade zu den drei tripinfo XML-Dateien (für Normal, Wetter, RL)
    normal_xml = r"outputs\tripinfo.xml"
    weather_xml = r"outputs\tripinfoWeather.xml"
    rl_xml      = r"outputs\tripinfoRL.xml"
    
    # Einlesen der Daten
    df_normal = parse_tripinfo(normal_xml)
    df_weather = parse_tripinfo(weather_xml)
    df_rl      = parse_tripinfo(rl_xml)
    
    # Binning: 24 Stunden in 1-Stunden-Intervallen
    simulation_duration = 86400
    time_interval = 3600
    bins = np.arange(0, simulation_duration + time_interval, time_interval)
    
    # Aggregation der Waiting Times
    stats_normal = aggregate_waiting_time(df_normal, bins)
    stats_weather = aggregate_waiting_time(df_weather, bins)
    stats_rl = aggregate_waiting_time(df_rl, bins)
    
    # Traffic Flow aus dem Normal-Run (Anzahl Trips pro Stunde)
    flow = get_traffic_flow(df_normal, bins)
    
    # Plot erstellen
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Barplot: Traffic Flow (linke y-Achse)
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
    
    # Zweite y-Achse: Median & 90th Percentile der Waiting Time
    ax2 = ax1.twinx()
    ax2.set_ylabel("Waiting Time (s)", color="black")
    # line_median, = ax2.plot(stats_normal["hour"], stats_normal["median"], marker=None, color="orange", linestyle="--", label="Median Waiting Time")
    line_p90, = ax2.plot(stats_normal["hour"], stats_normal["p90"], marker=None, color="orange", linestyle=None, label="90th Percentile Waiting Time")
    # Plot auch die anderen Runs als Linien:
    # median_weather = ax2.plot(stats_weather["hour"], stats_weather["median"], marker=None, color="red", linestyle="--", label="Median Waiting Time (Weather)")[0]
    # median_RL    = ax2.plot(stats_rl["hour"], stats_rl["median"], marker=None, color="green", linestyle="--", label="Median Waiting Time (RL)")[0]
    p90_Weather  = ax2.plot(stats_weather["hour"], stats_weather["p90"], marker=None, color="red", linestyle=None, label="90th Percentile Waiting Time (Weather)")[0]
    p90_RL       = ax2.plot(stats_rl["hour"], stats_rl["p90"], marker=None, color="green", linestyle=None, label="90th Percentile Waiting Time (RL)")[0]
    ax2.tick_params(axis="y", labelcolor="black")

    # Legende außerhalb platzieren
    lines = [bars, line_p90,  p90_Weather, p90_RL]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.title("Comparison of Waiting Times (Median/90th Percentile) and Traffic Flow per Hour")
    plt.tight_layout()
    plt.show()
    save_path = "outputs/" + file_name + ".png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    SaveMeanWaitingTimePlot("MeanWaitingTimePlot_Comparison_Final")
