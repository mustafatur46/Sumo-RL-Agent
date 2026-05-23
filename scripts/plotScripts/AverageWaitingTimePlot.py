import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_tripinfo(xml_path):
    """Liest tripinfo.xml ein und gibt DataFrame mit depart und waiting_time zurück."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for trip in root.findall("tripinfo"):
        depart = float(trip.get("depart"))
        waiting_time = float(trip.get("waitingTime"))
        data.append({"depart": depart, "waiting_time": waiting_time})
    return pd.DataFrame(data)

def aggregate_waiting_time(df, bins):
    """Ordnet jedem Trip eine Stunde zu und berechnet den Durchschnitt der waiting_time pro Stunde."""
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    agg = df.groupby("hour")["waiting_time"].mean().reset_index()
    agg.rename(columns={"waiting_time": "avg_waiting_time"}, inplace=True)
    return agg

def get_traffic_flow(df, bins):
    """Berechnet den Traffic Flow (Anzahl Trips) pro Stunde."""
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    flow = df.groupby("hour").size().reset_index(name="trip_count")
    return flow

def SaveAverageWaitingTimePlot(file_name):
    # Pfade zu den drei tripinfo XML-Dateien
    normal_xml = r"outputs\tripinfo.xml"
    weather_xml = r"outputs\tripinfoWeather.xml"
    rl_xml      = r"outputs\tripinfoRLB.xml"
    
    # Daten einlesen
    df_normal = parse_tripinfo(normal_xml)
    df_weather = parse_tripinfo(weather_xml)
    df_rl = parse_tripinfo(rl_xml)
    
    # Binning: 24 Stunden, 1-Stunden-Intervalle
    simulation_duration = 86400
    time_interval = 3600
    bins = np.arange(0, simulation_duration + time_interval, time_interval)
    
    # Durchschnittliche Waiting Time pro Stunde
    agg_normal = aggregate_waiting_time(df_normal, bins)
    agg_weather = aggregate_waiting_time(df_weather, bins)
    agg_rl = aggregate_waiting_time(df_rl, bins)
    
    # Traffic Flow aus Normal-Run
    flow = get_traffic_flow(df_normal, bins)
    
    # Plot erstellen
    sns.set_theme(style="darkgrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Traffic Flow als Barplot (linke y-Achse)
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
    
    # Zweite y-Achse: Average Waiting Time
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Waiting Time (s)", color="black")
    line_normal, = ax2.plot(agg_normal["hour"], agg_normal["avg_waiting_time"], marker=None, color="orange", label="Avg Waiting (Normal)")
    line_weather, = ax2.plot(agg_weather["hour"], agg_weather["avg_waiting_time"], marker=None, color="red", label="Avg Waiting (Weather)")
    line_rl, = ax2.plot(agg_rl["hour"], agg_rl["avg_waiting_time"], marker=None, color="green",  label="Avg Waiting (RL)")
    ax2.tick_params(axis="y", labelcolor="black")
    
    # Legende außerhalb platzieren
    lines = [bars, line_normal,line_rl, line_weather]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.title("Comparison of Traffic Flow and Average Waiting Time per Hour")
    plt.tight_layout()
    plt.show()
    save_path = "outputs/" + file_name + ".png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    SaveAverageWaitingTimePlot("AverageWaitingTimePlot_Comparison_Final")
