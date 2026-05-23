import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_tripinfo_for_duration(xml_path):
    """Liest tripinfo.xml ein und gibt DataFrame mit depart und duration zurück."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for trip in root.findall("tripinfo"):
        depart = float(trip.get("depart"))
        duration = float(trip.get("duration"))
        data.append({"depart": depart, "duration": duration})
    return pd.DataFrame(data)

def aggregate_duration(df, bins):
    """Ordnet jedem Trip eine Stunde zu und berechnet den Durchschnitt der Dauer pro Stunde."""
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    agg = df.groupby("hour")["duration"].mean().reset_index()
    agg.rename(columns={"duration": "avg_trip_duration"}, inplace=True)
    return agg

def get_traffic_flow(df, bins):
    """Berechnet den Traffic Flow (Anzahl Trips) pro Stunde aus dem DataFrame."""
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(0, len(bins)-1)).astype(int)
    flow = df.groupby("hour").size().reset_index(name="trip_count")
    return flow

def SaveTripDurationPlot(file_name):
    # Pfade zu den drei tripinfo XML-Dateien
    normal_xml = r"outputs\tripinfo.xml"
    weather_xml = r"outputs\tripinfoWeather.xml"
    rl_xml      = r"outputs\tripinfoRL.xml"

    # Einlesen der Daten
    df_normal = parse_tripinfo_for_duration(normal_xml)
    df_weather = parse_tripinfo_for_duration(weather_xml)
    df_rl      = parse_tripinfo_for_duration(rl_xml)
    
    # Binning: 24 Stunden in 1-Stunden-Intervallen
    simulation_duration = 86400
    time_interval = 3600
    bins = np.arange(0, simulation_duration + time_interval, time_interval)
    
    # Aggregation der Trip Duration pro Stunde
    agg_normal = aggregate_duration(df_normal, bins)
    agg_weather = aggregate_duration(df_weather, bins)
    agg_rl = aggregate_duration(df_rl, bins)
    
    # Traffic Flow aus dem Normal-Run
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
    
    # Zweite y-Achse: Average Trip Duration
    ax2 = ax1.twinx()
    ax2.set_ylabel("Average Trip Duration (s)", color="black")
    line_normal, = ax2.plot(agg_normal["hour"], agg_normal["avg_trip_duration"], marker=None, color="orange", label="Avg Duration (Normal)")
    line_weather, = ax2.plot(agg_weather["hour"], agg_weather["avg_trip_duration"], marker=None, color="red",label="Avg Duration (Weather)")
    line_rl, = ax2.plot(agg_rl["hour"], agg_rl["avg_trip_duration"], marker=None, color="green", label="Avg Duration (RL)")
    ax2.tick_params(axis="y", labelcolor="black")
    
    # Legende außerhalb platzieren
    lines = [bars,line_normal, line_rl, line_weather]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    plt.title("Comparison of Traffic Flow and Average Trip Duration per Hour")
    plt.tight_layout()
    plt.show()
    save_path = "outputs/" + file_name + ".png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    SaveTripDurationPlot("AverageTripDurationPlot_Comparison_Final")
