import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_tripinfo(xml_path):
    """
    Liest tripinfo.xml ein und gibt DataFrame mit depart und Trip-Anzahl zurück.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for trip in root.findall("tripinfo"):
        depart = float(trip.get("depart"))
        data.append({"depart": depart})
    return pd.DataFrame(data)


def parse_summary_speed(xml_path):
    """
    Liest summary.xml ein und gibt DataFrame mit time und meanSpeed_kmh zurück.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for step in root.findall("step"):
        time = float(step.get("time"))
        mean_speed = float(step.get("meanSpeed")) * 3.6  # in km/h
        data.append({"time": time, "meanSpeed_kmh": mean_speed})
    return pd.DataFrame(data)


def aggregate_speed(df, time_col, value_col, bins):
    """
    Ordnet jedem Zeitwert eine Stunde zu und berechnet den Durchschnitt der Geschwindigkeit pro Stunde.
    """
    df["hour"] = pd.cut(df[time_col], bins=bins, right=False, labels=np.arange(1, len(bins))).astype(int)
    agg = df.groupby("hour")[value_col].mean().reset_index()
    return agg


def get_traffic_flow(df, bins):
    """
    Berechnet den Traffic Flow (Anzahl Trips) pro Stunde.
    """
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(1, len(bins))).astype(int)
    flow = df.groupby("hour").size().reset_index(name="trip_count")
    return flow


def SaveAverageSpeedPlot(file_name):
    # Pfade zu den XML-Dateien
    tripinfo_xml = r"outputs/tripinfo.xml"
    summary_w_xml = r"outputs/summaryWeather.xml"
    summary_rl_xml = r"outputs/summaryRL.xml"
    summary_normal_xml = r"outputs/summary.xml"
    # Daten einlesen
    df_trips = parse_tripinfo(tripinfo_xml)
    df_speed_w = parse_summary_speed(summary_w_xml)
    df_speed_rl = parse_summary_speed(summary_rl_xml)
    df_speed_normal = parse_summary_speed(summary_normal_xml)

    # Binning: 24 Stunden, 1-Stunden-Intervalle
    simulation_duration = 86400
    time_interval = 3600
    bins = np.arange(0, simulation_duration + time_interval, time_interval)

    # Traffic Flow und Durchschnittsgeschwindigkeit pro Stunde
    flow = get_traffic_flow(df_trips, bins)
    agg_speed_w = aggregate_speed(df_speed_w, "time", "meanSpeed_kmh", bins)
    agg_speed_rl = aggregate_speed(df_speed_rl, "time", "meanSpeed_kmh", bins)
    agg_speed_normal = aggregate_speed(df_speed_normal, "time", "meanSpeed_kmh", bins)

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

    # Zweite y-Achse: Durchschnittsgeschwindigkeit
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Speed (km/h)", color="black")
    line_w, = ax2.plot(agg_speed_w["hour"], agg_speed_w["meanSpeed_kmh"], marker=None, color="red", label="Mean Speed (Weather)")
    line_rl, = ax2.plot(agg_speed_rl["hour"], agg_speed_rl["meanSpeed_kmh"], marker=None, color="green", label="Mean Speed (RL)")
    line_normal, = ax2.plot(agg_speed_normal["hour"], agg_speed_normal["meanSpeed_kmh"], marker=None, color="orange", label="Mean Speed (Normal)")
    ax2.tick_params(axis="y", labelcolor="black")

    # Legende außerhalb platzieren
    lines = [bars, line_w,line_rl, line_normal]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.title("Comparison of Traffic Flow and Mean Speed per Hour")
    plt.tight_layout()
    plt.show()
    # Speichern
    save_path = f"outputs/{file_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    SaveAverageSpeedPlot("AverageSpeedPlot_Comparison_Final")
