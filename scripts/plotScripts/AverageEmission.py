import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_tripinfo(xml_path):
    """
    Liest tripinfo.xml ein und gibt DataFrame mit depart-Zeiten zurück.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for trip in root.findall("tripinfo"):
        depart = float(trip.get("depart"))
        data.append({"depart": depart})
    return pd.DataFrame(data)


def parse_emissions(xml_path, pollutant="CO2"):
    """
    Liest eine Emissions-XML-Datei ein und gibt DataFrame mit Zeit und Summe des angegebenen pollutants pro timestep.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data = []
    for ts in root.findall("timestep"):
        time = float(ts.get("time"))
        total = 0.0
        for veh in ts.findall("vehicle"):
            val = float(veh.get(pollutant, 0.0))
            total += val
        data.append({"time": time, pollutant: total})
    return pd.DataFrame(data)


def aggregate_hourly(df, time_col, value_col, bins):
    """
    Aggregiert Werte pro Stunde (3600 s) mit der Summe.
    """
    df["hour"] = pd.cut(df[time_col], bins=bins, right=False, labels=np.arange(1, len(bins))).astype(int)
    agg = df.groupby("hour")[value_col].sum().reset_index()
    return agg


def get_traffic_flow(df, bins):
    """
    Berechnet den Traffic Flow (Anzahl Trips) pro Stunde.
    """
    df["hour"] = pd.cut(df["depart"], bins=bins, right=False, labels=np.arange(1, len(bins))).astype(int)
    flow = df.groupby("hour").size().reset_index(name="trip_count")
    return flow


def SaveAverageEmissionPlot(file_name):
    # Pfade zu den XML-Dateien
    tripinfo_xml = r"outputs/tripinfo.xml"
    emiss_w = r"outputs/emissionsWeather.xml"
    emiss_normal = r"outputs/emissions.xml"
    emiss_rl = r"outputs/emissionsRL.xml"

    # Daten einlesen
    df_trips = parse_tripinfo(tripinfo_xml)
    df_em_w = parse_emissions(emiss_w, pollutant="CO2")
    df_em_rl = parse_emissions(emiss_rl, pollutant="CO2")
    df_em_normal = parse_emissions(emiss_normal, pollutant="CO2")

    # Binning: 24 Stunden, 1-Stunden-Intervalle
    simulation_duration = 86400
    time_interval = 3600
    bins = np.arange(0, simulation_duration + time_interval, time_interval)

    # Traffic Flow und Emissionen pro Stunde
    flow = get_traffic_flow(df_trips, bins)
    agg_em_w = aggregate_hourly(df_em_w, "time", "CO2", bins)
    agg_em_rl = aggregate_hourly(df_em_rl, "time", "CO2", bins)
    agg_em_normal = aggregate_hourly(df_em_normal, "time", "CO2", bins)

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

    # Zweite y-Achse: Emissionen
    ax2 = ax1.twinx()
    ax2.set_ylabel("CO2 Emissions (g)", color="black")
    line_w, = ax2.plot(agg_em_w["hour"], agg_em_w["CO2"], marker='o', color="red", label="Emissions Weather")
    line_rl, = ax2.plot(agg_em_rl["hour"], agg_em_rl["CO2"], marker='s', color="green", label="Emissions RL")
    line_normal, = ax2.plot(agg_em_normal["hour"], agg_em_normal["CO2"], marker='^', color="orange", label="Emissions Normal")
    ax2.tick_params(axis="y", labelcolor="black")

    # Legende zusammenführen
    lines = [bars, line_w,line_rl, line_normal]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.title("Hourly Traffic Flow and CO2 Emissions Comparison")
    plt.tight_layout()
    plt.show()
    # Speichern
    save_path = f"outputs/{file_name}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

if __name__ == '__main__':
    SaveAverageEmissionPlot("AverageEmissions_Comparison_Final")