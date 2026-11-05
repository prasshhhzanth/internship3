"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Scenario: last-mile delivery performance for an e-commerce network in Kolkata.

Input : data/delivery_clean.csv  (output of the Week 2 preprocessing pipeline)
Run   : python week3_eda.py
Output: figures/*.png and outputs/eda_results.json
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_PATHS = ["data/delivery_clean.csv", "../week2/data/delivery_clean.csv"]
PROMISED_MIN = 40
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Palette: one blue for single series, fixed categorical order, sequential blues
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})
TRAFFIC_ORDER = ["Low", "Medium", "High", "Jam"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=170)
    plt.close(fig)


# =====================================================================
# 1. LOAD AND DEFINE THE DATASET
# =====================================================================
def load():
    for path in CLEAN_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Order_Date"])
            print(f"Loaded {path}: {df.shape}")
            return df
    raise FileNotFoundError("Run the Week 2 pipeline first to create delivery_clean.csv")


# Cost model (scenario assumption): fixed handling + vehicle running cost per km
# + partner pay per minute. Rates in Indian rupees.
COST_PER_KM = {"motorcycle": 3.0, "scooter": 2.6, "electric_scooter": 1.2, "bicycle": 0.5}
FIXED_COST, WAGE_PER_MIN = 15.0, 1.5


def add_metrics(df):
    df = df.copy()
    df["is_late"] = (df["time_taken_min"] > PROMISED_MIN).astype(int)
    df["delivery_cost_rs"] = (FIXED_COST
                              + df["Type_of_vehicle"].map(COST_PER_KM) * df["distance_km"]
                              + WAGE_PER_MIN * df["time_taken_min"]).round(2)
    df["traffic_level"] = df["Road_traffic_density"].map({t: i for i, t in enumerate(TRAFFIC_ORDER)})
    df["festival_flag"] = (df["Festival"] == "Yes").astype(int)
    df["Road_traffic_density"] = pd.Categorical(df["Road_traffic_density"], TRAFFIC_ORDER, ordered=True)
    df["shipments"] = 1
    return df


# =====================================================================
# 2. DESCRIPTIVE STATISTICS
# =====================================================================
def descriptive(df):
    cols = ["time_taken_min", "distance_km", "delivery_cost_rs", "pickup_wait_min",
            "Delivery_person_Ratings", "Delivery_person_Age"]
    desc = df[cols].describe().T
    desc["median"] = df[cols].median()
    desc["mode"] = df[cols].mode().iloc[0]
    desc["skew"] = df[cols].skew()
    desc["kurtosis"] = df[cols].kurt()
    desc = desc[["mean", "median", "mode", "std", "min", "max", "skew", "kurtosis"]].round(2)
    print("\nCentral tendency and spread:\n", desc)

    kpis = {
        "orders": int(len(df)),
        "on_time_rate_pct": round(100 * (1 - df["is_late"].mean()), 1),
        "avg_delivery_min": round(df["time_taken_min"].mean(), 1),
        "avg_cost_rs": round(df["delivery_cost_rs"].mean(), 1),
        "total_cost_rs": round(df["delivery_cost_rs"].sum(), 0),
        "avg_distance_km": round(df["distance_km"].mean(), 2),
        "daily_orders_avg": round(df.groupby("Order_Date").size().mean(), 0),
    }
    print("\nKPIs:", kpis)
    return desc, kpis


def group_tables(df):
    def summ(by):
        g = df.groupby(by, observed=True).agg(
            orders=("shipments", "sum"),
            avg_time=("time_taken_min", "mean"),
            late_rate=("is_late", "mean"),
            avg_cost=("delivery_cost_rs", "mean"))
        g["late_rate"] *= 100
        return g.round(1)

    tables = {
        "traffic": summ("Road_traffic_density"),
        "weather": summ("Weatherconditions").sort_values("avg_time"),
        "vehicle": summ("Type_of_vehicle"),
        "multiple_deliveries": summ("multiple_deliveries"),
        "festival": summ("Festival"),
        "city": summ("City"),
        "peak": summ("is_peak_hour"),
    }
    for k, t in tables.items():
        print(f"\nBy {k}:\n{t}")
    return tables


def correlations(df):
    cols = {"time_taken_min": "Delivery time", "distance_km": "Distance",
            "traffic_level": "Traffic level", "multiple_deliveries": "Multiple deliveries",
            "Delivery_person_Ratings": "Partner rating", "Delivery_person_Age": "Partner age",
            "pickup_wait_min": "Pickup wait", "festival_flag": "Festival",
            "delivery_cost_rs": "Delivery cost"}
    corr = df[list(cols)].rename(columns=cols).corr(method="pearson").round(2)
    spearman = df[list(cols)].rename(columns=cols).corr(method="spearman")["Delivery time"].round(2)
    print("\nCorrelation with delivery time (Pearson):\n",
          corr["Delivery time"].sort_values(ascending=False))
    return corr, spearman


# =====================================================================
# 3. VISUALIZATIONS
# =====================================================================
def fig_distribution(df, kpis):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(df["time_taken_min"], bins=range(10, 56, 2), color=BLUE, edgecolor="white", linewidth=1)
    ax.axvline(df["time_taken_min"].mean(), color=INK, lw=1.5, ls="-")
    ax.axvline(PROMISED_MIN, color=ORANGE, lw=2, ls="--")
    ymax = ax.get_ylim()[1]
    ax.text(df["time_taken_min"].mean() - 0.7, ymax * 0.97, f"mean {kpis['avg_delivery_min']} min",
            ha="right", va="top", color=INK, fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(PROMISED_MIN + 0.5, ymax * 0.92, f"40-min promise\n{100 - kpis['on_time_rate_pct']:.1f}% late",
            ha="left", va="top", color=INK, fontsize=9)
    ax.set_xlabel("Delivery time (minutes)"); ax.set_ylabel("Orders")
    ax.set_title("Distribution of delivery time")
    save(fig, "fig1_delivery_time_distribution")


def fig_traffic_box(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    sns.boxplot(data=df, x="Road_traffic_density", y="time_taken_min", order=TRAFFIC_ORDER,
                color="#86b6ef", width=0.5, linewidth=1, fliersize=2, ax=ax)
    ax.axhline(PROMISED_MIN, color=ORANGE, lw=1.5, ls="--")
    med = df.groupby("Road_traffic_density", observed=True)["time_taken_min"].median()
    for i, t in enumerate(TRAFFIC_ORDER):
        ax.text(i + 0.3, med[t], f"{med[t]:.0f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Road traffic density"); ax.set_ylabel("Delivery time (min)")
    ax.set_title("Delivery time by traffic density")
    save(fig, "fig2_time_by_traffic")


def fig_distance_scatter(df):
    s = df.sample(min(3000, len(df)), random_state=1)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = {"Low": AQUA, "Medium": BLUE, "High": YELLOW, "Jam": ORANGE}
    for t in TRAFFIC_ORDER:
        d = s[s["Road_traffic_density"] == t]
        ax.scatter(d["distance_km"], d["time_taken_min"], s=10, alpha=0.55,
                   color=colors[t], edgecolors="none", label=t)
    m, c = np.polyfit(df["distance_km"], df["time_taken_min"], 1)
    xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 50)
    ax.plot(xs, m * xs + c, color=INK, lw=2)
    ax.text(xs[-1], m * xs[-1] + c + 1.5, f"+{m:.2f} min per km", ha="right", fontsize=9, color=INK)
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Delivery time (min)")
    ax.set_title("Distance vs delivery time, coloured by traffic (3,000-order sample)")
    ax.legend(title="Traffic", frameon=False, fontsize=8, title_fontsize=8, loc="lower right", ncol=4, markerscale=2.5)
    save(fig, "fig3_distance_vs_time")
    return m, c


def fig_corr(corr):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True,
                fmt=".2f", annot_kws={"size": 8}, linewidths=1.5, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Pearson r"}, ax=ax)
    ax.set_title("Correlation matrix of key variables")
    ax.grid(False)
    save(fig, "fig4_correlation_heatmap")


def fig_hourly(df):
    h = df.groupby("order_hour").agg(orders=("shipments", "sum"), late=("is_late", "mean"))
    h = h[h["orders"] >= 50]          # ignore hours with too few orders to be reliable
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axes[0].bar(h.index, h["orders"], color=BLUE, width=0.75)
    axes[0].set_ylabel("Orders"); axes[0].set_title("Order volume and late rate by hour of day")
    axes[1].plot(h.index, 100 * h["late"], color=BLUE, lw=2, marker="o", ms=4)
    axes[1].set_ylabel("Late deliveries (%)"); axes[1].set_xlabel("Hour of order")
    axes[1].set_ylim(0, 45)
    for a in axes:
        for hr in [12, 13, 19, 20, 21]:
            a.axvspan(hr - 0.5, hr + 0.5, color="#eef4fc", lw=0, zorder=0)
    axes[1].text(20, axes[1].get_ylim()[1] * 0.95, "peak hours shaded", ha="center", fontsize=8, color=MUTED)
    save(fig, "fig5_hourly_pattern")
    return h


def fig_heatmap_traffic_weather(df):
    pv = (df.pivot_table(index="Weatherconditions", columns="Road_traffic_density",
                         values="is_late", aggfunc="mean", observed=True) * 100)
    pv = pv.loc[pv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    sns.heatmap(pv, cmap="Blues", annot=True, fmt=".0f", linewidths=1.5, linecolor="white",
                cbar_kws={"label": "% late"}, ax=ax)
    ax.set_xlabel("Road traffic density"); ax.set_ylabel("Weather")
    ax.set_title("Late delivery rate (%) by weather and traffic")
    ax.grid(False)
    save(fig, "fig6_late_rate_weather_traffic")
    return pv.round(1)


def fig_cost(df):
    v = df.groupby("Type_of_vehicle").agg(avg_cost=("delivery_cost_rs", "mean"),
                                          per_km=("delivery_cost_rs", lambda s: s.sum()),
                                          km=("distance_km", "sum"), orders=("shipments", "sum"))
    v["cost_per_km"] = v["per_km"] / v["km"]
    v = v.sort_values("avg_cost")
    d = df.groupby("Road_traffic_density", observed=True)["delivery_cost_rs"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    axes[0].barh(v.index.str.replace("_", " "), v["avg_cost"], color=BLUE, height=0.55)
    for i, val in enumerate(v["avg_cost"]):
        axes[0].text(val + 1, i, f"Rs {val:.0f}", va="center", fontsize=8, color=INK)
    axes[0].set_title("Avg cost per delivery by vehicle"); axes[0].set_xlabel("Rs")
    axes[1].bar(d.index.astype(str), d.values, color=BLUE, width=0.55)
    for i, val in enumerate(d.values):
        axes[1].text(i, val + 1, f"{val:.0f}", ha="center", fontsize=8, color=INK)
    axes[1].set_title("Avg cost per delivery by traffic"); axes[1].set_ylabel("Rs")
    save(fig, "fig7_cost_drivers")
    return v.round(2)


def fig_load(df):
    m = df.groupby("multiple_deliveries")["is_late"].mean() * 100
    f = df.groupby("Festival")["time_taken_min"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(m.index.astype(int).astype(str), m.values, color=BLUE, width=0.55)
    for i, val in enumerate(m.values):
        axes[0].text(i, val + 1, f"{val:.0f}%", ha="center", fontsize=8, color=INK)
    axes[0].set_xlabel("Extra orders on the same trip"); axes[0].set_ylabel("Late deliveries (%)")
    axes[0].set_title("Late rate by trip load")
    axes[1].bar(f.index, f.values, color=BLUE, width=0.45)
    for i, val in enumerate(f.values):
        axes[1].text(i, val + 0.6, f"{val:.1f} min", ha="center", fontsize=8, color=INK)
    axes[1].set_title("Avg delivery time on festival days"); axes[1].set_ylabel("Minutes")
    save(fig, "fig8_load_and_festival")


# =====================================================================
if __name__ == "__main__":
    df = add_metrics(load())
    desc, kpis = descriptive(df)
    tables = group_tables(df)
    corr, spearman = correlations(df)

    fig_distribution(df, kpis)
    fig_traffic_box(df)
    slope, intercept = fig_distance_scatter(df)
    fig_corr(corr)
    hourly = fig_hourly(df)
    heat = fig_heatmap_traffic_weather(df)
    vehicle_cost = fig_cost(df)
    fig_load(df)

    results = {
        "kpis": kpis,
        "descriptive": desc.to_dict(),
        "tables": {k: v.reset_index().astype({v.index.name or "index": str}).to_dict("records")
                   for k, v in tables.items()},
        "corr_with_time": corr["Delivery time"].to_dict(),
        "spearman_with_time": spearman.to_dict(),
        "minutes_per_km": round(slope, 2),
        "late_heatmap": heat.to_dict(),
        "vehicle_cost": vehicle_cost.reset_index().to_dict("records"),
        "hourly": hourly.reset_index().to_dict("records"),
    }
    with open("outputs/eda_results.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\nSaved figures/*.png and outputs/eda_results.json")
