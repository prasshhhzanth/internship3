# Logistics Data Analyst Internship – Week 3

**Advanced Data Analysis and Visualization in Logistics**

`week3_eda.py` analyses the cleaned delivery data from Week 2:

- Adds a late-delivery flag (over the 40-minute promise) and a transport cost model
- Descriptive statistics: mean, median, std, skew, kurtosis, KPIs
- Group summaries by traffic, weather, vehicle, trip load, festival, city and peak hour
- Pearson and Spearman correlations
- Eight charts: delivery-time distribution, time by traffic, distance vs time, correlation heatmap, hourly pattern, weather × traffic late-rate heatmap, cost drivers, trip load and festivals

## Run

Copy `delivery_clean.csv` from Week 2 into `data/` (or keep the `week2` folder next to this one), then:

```bash
pip install -r requirements.txt
python week3_eda.py
```

Outputs: `figures/*.png` and `outputs/eda_results.json`.
