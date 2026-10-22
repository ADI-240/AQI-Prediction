"""IntelliAir OS - Smart City AQI forecasting dashboard (Streamlit).

Run:  streamlit run dashboard.py
Needs: data/city_day.csv. models/multi_horizon_aqi_model.pkl is used if present and loadable;
otherwise the model is trained from the CSV on first start (so deployment never breaks on a
scikit-learn version mismatch).
"""
import math
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="IntelliAir OS | Smart City Environmental Command Center",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent


def find_file(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError("Could not find any of: " + ", ".join(str(c) for c in candidates))


# ---------------------------------------------------------------- compatibility
# Streamlit removed `use_container_width` in favour of width="stretch"; support both.
def plot(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, width="stretch")
    except Exception:
        st.dataframe(df, use_container_width=True)


def sidebar_button(label, **kw):
    try:
        return st.sidebar.button(label, width="stretch", **kw)
    except Exception:
        return st.sidebar.button(label, use_container_width=True, **kw)


# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.top-navbar { background: linear-gradient(135deg,#0f172a 0%,#1e293b 100%); padding: 24px 32px; border-radius: 16px;
  color:#fff; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(15,23,42,.15); border: 1px solid #334155; }
.top-navbar h1 { color:#fff; font-size: 26px; font-weight: 800; letter-spacing:-.5px; margin:0; }
.top-navbar p { color:#94a3b8; font-size: 14px; margin: 6px 0 0 0; }
.live-pulse { display:inline-flex; align-items:center; background: rgba(16,185,129,.15); color:#10b981;
  padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; border: 1px solid rgba(16,185,129,.3); }
.pulse-dot { width:8px; height:8px; background:#10b981; border-radius:50%; margin-right:8px; box-shadow:0 0 8px #10b981; }
.metric-card { background:#fff; border:1px solid #e2e8f0; border-radius:14px; padding:20px;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,.05); transition: transform .2s ease, box-shadow .2s ease; }
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 20px -5px rgba(0,0,0,.08); }
.metric-label { color:#64748b; font-size:12px; text-transform:uppercase; font-weight:700; letter-spacing:.5px; }
.metric-value { color:#0f172a; font-size:28px; font-weight:800; margin:6px 0; font-family:'JetBrains Mono', monospace; }
.advisory-box { border-radius:12px; padding:18px 22px; margin:12px 0; font-size:14px; line-height:1.6; }
.advisory-good   { background:#ecfdf5; border-left:5px solid #10b981; color:#065f46; }
.advisory-mod    { background:#fffbeb; border-left:5px solid #f59e0b; color:#92400e; }
.advisory-poor   { background:#fef2f2; border-left:5px solid #ef4444; color:#991b1b; }
.advisory-severe { background:#faf5ff; border-left:5px solid #a855f7; color:#581c87; }
.culprit-banner { background:#fff1f2; border:1px solid #fecdd3; border-radius:12px; padding:16px 20px;
  color:#9f1239; font-size:14px; margin-top:14px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data + model
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3", "NH3", "NOx"]
FEATURE_COLS = POLLUTANTS + ["Month", "DayOfWeek", "Is_Weekend", "PM25_Lag1", "PM10_Lag1", "PM25_Roll7_Mean", "AQI"]
TARGET_COLS = ["AQI_T1", "AQI_T2", "AQI_T3"]

STATION_COORDS = {
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714}, "Aizawl": {"lat": 23.7271, "lon": 92.7176},
    "Amaravati": {"lat": 16.5131, "lon": 80.5165}, "Amritsar": {"lat": 31.6340, "lon": 74.8723},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946}, "Bhopal": {"lat": 23.2599, "lon": 77.4126},
    "Brajrajnagar": {"lat": 21.8235, "lon": 83.9216}, "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Chennai": {"lat": 13.0827, "lon": 80.2707}, "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
    "Delhi": {"lat": 28.6139, "lon": 77.2090}, "Ernakulam": {"lat": 9.9816, "lon": 76.2999},
    "Gurugram": {"lat": 28.4595, "lon": 77.0266}, "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867}, "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Jorapokhar": {"lat": 23.7022, "lon": 86.4138}, "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639}, "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777}, "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Shillong": {"lat": 25.5788, "lon": 91.8933}, "Talcher": {"lat": 20.9509, "lon": 85.2166},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366}, "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
}


def _train_from_csv(hist):
    """Fallback: train the same multi-horizon model from the CSV (same logic as src/train.py)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    df = hist.dropna(subset=["AQI"]).sort_values(["City", "Date"]).reset_index(drop=True)
    for p in POLLUTANTS:
        df[p] = df.groupby("City")[p].transform(lambda g: g.ffill().bfill())
        df[p] = df[p].fillna(df[p].median())
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    g = df.groupby("City")
    df["PM25_Lag1"], df["PM10_Lag1"] = g["PM2.5"].shift(1), g["PM10"].shift(1)
    df["PM25_Roll7_Mean"] = g["PM2.5"].transform(lambda x: x.shift(1).rolling(7).mean())
    for k in (1, 2, 3):
        df[f"AQI_T{k}"] = g["AQI"].shift(-k)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS)
    engine = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42))
    return engine.fit(df[FEATURE_COLS], df[TARGET_COLS])


@st.cache_resource(show_spinner="Loading data and model…")
def load_all_artifacts():
    csv = find_file(BASE / "data" / "city_day.csv", BASE / "city_day.csv", "city_day.csv")
    hist = pd.read_csv(csv)
    hist["Date"] = pd.to_datetime(hist["Date"])
    model, source = None, "pre-trained file"
    try:
        model = joblib.load(find_file(BASE / "models" / "multi_horizon_aqi_model.pkl",
                                      "models/multi_horizon_aqi_model.pkl"))
        model.predict(pd.DataFrame([[0.0] * len(FEATURE_COLS)], columns=FEATURE_COLS))  # smoke test
    except Exception:
        model, source = None, "trained at start-up"
    if model is None:
        with st.spinner("Training the forecasting model (first start only, about a minute)…"):
            model = _train_from_csv(hist)
    return model, hist, source


try:
    model, df_history, model_source = load_all_artifacts()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}\n\nPut `city_day.csv` in the `data/` folder of the repository.")
    st.stop()

feature_cols = FEATURE_COLS
station_coords = {c: v for c, v in STATION_COORDS.items() if c in set(df_history["City"])}
all_cities = sorted(station_coords.keys())


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_closest_hub(u_lat, u_lon):
    best, best_d = None, float("inf")
    for city, pos in station_coords.items():
        d = haversine_distance(u_lat, u_lon, pos["lat"], pos["lon"])
        if d < best_d:
            best, best_d = city, d
    return best, best_d


@st.cache_data(ttl=600, show_spinner=False)
def get_live_meteorology(lat, lon):
    fallback = {"temp": 28.5, "humidity": 62, "wind": 8.2, "pressure": 1012.0, "live": False}
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure")
        res = requests.get(url, timeout=3).json().get("current", {})
        return {
            "temp": res.get("temperature_2m", fallback["temp"]),
            "humidity": res.get("relative_humidity_2m", fallback["humidity"]),
            "wind": res.get("wind_speed_10m", fallback["wind"]),
            "pressure": res.get("surface_pressure", fallback["pressure"]),
            "live": "temperature_2m" in res,
        }
    except Exception:
        return fallback


def get_cpcb_status(val):
    if val <= 50:
        return "Good", "#10b981", "advisory-good", "Air quality is satisfactory; minimal environmental risk."
    if val <= 100:
        return "Satisfactory", "#3b82f6", "advisory-good", "Minor breathing discomfort to sensitive individuals."
    if val <= 200:
        return "Moderate", "#f59e0b", "advisory-mod", "Breathing discomfort to people with lung/heart disease and asthma."
    if val <= 300:
        return "Poor", "#ea580c", "advisory-poor", "Breathing discomfort to most people on prolonged outdoor exposure."
    if val <= 400:
        return "Very Poor", "#dc2626", "advisory-poor", "Respiratory illness on prolonged exposure. High municipal alert."
    return "Severe", "#9333ea", "advisory-severe", "Critical emergency. Significant health impacts on healthy individuals."


def card(label, value, sub="", color="#0f172a", border=None, value_size=None):
    style = f"border-top: 4px solid {border};" if border else ""
    vstyle = f"color:{color};" + (f"font-size:{value_size}px;" if value_size else "")
    st.markdown(
        f'<div class="metric-card" style="{style}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="{vstyle}">{value}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- header
st.markdown(
    f"""
<div class="top-navbar">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <h1>IntelliAir OS • Smart City Environmental Telemetry Platform</h1>
      <p>Multi-horizon AQI forecasting, CPCB health advisories and GRAP policy guidance</p>
    </div>
    <div style="text-align:right;">
      <div class="live-pulse"><span class="pulse-dot"></span>MODEL ONLINE • CPCB HISTORICAL DATA</div>
      <p style="font-size:11px; color:#64748b; margin-top:6px;">Page loaded: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("### 🎛️ Station Command Center")

default_city = "Kolkata" if "Kolkata" in all_cities else all_cities[0]
if "active_city" not in st.session_state:
    st.session_state["active_city"] = default_city


def _apply_location(lat, lon, label):
    city, dist = find_closest_hub(lat, lon)
    st.session_state["active_city"] = city
    note = f"{label}: nearest station is **{city}** (~{dist:.0f} km away)."
    if dist > 300:
        note += " ⚠️ No monitoring station is close to you, so this is a distant proxy."
    st.session_state["match_msg"] = note


# 1) Automatic detection through the browser's GPS (asks the visitor for permission)
try:
    from streamlit_geolocation import streamlit_geolocation
    st.sidebar.markdown("**📍 Detect my location**")
    st.sidebar.caption("Click the target icon, then allow location access in your browser.")
    with st.sidebar:
        gps = streamlit_geolocation()
    if gps and gps.get("latitude") is not None and gps.get("longitude") is not None:
        gps_key = (round(gps["latitude"], 4), round(gps["longitude"], 4))
        if st.session_state.get("last_gps") != gps_key:      # apply once per new fix, so manual choice sticks
            st.session_state["last_gps"] = gps_key
            _apply_location(gps["latitude"], gps["longitude"], "GPS")
except ImportError:
    pass

# 2) Manual coordinates
with st.sidebar.expander("🧭 Or enter coordinates"):
    st.number_input("Latitude", value=22.57, min_value=6.0, max_value=37.0, format="%.4f", key="gps_lat")
    st.number_input("Longitude", value=88.36, min_value=68.0, max_value=98.0, format="%.4f", key="gps_lon")
    sidebar_button("Match nearest station",
                   on_click=lambda: _apply_location(st.session_state["gps_lat"], st.session_state["gps_lon"], "Coordinates"))

if "match_msg" in st.session_state:
    st.sidebar.success(st.session_state["match_msg"])

# 3) Or simply pick any city
selected_station = st.sidebar.selectbox("🏙️ Or choose any monitoring city:", options=all_cities, key="active_city")

coords = station_coords[selected_station]
weather = get_live_meteorology(round(coords["lat"], 3), round(coords["lon"], 3))

city_records = df_history[df_history["City"] == selected_station].sort_values("Date")
aqi_rows = city_records.dropna(subset=["AQI"])
last_aqi_date = aqi_rows["Date"].iloc[-1].strftime("%d %b %Y") if not aqi_rows.empty else "n/a"


def latest_reading(col, default):
    """Most recent non-null value for this city (the very last row is often NaN)."""
    if col in city_records and city_records[col].notna().any():
        return float(city_records[col].dropna().iloc[-1])
    return float(default)


def slider(label, lo, hi, col, default, step=0.1):
    # Clamp the default: raw readings can exceed the slider range, and Streamlit
    # raises an exception if a slider's default value is out of bounds.
    value = min(max(latest_reading(col, default), lo), hi)
    return st.sidebar.slider(label, float(lo), float(hi), float(round(value, 2)), step=step,
                             key=f"{col}_{selected_station}")


st.sidebar.markdown("---")
st.sidebar.markdown(f"***Fine-tune telemetry ({selected_station})***")
st.sidebar.caption(f"Defaults = latest recorded readings (last AQI record: {last_aqi_date}). Adjust to simulate changes.")

s_pm25 = slider("PM2.5 (µg/m³)", 0, 500, "PM2.5", 68.0)
s_pm10 = slider("PM10 (µg/m³)", 0, 500, "PM10", 135.0)
s_no2 = slider("NO2 (µg/m³)", 0, 200, "NO2", 32.0)
s_co = slider("CO (mg/m³)", 0, 10, "CO", 1.2, step=0.05)
s_so2 = slider("SO2 (µg/m³)", 0, 100, "SO2", 14.0)
s_o3 = slider("O3 (µg/m³)", 0, 150, "O3", 30.0)
s_nh3 = slider("NH3 (µg/m³)", 0, 100, "NH3", 16.0)
s_nox = slider("NOx (ppb)", 0, 300, "NOx", 24.0)
baseline_aqi = latest_reading("AQI", 145.0)

# ---------------------------------------------------------------- inference
now = datetime.now()
sample = pd.DataFrame([{
    "PM2.5": s_pm25, "PM10": s_pm10, "NO2": s_no2, "CO": s_co,
    "SO2": s_so2, "O3": s_o3, "NH3": s_nh3, "NOx": s_nox,
    "Month": now.month, "DayOfWeek": now.weekday(),
    "Is_Weekend": 1 if now.weekday() >= 5 else 0,
    # Approximations: the app has no live feed of yesterday's readings
    "PM25_Lag1": s_pm25 * 0.96, "PM10_Lag1": s_pm10 * 0.96, "PM25_Roll7_Mean": s_pm25 * 0.98,
    "AQI": baseline_aqi,
}])[feature_cols]

t1_val, t2_val, t3_val = [float(v) for v in model.predict(sample)[0][:3]]

tab_dashboard, tab_all, tab_gis = st.tabs(
    ["📊 Forecast for selected city", "🌆 Compare all cities", "🗺️ Station map"])

# ================================================================ TAB 1
with tab_dashboard:
    st.markdown("#### 🛰️ Environmental Context")
    c1, c2, c3, c4 = st.columns(4)
    src = "Source: Open-Meteo (live)" if weather["live"] else "Offline - showing typical values"
    with c1:
        card("Active Monitoring Hub", selected_station,
             f'<div style="font-size:11px;color:#10b981;">📍 {coords["lat"]}° N | {coords["lon"]}° E</div>', value_size=22)
    with c2:
        card("Ambient Temperature", f"{weather['temp']} °C", f'<div style="font-size:11px;color:#64748b;">{src}</div>')
    with c3:
        card("Relative Humidity", f"{weather['humidity']} %", '<div style="font-size:11px;color:#64748b;">Atmospheric vapour level</div>')
    with c4:
        ok = weather["wind"] > 8
        card("Surface Wind", f'{weather["wind"]} <span style="font-size:16px;">km/h</span>',
             f'<div style="font-size:11px;color:{"#10b981" if ok else "#ef4444"};">'
             f'{"Favourable ventilation" if ok else "Stagnant air dispersion"}</div>')

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📈 Multi-Horizon 3-Day AQI Forecast")

    horizon_defs = [
        ("Latest Recorded AQI", baseline_aqi, None),
        ("Day 1 (T+1)", t1_val, t1_val - baseline_aqi),
        ("Day 2 (T+2)", t2_val, t2_val - baseline_aqi),
        ("Day 3 (T+3)", t3_val, t3_val - baseline_aqi),
    ]
    status = [get_cpcb_status(v) for _, v, _ in horizon_defs]
    cols = st.columns(4)
    for col, (label, val, diff), (tier, color, _, _) in zip(cols, horizon_defs, status):
        with col:
            if diff is None:
                sub = f'<div style="font-weight:700;color:{color};font-size:13px;">{tier}</div>'
            else:
                arrow = "▲" if diff > 0 else ("▼" if diff < 0 else "")
                sub = f'<div style="font-weight:700;color:{color};font-size:13px;">{tier} ({arrow} {abs(diff):.1f})</div>'
            card(label, f"{val:.1f}", sub, color=color, border=color)

    tier_t1, _, css_t1, desc_t1 = status[1]
    st.markdown(
        f'<div class="advisory-box {css_t1}"><strong>⚠️ Health advisory for tomorrow ({tier_t1}):</strong> {desc_t1}</div>',
        unsafe_allow_html=True,
    )

    limits = [("PM2.5", s_pm25, 60, "µg/m³"), ("PM10", s_pm10, 100, "µg/m³"),
              ("NO2", s_no2, 80, "µg/m³"), ("CO", s_co, 2.0, "mg/m³")]
    culprits = [f"{n} ({v:.1f} {u} - exceeds {lim} {u})" for n, v, lim, u in limits if v > lim]
    if culprits:
        st.markdown(
            f'<div class="culprit-banner"><strong>🚨 Pollutant exceedances:</strong> {" • ".join(culprits)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c_chart, c_dispatch = st.columns([1.6, 1.0])

    with c_chart:
        st.markdown("#### 📉 3-Day Trajectory vs CPCB Thresholds")
        names = ["Latest", "Day 1", "Day 2", "Day 3"]
        aqi_vals = [v for _, v, _ in horizon_defs]
        fig = go.Figure()
        bands = [(0, 50, "#10b981", "Good"), (50, 100, "#3b82f6", "Satisfactory"), (100, 200, "#f59e0b", "Moderate"),
                 (200, 300, "#ea580c", "Poor"), (300, 400, "#dc2626", "Very Poor"), (400, 550, "#9333ea", "Severe")]
        for y0, y1, color, label in bands:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.08, line_width=0,
                          annotation_text=label, annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=names, y=aqi_vals, mode="lines+markers+text",
            line=dict(color="#0f172a", width=3),
            marker=dict(size=12, color=[s[1] for s in status], line=dict(width=2, color="#ffffff")),
            text=[f"{v:.1f}" for v in aqi_vals], textposition="top center", name="AQI",
        ))
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=360,
                          yaxis=dict(title="Air Quality Index (AQI)", range=[0, max(max(aqi_vals) + 60, 220)]),
                          xaxis=dict(title="Forecast horizon"), showlegend=False, plot_bgcolor="white")
        plot(fig)

    with c_dispatch:
        st.markdown("#### 🏛️ Municipal Policy Actions (GRAP)")
        peak = max(aqi_vals[1:])  # forecast days only
        if peak <= 100:
            st.success("✅ **Normal operations**: regular street sweeping, green-cover watering, standard emission compliance.")
        elif peak <= 200:
            st.warning("⚠️ **Stage I (Moderate)**: dust control at construction sites, ban on open waste burning, mechanised road cleaning.")
        elif peak <= 300:
            st.warning("🟠 **Stage II (Poor)**: ban on diesel generators, intensified water sprinkling, more bus and metro services.")
        elif peak <= 400:
            st.error("🔴 **Stage III (Very Poor)**: halt non-essential construction/demolition, restrict BS-III petrol and BS-IV diesel vehicles.")
        else:
            st.error("🚨 **Stage IV (Severe+)**: ban non-essential trucks, shut high-emission brick kilns, work-from-home advisory.")

# ================================================================ TAB: ALL CITIES
with tab_all:
    st.markdown("#### 🌆 3-Day Forecast for Every Monitored City")
    st.caption("Each city uses its own latest recorded pollutant readings. Pick a row's city in the sidebar to explore it in detail.")
    latest = (df_history.sort_values("Date").groupby("City")[POLLUTANTS + ["AQI"]].last()  # last non-null per column
              .reindex(all_cities))
    latest = latest.fillna(latest.median())
    batch = latest.copy()
    batch["Month"], batch["DayOfWeek"] = now.month, now.weekday()
    batch["Is_Weekend"] = 1 if now.weekday() >= 5 else 0
    batch["PM25_Lag1"], batch["PM10_Lag1"], batch["PM25_Roll7_Mean"] = batch["PM2.5"] * .96, batch["PM10"] * .96, batch["PM2.5"] * .98
    preds = model.predict(batch[feature_cols])
    all_df = pd.DataFrame({"City": all_cities, "Latest AQI": latest["AQI"].round(1).values,
                           "Day 1": preds[:, 0].round(1), "Day 2": preds[:, 1].round(1), "Day 3": preds[:, 2].round(1)})
    all_df["Peak status"] = [get_cpcb_status(max(r))[0] for r in preds]
    all_df = all_df.sort_values("Day 1", ascending=False).reset_index(drop=True)

    fig_all = go.Figure(go.Bar(
        x=all_df["City"], y=all_df["Day 1"],
        marker_color=[get_cpcb_status(v)[1] for v in all_df["Day 1"]],
        text=all_df["Day 1"], textposition="outside"))
    fig_all.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white",
                          yaxis_title="Forecast AQI (tomorrow)", xaxis_tickangle=-45)
    plot(fig_all)
    table(all_df)

# ================================================================ TAB 2
with tab_gis:
    st.markdown("#### 🗺️ CPCB Air Quality Monitoring Stations")
    st.caption("Mean historical AQI (2015-2020) for each monitored city.")

    rows = []
    for city, c in station_coords.items():
        sub = df_history[df_history["City"] == city]["AQI"]
        mean_aqi = float(sub.mean()) if sub.notna().any() else float("nan")
        tier = get_cpcb_status(mean_aqi)[0] if mean_aqi == mean_aqi else "No data"
        rows.append({"City": city, "lat": c["lat"], "lon": c["lon"],
                     "Historical_AQI": round(mean_aqi, 1) if mean_aqi == mean_aqi else None, "Status": tier})
    map_df = pd.DataFrame(rows)

    color_map = {"Good": "#10b981", "Satisfactory": "#3b82f6", "Moderate": "#f59e0b", "Poor": "#ea580c",
                 "Very Poor": "#dc2626", "Severe": "#9333ea", "No data": "#94a3b8"}
    common = dict(lat="lat", lon="lon", hover_name="City",
                  hover_data={"Historical_AQI": True, "Status": True, "lat": False, "lon": False},
                  color="Status", color_discrete_map=color_map, zoom=3.6, center={"lat": 22.5, "lon": 82.0})
    # Plotly 6+ renamed scatter_mapbox -> scatter_map (and removed the old one in newer releases)
    if hasattr(px, "scatter_map"):
        fig_map = px.scatter_map(map_df, map_style="carto-positron", **common)
    else:
        fig_map = px.scatter_mapbox(map_df, mapbox_style="carto-positron", **common)
    fig_map.update_traces(marker=dict(size=16))
    fig_map.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=520)
    plot(fig_map)

    st.markdown("##### 📊 Network Ranking by Mean AQI")
    table(map_df[["City", "Historical_AQI", "Status"]]
          .sort_values("Historical_AQI", ascending=False).reset_index(drop=True))

st.caption(f"Model: {model_source} • Data: CPCB city_day.csv (2015-2020) • Forecasts are indicative, not official advisories.")
