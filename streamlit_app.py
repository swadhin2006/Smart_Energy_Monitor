"""
Smart IoT Energy Meter — Full Dashboard
Features:
  1. Dashboard           - live metrics, trends
  2. Appliance Detector  - rule-based NILM power signature
  3. Anomaly Detector    - Isolation Forest ML
  4. Energy Forecast     - XGBoost ML
  5. Analytics           - monthly, shift-wise, cost
  6. Alerts & Budget     - threshold alerts + email
  7. Audit Report        - PDF export
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests, os, joblib, smtplib, io
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fpdf import FPDF

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# Resolve BASE path — works locally AND on Streamlit Cloud
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except Exception:
    BASE = os.getcwd()

# ── Auto-train if models are missing (runs on Streamlit Cloud first boot) ─────
def _auto_train():
    """Generate data and train models if pkl files don't exist."""
    model_path = os.path.join(BASE, "energy_forecast_model.pkl")
    if os.path.exists(model_path):
        return  # already trained

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from xgboost import XGBRegressor
    import warnings
    warnings.filterwarnings('ignore')

    np.random.seed(42)

    # Generate data
    n_days, freq_minutes = 180, 15
    periods    = int(n_days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range(start='2025-01-01', periods=periods, freq=f'{freq_minutes}min')
    hour  = timestamps.hour.to_numpy()
    dow   = timestamps.dayofweek.to_numpy()
    month = timestamps.month.to_numpy()

    voltage = 220 + np.random.normal(0, 3, periods) + 2 * np.sin(2 * np.pi * hour / 24)
    base_i  = (0.5
               + 1.5 * ((hour >= 7)  & (hour <= 9)).astype(float)
               + 2.0 * ((hour >= 18) & (hour <= 22)).astype(float)
               + 0.3 * (dow >= 5).astype(float)
               + 0.2 * ((month >= 5) & (month <= 8)).astype(float))
    current      = base_i + np.abs(np.random.normal(0, 0.3, periods))
    power_factor = np.random.uniform(0.85, 0.99, periods)
    power        = voltage * current * power_factor
    energy_kwh   = power * (freq_minutes / 60) / 1000
    cost         = energy_kwh * 7.0
    temperature  = 25 + 8 * np.sin(2 * np.pi * (month - 3) / 12) + np.random.normal(0, 1.5, periods)

    anomaly_idx  = np.random.choice(periods, size=int(0.02 * periods), replace=False)
    anomaly_type = np.random.choice(['voltage_spike','current_surge','dropout'], size=len(anomaly_idx))
    is_anomaly   = np.zeros(periods, dtype=int)
    for idx, atype in zip(anomaly_idx, anomaly_type):
        is_anomaly[idx] = 1
        if atype == 'voltage_spike':
            voltage[idx] *= np.random.uniform(1.15, 1.30)
        elif atype == 'current_surge':
            current[idx] *= np.random.uniform(2.0, 4.0)
        else:
            voltage[idx] = 0.0; current[idx] = 0.0

    df = pd.DataFrame({
        'timestamp': timestamps, 'voltage_v': np.round(voltage,2),
        'current_a': np.round(current,3), 'power_w': np.round(power,2),
        'energy_kwh': np.round(energy_kwh,4), 'power_factor': np.round(power_factor,3),
        'temperature_c': np.round(temperature,1), 'cost_inr': np.round(cost,4),
        'is_anomaly': is_anomaly
    })

    # Feature engineering
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month']       = df['timestamp'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour']= df['hour'].isin([7,8,9,18,19,20,21,22]).astype(int)
    df['hour_sin']    = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']    = np.cos(2*np.pi*df['hour']/24)
    df['month_sin']   = np.sin(2*np.pi*df['month']/12)
    df['month_cos']   = np.cos(2*np.pi*df['month']/12)
    df['apparent_power']     = df['voltage_v'] * df['current_a']
    df['reactive_power']     = df['apparent_power'] * np.sqrt(np.maximum(1 - df['power_factor']**2, 0))
    df['power_roll_mean_4']  = df['power_w'].rolling(4,  min_periods=1).mean()
    df['power_roll_mean_24'] = df['power_w'].rolling(24, min_periods=1).mean()
    df['power_roll_std_4']   = df['power_w'].rolling(4,  min_periods=1).std().fillna(0)
    df['energy_lag_1']       = df['energy_kwh'].shift(1).fillna(0)
    df['energy_lag_4']       = df['energy_kwh'].shift(4).fillna(0)
    df['energy_lag_96']      = df['energy_kwh'].shift(96).fillna(0)

    # Forecast model
    FORECAST_FEATURES = [
        'voltage_v','current_a','power_factor','temperature_c',
        'hour_sin','hour_cos','month_sin','month_cos',
        'is_weekend','is_peak_hour','apparent_power','reactive_power',
        'power_roll_mean_4','power_roll_mean_24','power_roll_std_4',
        'energy_lag_1','energy_lag_4','energy_lag_96'
    ]
    X = df[FORECAST_FEATURES].values
    y = df['energy_kwh'].values
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       n_jobs=-1, verbosity=0)
    xgb.fit(X_tr_s, y_tr)
    joblib.dump(xgb,    os.path.join(BASE, 'energy_forecast_model.pkl'))
    joblib.dump(scaler, os.path.join(BASE, 'energy_forecast_scaler.pkl'))
    joblib.dump(FORECAST_FEATURES, os.path.join(BASE, 'forecast_features.pkl'))

    # Anomaly model
    ANOMALY_FEATURES = ['voltage_v','current_a','power_w','power_factor',
                        'apparent_power','reactive_power','power_roll_std_4']
    X_an  = df[ANOMALY_FEATURES].values
    asc   = StandardScaler()
    X_ans = asc.fit_transform(X_an)
    iso   = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(X_ans)
    raw_pred  = iso.predict(X_ans)
    anom_pred = (raw_pred == -1).astype(int)
    scores    = -iso.score_samples(X_ans)
    joblib.dump(iso,  os.path.join(BASE, 'anomaly_model.pkl'))
    joblib.dump(asc,  os.path.join(BASE, 'anomaly_scaler.pkl'))
    joblib.dump(ANOMALY_FEATURES, os.path.join(BASE, 'anomaly_features.pkl'))

    # Save processed data
    df['anomaly_pred']  = anom_pred
    df['anomaly_score'] = np.round(scores, 4)
    df['year_month']    = df['timestamp'].dt.to_period('M').astype(str)
    df.to_csv(os.path.join(BASE, 'energy_data_processed.csv'), index=False)
    df[['timestamp','voltage_v','current_a','power_w','energy_kwh',
        'power_factor','temperature_c','cost_inr','is_anomaly']].to_csv(
        os.path.join(BASE, 'energy_data.csv'), index=False)

# Run auto-train (safe to call every time — skips if models exist)
with st.spinner("Initialising models... (first run takes ~30 seconds)"):
    _auto_train()

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        fm  = joblib.load(os.path.join(BASE, "energy_forecast_model.pkl"))
        fs  = joblib.load(os.path.join(BASE, "energy_forecast_scaler.pkl"))
        am  = joblib.load(os.path.join(BASE, "anomaly_model.pkl"))
        as_ = joblib.load(os.path.join(BASE, "anomaly_scaler.pkl"))
        return fm, fs, am, as_, True
    except Exception:
        return None, None, None, None, False

forecast_model, forecast_scaler, anomaly_model, anomaly_scaler, MODELS_OK = load_models()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for p in [os.path.join(BASE, "energy_data_processed.csv"),
              os.path.join(os.getcwd(), "energy_data_processed.csv")]:
        if os.path.exists(p):
            return pd.read_csv(p, parse_dates=['timestamp'])
    return None

df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# APPLIANCE SIGNATURES  (rule-based NILM)
# ══════════════════════════════════════════════════════════════════════════════
APPLIANCES = {
    "LED Lights":      (5,   40,   "💡", "#F4D03F"),
    "Fan":             (40,  90,   "🌀", "#85C1E9"),
    "Television":      (80,  160,  "📺", "#A9DFBF"),
    "Refrigerator":    (100, 220,  "🧊", "#AED6F1"),
    "Washing Machine": (280, 560,  "🫧", "#D7BDE2"),
    "Air Conditioner": (900, 2200, "❄️", "#F1948A"),
    "Geyser/Heater":   (1400,3200, "🔥", "#F0B27A"),
    "Computer/Laptop": (50,  200,  "💻", "#A3E4D7"),
    "Microwave":       (600, 1200, "📡", "#FAD7A0"),
}

def detect_appliances(power_w: float) -> list:
    """Return list of likely running appliances based on power reading."""
    detected = []
    for name, (lo, hi, icon, color) in APPLIANCES.items():
        if lo <= power_w <= hi:
            detected.append({"name": name, "icon": icon, "color": color,
                             "range": f"{lo}–{hi}W"})
    if not detected:
        if power_w < 5:
            detected.append({"name": "Standby / Idle", "icon": "😴",
                             "color": "#D5D8DC", "range": "0–5W"})
        else:
            detected.append({"name": "Heavy / Unknown Load", "icon": "⚠️",
                             "color": "#E74C3C", "range": f"{power_w:.0f}W"})
    return detected

def appliance_breakdown(df_in: pd.DataFrame) -> pd.DataFrame:
    """For each row in df, find dominant appliance."""
    rows = []
    for _, row in df_in.iterrows():
        apps = detect_appliances(row['power_w'])
        for a in apps:
            rows.append({
                "timestamp": row['timestamp'],
                "power_w":   row['power_w'],
                "appliance": a['name'],
                "icon":      a['icon'],
            })
    return pd.DataFrame(rows)

# ══════════════════════════════════════════════════════════════════════════════
# ML HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_forecast_offline(inputs: dict) -> dict:
    v, i, pf = inputs['voltage_v'], inputs['current_a'], inputs['power_factor']
    ap = v * i
    rp = ap * np.sqrt(max(1 - pf**2, 0))
    h, mo = inputs['hour'], inputs['month']
    row = np.array([[v, i, pf, inputs['temperature_c'],
                     np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
                     np.sin(2*np.pi*mo/12), np.cos(2*np.pi*mo/12),
                     inputs['is_weekend'], inputs['is_peak_hour'],
                     ap, rp,
                     inputs['power_roll_mean_4'], inputs['power_roll_mean_24'],
                     inputs['power_roll_std_4'],
                     inputs['energy_lag_1'], inputs['energy_lag_4'],
                     inputs['energy_lag_96']]])
    scaled = forecast_scaler.transform(row)
    kwh    = max(float(forecast_model.predict(scaled)[0]), 0.0)
    return {"predicted_energy_kwh": round(kwh, 5),
            "predicted_cost_inr":   round(kwh * 7, 4)}

def make_anomaly_offline(inputs: dict) -> dict:
    row = np.array([[inputs['voltage_v'], inputs['current_a'], inputs['power_w'],
                     inputs['power_factor'], inputs['apparent_power'],
                     inputs['reactive_power'], inputs['power_roll_std_4']]])
    scaled  = anomaly_scaler.transform(row)
    raw     = anomaly_model.predict(scaled)[0]
    score   = float(-anomaly_model.score_samples(scaled)[0])
    is_anom = raw == -1
    if is_anom:
        if inputs['voltage_v'] > 250:
            rec = "Voltage spike detected! Check surge protector or ZMPT101B calibration."
        elif inputs['current_a'] > 10:
            rec = "Current surge! Possible short circuit or overloaded appliance."
        elif inputs['voltage_v'] < 10:
            rec = "Power dropout! Check wiring or supply interruption."
        else:
            rec = "Unusual reading. Inspect connections and appliances."
        label = "ANOMALY"
    else:
        rec   = "All readings normal."
        label = "NORMAL"
    return {"is_anomaly": is_anom, "anomaly_score": round(score, 4),
            "label": label, "recommendation": rec}

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL ALERT
# ══════════════════════════════════════════════════════════════════════════════
def send_email_alert(to_email: str, subject: str, body: str,
                     smtp_host: str, smtp_port: int,
                     sender: str, password: str) -> tuple:
    try:
        msg = MIMEMultipart()
        msg['From']    = sender
        msg['To']      = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())
        return True, "Email sent successfully!"
    except Exception as e:
        return False, str(e)

# ══════════════════════════════════════════════════════════════════════════════
# PDF AUDIT REPORT
# ══════════════════════════════════════════════════════════════════════════════
def _safe(text: str) -> str:
    """Replace characters unsupported by Helvetica with ASCII equivalents."""
    return (text
            .replace('\u2014', '-')   # em dash  —  -> -
            .replace('\u2013', '-')   # en dash  –  -> -
            .replace('\u2019', "'")   # right single quote
            .replace('\u2018', "'")   # left single quote
            .replace('\u201c', '"')   # left double quote
            .replace('\u201d', '"')   # right double quote
            .replace('\u20b9', 'Rs')  # ₹ -> Rs
            .replace('\u00b0', ' deg')# °
            .replace('\u2248', '~')   # ≈
            .replace('\u2265', '>=')  # ≥
            .replace('\u2264', '<=')  # ≤
            .encode('latin-1', errors='replace').decode('latin-1')
            )

def generate_pdf_report(df_in: pd.DataFrame, budget: float) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_fill_color(30, 30, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, _safe("Smart Energy Audit Report"), new_x="LMARGIN", new_y="NEXT", fill=True, align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, _safe(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
                   f"Period: {df_in['timestamp'].min().date()} to {df_in['timestamp'].max().date()}"),
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_kwh   = df_in['energy_kwh'].sum()
    total_cost  = df_in['cost_inr'].sum()
    avg_power   = df_in['power_w'].mean()
    max_power   = df_in['power_w'].max()
    anom_count  = int(df_in['anomaly_pred'].sum()) if 'anomaly_pred' in df_in.columns else 0
    efficiency  = max(0, min(100, round(100 - (anom_count / max(len(df_in), 1)) * 100, 1)))
    over_budget = total_cost > budget if budget > 0 else False

    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 8, _safe("1. Executive Summary"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 30, 60)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    metrics = [
        ("Total Energy Consumed", f"{total_kwh:.2f} kWh"),
        ("Total Cost",            f"Rs {total_cost:.2f}"),
        ("Average Power",         f"{avg_power:.1f} W"),
        ("Peak Power",            f"{max_power:.1f} W"),
        ("Anomalies Detected",    str(anom_count)),
        ("Efficiency Score",      f"{efficiency} / 100"),
        ("Monthly Budget",        f"Rs {budget:.2f}" if budget > 0 else "Not set"),
        ("Budget Status",         "OVER BUDGET" if over_budget else "Within Budget"),
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    for label, val in metrics:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, _safe(label + ":"), border=0)
        pdf.set_font("Helvetica", "", 10)
        if "OVER" in val:
            pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 7, _safe(val), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Monthly breakdown ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 8, _safe("2. Monthly Breakdown"), new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    monthly = df_in.groupby(df_in['timestamp'].dt.to_period('M').astype(str)).agg(
        kwh  = ('energy_kwh', 'sum'),
        cost = ('cost_inr',   'sum'),
        avg_v= ('voltage_v',  'mean'),
        avg_i= ('current_a',  'mean'),
    ).round(2).reset_index()

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 230, 255)
    pdf.set_text_color(0, 0, 0)
    for col, w in [("Month",16),("kWh",14),("Cost(Rs)",20),("Avg V",14),("Avg A",14)]:
        pdf.cell(w*2, 6, col, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for _, row in monthly.iterrows():
        for val, w in [(str(row.iloc[0]),16),(str(row['kwh']),14),
                       (str(row['cost']),20),(str(row['avg_v']),14),(str(row['avg_i']),14)]:
            pdf.cell(w*2, 6, _safe(val), border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # ── Appliance estimate ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 8, _safe("3. Estimated Appliance Usage"), new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0,0,0)
    app_counts = {}
    for pw in df_in['power_w']:
        for app in detect_appliances(pw):
            app_counts[app['name']] = app_counts.get(app['name'], 0) + 1
    total_intervals = len(df_in)
    for app, cnt in sorted(app_counts.items(), key=lambda x: -x[1]):
        pct  = cnt / total_intervals * 100
        hrs  = cnt * 0.25
        est_kwh = hrs * dict(
            [(k, (v[0]+v[1])/2/1000) for k,v in APPLIANCES.items()]
        ).get(app, 0.5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 6, _safe(app + ":"))
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6,
                 _safe(f"{pct:.1f}% of time  |  ~{hrs:.0f} hrs  |  ~{est_kwh:.1f} kWh est."),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Recommendations ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 30, 60)
    pdf.cell(0, 8, _safe("4. Recommendations"), new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0,0,0)

    recs = []
    if avg_power > 800:
        recs.append("High average power. Consider replacing old appliances with energy-star rated ones.")
    if anom_count > 50:
        recs.append(f"{anom_count} anomalies detected. Schedule an electrical inspection.")
    if over_budget:
        recs.append("Cost exceeded monthly budget. Shift heavy loads (AC, Geyser) to off-peak hours.")
    recs += [
        "Set AC temperature to 24 deg C - saves ~18% energy vs 20 deg C.",
        "Use LED bulbs throughout - 80% less energy than incandescent.",
        "Unplug standby devices - they consume 5-10% of total energy.",
        "Run washing machine on full load and during off-peak hours (10 PM - 6 AM).",
    ]
    for i, rec in enumerate(recs, 1):
        pdf.multi_cell(0, 6, _safe(f"{i}. {rec}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150,150,150)
    pdf.cell(0, 6, _safe("Generated by Smart IoT Energy Meter System - ESP32 + ACS712 + ZMPT101B"),
             align="C", new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://img.icons8.com/color/96/lightning-bolt.png", width=60)
st.sidebar.title("⚡ Smart Energy Monitor")
st.sidebar.caption("ESP32 + ACS712 + ZMPT101B + Blynk 2.0")

page = st.sidebar.radio("Navigation", [
    "📊 Dashboard",
    "🔌 Appliance Detector",
    "🚨 Anomaly Detector",
    "🔮 Energy Forecast",
    "📈 Analytics",
    "🔔 Alerts & Budget",
    "📄 Audit Report",
])

mode = st.sidebar.selectbox("Prediction Mode",
                             ["Offline (Direct)", "API (FastAPI)"], index=0)
st.sidebar.markdown("---")

# Budget in sidebar state
if 'budget' not in st.session_state:
    st.session_state['budget'] = 1500.0
if 'alert_threshold_w' not in st.session_state:
    st.session_state['alert_threshold_w'] = 2000.0

# Live budget progress
if df is not None:
    cur_month = df['timestamp'].max().strftime('%Y-%m')
    df_month  = df[df['timestamp'].dt.to_period('M').astype(str) == cur_month]
    spent     = float(df_month['cost_inr'].sum())
    budget    = st.session_state['budget']
    pct       = min(spent / budget * 100, 100) if budget > 0 else 0
    color     = "🟢" if pct < 70 else ("🟡" if pct < 90 else "🔴")
    st.sidebar.markdown(f"**Monthly Budget Tracker**")
    st.sidebar.progress(pct / 100)
    st.sidebar.caption(f"{color} ₹{spent:.0f} / ₹{budget:.0f} ({pct:.1f}%)")
    st.sidebar.markdown("---")

st.sidebar.info("**Models:**\n- XGBoost → Forecast\n- Isolation Forest → Anomaly")
st.sidebar.caption(f"📁 `{BASE}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("⚡ Smart IoT Energy Meter Dashboard")
    st.caption("Real-time monitoring via ESP32 + ACS712 + ZMPT101B")

    if df is None:
        st.warning("No data found. Run `python train_models.py` first.")
        st.stop()

    latest = df.iloc[-1]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("🔌 Voltage",     f"{latest['voltage_v']:.1f} V")
    c2.metric("⚡ Current",     f"{latest['current_a']:.3f} A")
    c3.metric("💡 Power",       f"{latest['power_w']:.1f} W")
    c4.metric("📦 Energy",      f"{latest['energy_kwh']:.4f} kWh")
    c5.metric("💰 Cost",        f"₹{latest['cost_inr']:.4f}")
    anom_today = int(df[df['timestamp'].dt.date == df['timestamp'].max().date()]
                     .get('anomaly_pred', pd.Series([0])).sum())
    c6.metric("🚨 Alerts Today", str(anom_today))

    # ── Live appliance detection ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔌 Currently Detected Appliances")
    apps = detect_appliances(latest['power_w'])
    cols = st.columns(len(apps))
    for col, app in zip(cols, apps):
        col.markdown(
            f"<div style='background:{app['color']};padding:12px;border-radius:10px;"
            f"text-align:center;font-size:18px'>"
            f"{app['icon']} <b>{app['name']}</b><br>"
            f"<small>{app['range']}</small></div>",
            unsafe_allow_html=True
        )

    # ── Budget alert banner ────────────────────────────────────────────────────
    st.markdown("")
    budget = st.session_state['budget']
    if df is not None:
        cur_month = df['timestamp'].max().strftime('%Y-%m')
        spent = float(df[df['timestamp'].dt.to_period('M').astype(str) == cur_month]['cost_inr'].sum())
        if budget > 0 and spent >= budget:
            st.error(f"🚨 **Budget Exceeded!** Spent ₹{spent:.2f} of ₹{budget:.2f} budget this month.")
        elif budget > 0 and spent >= budget * 0.9:
            st.warning(f"⚠️ **Approaching Budget!** Spent ₹{spent:.2f} — 90% of ₹{budget:.2f} used.")

    # ── Threshold alert ────────────────────────────────────────────────────────
    threshold_w = st.session_state['alert_threshold_w']
    if latest['power_w'] > threshold_w:
        st.error(f"🔴 **High Power Alert!** Current reading {latest['power_w']:.0f}W exceeds "
                 f"threshold of {threshold_w:.0f}W")

    st.markdown("---")

    # ── Trend charts ──────────────────────────────────────────────────────────
    last7 = df[df['timestamp'] >= df['timestamp'].max() - timedelta(days=7)]
    fig_trend = px.line(last7, x='timestamp', y='power_w',
                        title='Power Consumption — Last 7 Days',
                        labels={'power_w': 'Power (W)', 'timestamp': 'Time'},
                        color_discrete_sequence=['#FF6B35'])
    st.plotly_chart(fig_trend, width='stretch')

    col_a, col_b = st.columns(2)
    with col_a:
        daily = df.groupby(df['timestamp'].dt.date)['energy_kwh'].sum().reset_index()
        daily.columns = ['date', 'daily_kwh']
        fig_d = px.bar(daily.tail(30), x='date', y='daily_kwh',
                       title='Daily Energy (last 30 days)', labels={'daily_kwh': 'kWh'},
                       color='daily_kwh', color_continuous_scale='Blues')
        st.plotly_chart(fig_d, width='stretch')

    with col_b:
        hourly = df.groupby('hour')['power_w'].mean().reset_index()
        fig_h  = px.line(hourly, x='hour', y='power_w', title='Avg Power by Hour',
                         markers=True, color_discrete_sequence=['#4CAF50'])
        fig_h.add_vrect(x0=7, x1=9,   fillcolor='orange', opacity=0.15,
                        annotation_text='Morning peak')
        fig_h.add_vrect(x0=18, x1=22, fillcolor='red',    opacity=0.15,
                        annotation_text='Evening peak')
        st.plotly_chart(fig_h, width='stretch')

    anom_count = int(df['anomaly_pred'].sum()) if 'anomaly_pred' in df.columns else 0
    st.info(f"🚨 Total anomalies in dataset: **{anom_count}** ({anom_count/len(df)*100:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — APPLIANCE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔌 Appliance Detector":
    st.title("🔌 Appliance-Level Power Detection")
    st.caption("Rule-based NILM (Non-Intrusive Load Monitoring) — no extra hardware needed")

    tab_live, tab_history, tab_ref = st.tabs(
        ["🔴 Live Detection", "📊 Historical Breakdown", "📋 Appliance Reference"])

    # ── Live detection ────────────────────────────────────────────────────────
    with tab_live:
        st.subheader("Enter Current Power Reading")
        col1, col2 = st.columns([1, 2])
        with col1:
            pw_input = st.number_input("Power (W)", min_value=0.0,
                                       value=float(df.iloc[-1]['power_w']) if df is not None else 500.0,
                                       step=10.0)
            voltage_in = st.number_input("Voltage (V)", min_value=0.0, value=220.5, step=0.1)
            current_in = st.number_input("Current (A)", min_value=0.0, value=2.3, step=0.01)
            detect_btn = st.button("🔍 Detect Appliances", type="primary")

        with col2:
            if detect_btn or pw_input:
                apps = detect_appliances(pw_input)
                st.markdown("### Detected Appliances")
                for app in apps:
                    st.markdown(
                        f"<div style='background:{app['color']};padding:16px;"
                        f"border-radius:12px;margin:6px 0;font-size:16px'>"
                        f"<b>{app['icon']} {app['name']}</b> &nbsp;|&nbsp; "
                        f"Typical range: {app['range']}</div>",
                        unsafe_allow_html=True
                    )

                # Power pie vs appliance ranges
                fig_pie = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pw_input,
                    title={"text": "Power (W)"},
                    gauge={
                        'axis': {'range': [0, 3500]},
                        'bar':  {'color': "#FF6B35"},
                        'steps': [
                            {'range': [0,   90],   'color': '#d4efdf'},
                            {'range': [90,  500],  'color': '#fef9e7'},
                            {'range': [500, 1500],  'color': '#fdebd0'},
                            {'range': [1500,3500],  'color': '#fadbd8'},
                        ],
                        'threshold': {'line': {'color': 'red','width':4},
                                      'thickness': 0.75,
                                      'value': st.session_state['alert_threshold_w']}
                    }
                ))
                st.plotly_chart(fig_pie, width='stretch')

                # Estimated cost
                kwh_hr = pw_input / 1000
                st.info(f"⏱ At this load: **{kwh_hr:.3f} kWh/hr** = "
                        f"**₹{kwh_hr*7:.2f}/hr** = "
                        f"**₹{kwh_hr*7*24:.2f}/day**")

    # ── Historical breakdown ───────────────────────────────────────────────────
    with tab_history:
        if df is None:
            st.warning("No data available.")
        else:
            st.subheader("Appliance Usage Over Time")
            n_sample = st.slider("Sample size (rows)", 500, 5000, 1000, 500)
            sample   = df.sample(n_sample).sort_values('timestamp')

            app_counts = {}
            for pw in sample['power_w']:
                for app in detect_appliances(pw):
                    app_counts[app['name']] = app_counts.get(app['name'], 0) + 1

            app_df = pd.DataFrame([
                {"Appliance": k,
                 "Intervals": v,
                 "Hours": round(v * 0.25, 1),
                 "% Time": round(v / n_sample * 100, 1),
                 "Est. kWh": round(v * 0.25 * dict(
                     [(k2, (v2[0]+v2[1])/2/1000) for k2,v2 in APPLIANCES.items()]
                 ).get(k, 0.5), 2)}
                for k, v in sorted(app_counts.items(), key=lambda x: -x[1])
            ])

            col_l, col_r = st.columns(2)
            with col_l:
                st.dataframe(app_df, width='stretch')
            with col_r:
                fig_app = px.pie(app_df, names='Appliance', values='Intervals',
                                 title='Time Distribution by Appliance',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_app, width='stretch')

            # Shift-wise usage (Industrial / Office use case)
            st.markdown("---")
            st.subheader("🏭 Shift-Wise Consumption (Industrial / Office)")
            df_s = df.copy()
            df_s['shift'] = pd.cut(df_s['hour'],
                                   bins=[-1, 7, 15, 23],
                                   labels=['🌙 Night (0–8)', '☀️ Day (8–16)', '🌆 Evening (16–24)'])
            shift_summary = df_s.groupby('shift', observed=True).agg(
                avg_power   = ('power_w',    'mean'),
                total_kwh   = ('energy_kwh', 'sum'),
                total_cost  = ('cost_inr',   'sum'),
            ).round(2).reset_index()
            shift_summary.columns = ['Shift', 'Avg Power (W)', 'Total kWh', 'Total Cost (₹)']

            fig_shift = px.bar(shift_summary, x='Shift', y='Total kWh',
                               color='Avg Power (W)', text='Total kWh',
                               title='Energy Consumption by Shift',
                               color_continuous_scale='RdYlGn_r')
            fig_shift.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig_shift, width='stretch')
            col2.dataframe(shift_summary, width='stretch')

    # ── Reference table ───────────────────────────────────────────────────────
    with tab_ref:
        st.subheader("📋 Appliance Power Signature Reference")
        ref_df = pd.DataFrame([
            {"Icon": v[2], "Appliance": k, "Min W": v[0], "Max W": v[1],
             "Avg W": (v[0]+v[1])//2,
             "Cost/hr (₹)": round((v[0]+v[1])/2/1000*7, 2),
             "Cost/day (₹)": round((v[0]+v[1])/2/1000*7*8, 2)}
            for k,v in APPLIANCES.items()
        ])
        st.dataframe(ref_df, width='stretch', hide_index=True)
        st.caption("Cost calculated at ₹7/kWh. Daily cost assumes 8 hours of use.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detector":
    st.title("🚨 Real-Time Anomaly Detector")
    st.caption("Isolation Forest ML model — detects voltage spikes, current surges, power dropouts")

    with st.form("anomaly_form"):
        st.subheader("Live Sensor Reading")
        col1, col2 = st.columns(2)
        with col1:
            voltage      = st.number_input("Voltage (V)",    min_value=0.0, value=220.5, step=0.1)
            current      = st.number_input("Current (A)",    min_value=0.0, value=2.3,   step=0.01)
            power_w      = st.number_input("Power (W)",      min_value=0.0, value=460.0, step=1.0)
        with col2:
            power_factor = st.slider("Power Factor", 0.0, 1.0, 0.92, 0.01)
            rstd         = st.number_input("Roll Std Power 1h (W)", value=12.0)
        submitted = st.form_submit_button("🔍 Detect Anomaly", type="primary")

    if submitted:
        apparent = voltage * current
        reactive = apparent * np.sqrt(max(1 - power_factor**2, 0))
        inputs   = dict(voltage_v=voltage, current_a=current, power_w=power_w,
                        power_factor=power_factor, apparent_power=apparent,
                        reactive_power=reactive, power_roll_std_4=rstd)
        with st.spinner("Analyzing..."):
            try:
                if mode == "API (FastAPI)":
                    r = requests.post(f"{API_URL}/predict/anomaly", json=inputs, timeout=5)
                    result = r.json()
                else:
                    if not MODELS_OK:
                        st.error(f"Models not found in `{BASE}`. Run: python train_models.py")
                        st.stop()
                    result = make_anomaly_offline(inputs)

                is_anom = result['is_anomaly']
                score   = result['anomaly_score']
                label   = result['label']
                rec     = result['recommendation']

                if is_anom:
                    st.error(f"🚨 **{label}** — Anomaly Score: {score:.4f}")
                    st.warning(f"💡 **Recommendation:** {rec}")
                    # Also show appliance context
                    apps = detect_appliances(power_w)
                    st.markdown("**Possible appliances involved:**")
                    for app in apps:
                        st.markdown(f"- {app['icon']} {app['name']} ({app['range']})")
                else:
                    st.success(f"✅ **{label}** — Anomaly Score: {score:.4f}")
                    st.info(rec)

                col_g, col_d = st.columns(2)
                with col_g:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        title={'text': "Anomaly Score"},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar':  {'color': "crimson" if is_anom else "green"},
                            'steps': [
                                {'range': [0, 0.4],   'color': '#d4efdf'},
                                {'range': [0.4, 0.6], 'color': '#fef9e7'},
                                {'range': [0.6, 1.0], 'color': '#fadbd8'},
                            ],
                            'threshold': {'line': {'color': 'red', 'width': 4},
                                          'thickness': 0.75, 'value': 0.55}
                        }
                    ))
                    st.plotly_chart(fig_gauge, width='stretch')
                with col_d:
                    st.markdown("### Reading Summary")
                    st.json({
                        "voltage_v":    voltage,
                        "current_a":    current,
                        "power_w":      power_w,
                        "power_factor": power_factor,
                        "apparent_VA":  round(apparent, 2),
                        "reactive_VAR": round(reactive, 2),
                        "anomaly":      label,
                        "score":        score,
                    })
            except Exception as e:
                st.error(f"Detection failed: {e}")

    # ── Historical anomaly log ─────────────────────────────────────────────────
    if df is not None and 'anomaly_pred' in df.columns:
        st.markdown("---")
        st.subheader("📋 Historical Anomaly Log")
        anomalies = df[df['anomaly_pred'] == 1][
            ['timestamp','voltage_v','current_a','power_w','anomaly_score']
        ].sort_values('anomaly_score', ascending=False).head(50)
        st.dataframe(anomalies, width='stretch', hide_index=True)

        fig_tl = px.scatter(
            df.sample(min(3000, len(df))).sort_values('timestamp'),
            x='timestamp', y='power_w',
            color=df.sample(min(3000,len(df))).sort_values('timestamp')['anomaly_pred'].map({0:'Normal',1:'Anomaly'}),
            color_discrete_map={'Normal':'steelblue','Anomaly':'red'},
            title='Power Timeline with Anomalies',
            labels={'power_w': 'Power (W)'}
        )
        st.plotly_chart(fig_tl, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ENERGY FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Energy Forecast":
    st.title("🔮 Energy Consumption Forecast")
    st.caption("Predict next 15-minute energy usage (kWh) using XGBoost — R² = 0.9983")

    with st.form("forecast_form"):
        st.subheader("Sensor Inputs")
        col1, col2, col3 = st.columns(3)
        with col1:
            voltage      = st.number_input("Voltage (V)",      min_value=0.0, value=220.5, step=0.1)
            current      = st.number_input("Current (A)",      min_value=0.0, value=2.3,   step=0.01)
            power_factor = st.slider("Power Factor",           0.0, 1.0, 0.92, 0.01)
        with col2:
            temperature  = st.number_input("Temperature (°C)", min_value=-20.0, value=28.0, step=0.1)
            hour         = st.selectbox("Hour of Day",         list(range(24)), index=18)
            month        = st.selectbox("Month",               list(range(1, 13)), index=5)
        with col3:
            is_weekend   = st.radio("Weekend?",   [0,1], format_func=lambda x: "Yes" if x else "No")
            is_peak_hour = st.radio("Peak Hour?", [0,1], format_func=lambda x: "Yes" if x else "No")

        st.subheader("Rolling / Lag Features")
        c1, c2 = st.columns(2)
        with c1:
            rm4  = st.number_input("Roll Avg Power 1h (W)",        value=480.0)
            rm24 = st.number_input("Roll Avg Power 6h (W)",        value=460.0)
            rstd = st.number_input("Roll Std Power 1h (W)",        value=15.0)
        with c2:
            lag1  = st.number_input("Energy 15min ago (kWh)",      value=0.12, format="%.5f")
            lag4  = st.number_input("Energy 1hr ago (kWh)",        value=0.11, format="%.5f")
            lag96 = st.number_input("Energy same-time yest (kWh)", value=0.10, format="%.5f")

        submitted = st.form_submit_button("⚡ Predict Energy", type="primary")

    if submitted:
        inputs = dict(
            voltage_v=voltage, current_a=current, power_factor=power_factor,
            temperature_c=temperature, hour=hour, month=month,
            is_weekend=is_weekend, is_peak_hour=is_peak_hour,
            power_roll_mean_4=rm4, power_roll_mean_24=rm24, power_roll_std_4=rstd,
            energy_lag_1=lag1, energy_lag_4=lag4, energy_lag_96=lag96
        )
        with st.spinner("Predicting..."):
            try:
                if mode == "API (FastAPI)":
                    r = requests.post(f"{API_URL}/predict/energy", json=inputs, timeout=5)
                    result = r.json()
                else:
                    if not MODELS_OK:
                        st.error(f"Models not found in `{BASE}`. Run: python train_models.py")
                        st.stop()
                    result = make_forecast_offline(inputs)

                kwh  = result['predicted_energy_kwh']
                cost = result['predicted_cost_inr']

                st.success("Prediction complete!")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⚡ Predicted Energy",    f"{kwh:.5f} kWh")
                m2.metric("💰 Est. Cost (15 min)",  f"₹{cost:.4f}")
                m3.metric("🔋 Hourly Projection",   f"{kwh*4:.3f} kWh/hr")
                m4.metric("📅 Daily Projection",    f"{kwh*96:.2f} kWh/day")

                # Recommendations based on prediction
                daily_proj = kwh * 96
                if daily_proj > 20:
                    st.error("⚠️ High projected daily consumption. Consider reducing AC/Geyser usage.")
                elif daily_proj > 10:
                    st.warning("📊 Moderate consumption. Check if any appliances are left on standby.")
                else:
                    st.success("✅ Low consumption. Great energy efficiency!")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=kwh * 1000,
                    title={'text': "Predicted Energy (Wh)"},
                    gauge={
                        'axis': {'range': [0, 500]},
                        'bar':  {'color': "#FF6B35"},
                        'steps': [
                            {'range': [0,   100], 'color': '#d4edda'},
                            {'range': [100, 250], 'color': '#fff3cd'},
                            {'range': [250, 500], 'color': '#f8d7da'},
                        ]
                    }
                ))
                st.plotly_chart(fig_gauge, width='stretch')

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.title("📈 Energy Analytics")

    if df is None:
        st.warning("Run `python train_models.py` first.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📅 Monthly Summary", "🚨 Anomaly Timeline",
        "💰 Cost Analysis",   "🏭 Shift Analysis"])

    with tab1:
        monthly = df.groupby(df['timestamp'].dt.to_period('M').astype(str)).agg(
            total_kwh   = ('energy_kwh', 'sum'),
            total_cost  = ('cost_inr',   'sum'),
            avg_v       = ('voltage_v',  'mean'),
            avg_i       = ('current_a',  'mean'),
            anomalies   = ('anomaly_pred','sum') if 'anomaly_pred' in df.columns else ('energy_kwh','count'),
        ).round(3).reset_index()
        monthly.columns = ['Month','Total kWh','Total Cost (₹)','Avg V','Avg A','Anomalies']
        st.dataframe(monthly, width='stretch', hide_index=True)

        fig = px.bar(monthly, x='Month', y='Total kWh',
                     color='Total Cost (₹)', color_continuous_scale='RdYlGn_r',
                     text='Total kWh', title='Monthly Energy & Cost')
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, width='stretch')

        # Month-over-month change
        if len(monthly) >= 2:
            last  = monthly.iloc[-1]['Total kWh']
            prev  = monthly.iloc[-2]['Total kWh']
            delta = (last - prev) / prev * 100
            if delta > 0:
                st.warning(f"📈 Consumption increased by **{delta:.1f}%** vs last month.")
            else:
                st.success(f"📉 Consumption decreased by **{abs(delta):.1f}%** vs last month. Good job!")

    with tab2:
        if 'anomaly_pred' in df.columns:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Anomalies", int(df['anomaly_pred'].sum()))
            c2.metric("Anomaly Rate",    f"{df['anomaly_pred'].mean()*100:.2f}%")
            c3.metric("Max Score",       f"{df['anomaly_score'].max():.4f}" if 'anomaly_score' in df.columns else "N/A")

            sample = df.sample(min(3000, len(df))).sort_values('timestamp')
            fig_tl = px.scatter(sample, x='timestamp', y='power_w',
                                color=sample['anomaly_pred'].map({0:'Normal',1:'Anomaly'}),
                                color_discrete_map={'Normal':'steelblue','Anomaly':'red'},
                                title='Anomaly Timeline', labels={'power_w':'Power (W)'})
            st.plotly_chart(fig_tl, width='stretch')

            # Anomaly type breakdown by voltage/current thresholds
            if 'anomaly_pred' in df.columns:
                anom_df = df[df['anomaly_pred']==1].copy()
                anom_df['type'] = 'Unusual'
                anom_df.loc[anom_df['voltage_v'] > 250, 'type'] = 'Voltage Spike'
                anom_df.loc[anom_df['current_a'] > 10,  'type'] = 'Current Surge'
                anom_df.loc[anom_df['voltage_v'] < 10,  'type'] = 'Power Dropout'
                type_counts = anom_df['type'].value_counts().reset_index()
                type_counts.columns = ['Type', 'Count']
                fig_types = px.pie(type_counts, names='Type', values='Count',
                                   title='Anomaly Types Breakdown',
                                   color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_types, width='stretch')

    with tab3:
        df['date'] = df['timestamp'].dt.date
        daily_cost = df.groupby('date')['cost_inr'].sum().reset_index()
        daily_cost.columns = ['date', 'daily_cost_inr']

        fig_cost = px.area(daily_cost.tail(60), x='date', y='daily_cost_inr',
                           title='Daily Cost (₹) — last 60 days',
                           labels={'daily_cost_inr': 'Cost (₹)'},
                           color_discrete_sequence=['#FF6B35'])
        budget = st.session_state['budget']
        if budget > 0:
            daily_budget = budget / 30
            fig_cost.add_hline(y=daily_budget, line_dash='dash', line_color='red',
                               annotation_text=f'Daily budget limit ₹{daily_budget:.0f}')
        st.plotly_chart(fig_cost, width='stretch')

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Cost (6 months)", f"₹{df['cost_inr'].sum():.2f}")
        c2.metric("Avg Daily Cost",        f"₹{daily_cost['daily_cost_inr'].mean():.2f}")
        c3.metric("Peak Day Cost",         f"₹{daily_cost['daily_cost_inr'].max():.2f}")

    with tab4:
        st.subheader("🏭 Shift-Wise Analysis (Industrial / Office)")
        df_s = df.copy()
        df_s['shift'] = pd.cut(df_s['hour'],
                               bins=[-1, 7, 15, 23],
                               labels=['🌙 Night (0–8)', '☀️ Day (8–16)', '🌆 Evening (16–24)'])
        shift_agg = df_s.groupby('shift', observed=True).agg(
            avg_power  = ('power_w',    'mean'),
            total_kwh  = ('energy_kwh', 'sum'),
            total_cost = ('cost_inr',   'sum'),
        ).round(2).reset_index()
        shift_agg.columns = ['Shift', 'Avg Power (W)', 'Total kWh', 'Total Cost (₹)']

        col1, col2 = st.columns(2)
        with col1:
            fig_s = px.bar(shift_agg, x='Shift', y='Total kWh',
                           color='Avg Power (W)', text='Total kWh',
                           title='kWh by Shift', color_continuous_scale='Viridis')
            fig_s.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            st.plotly_chart(fig_s, width='stretch')
        with col2:
            st.dataframe(shift_agg, width='stretch', hide_index=True)
            night_kwh = float(shift_agg[shift_agg['Shift'].str.contains('Night')]['Total kWh'].sum())
            day_kwh   = float(shift_agg[shift_agg['Shift'].str.contains('Day')]['Total kWh'].sum())
            if night_kwh > day_kwh * 0.5:
                st.warning("🌙 High night-time consumption detected. Check for equipment left running overnight.")
            else:
                st.success("✅ Night consumption is within normal range.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ALERTS & BUDGET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Alerts & Budget":
    st.title("🔔 Alerts & Budget Management")

    tab_budget, tab_threshold, tab_email, tab_recs = st.tabs([
        "💰 Budget Tracker", "⚡ Threshold Alerts",
        "📧 Email Alerts",   "💡 Recommendations"])

    # ── Budget tracker ────────────────────────────────────────────────────────
    with tab_budget:
        st.subheader("Set Monthly Budget")
        new_budget = st.number_input("Monthly Budget (₹)", min_value=0.0,
                                     value=st.session_state['budget'], step=100.0)
        if st.button("💾 Save Budget"):
            st.session_state['budget'] = new_budget
            st.success(f"Budget set to ₹{new_budget:.2f}")

        if df is not None:
            st.markdown("### Monthly Spending vs Budget")
            monthly_cost = df.groupby(df['timestamp'].dt.to_period('M').astype(str))['cost_inr'].sum().reset_index()
            monthly_cost.columns = ['Month', 'Spent (₹)']
            monthly_cost['Budget (₹)']  = st.session_state['budget']
            monthly_cost['Status']       = monthly_cost['Spent (₹)'].apply(
                lambda x: '🔴 Over' if x > st.session_state['budget']
                          else ('🟡 Warning' if x > st.session_state['budget'] * 0.9 else '🟢 OK')
            )

            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(name='Spent', x=monthly_cost['Month'],
                                   y=monthly_cost['Spent (₹)'], marker_color='#FF6B35'))
            fig_b.add_trace(go.Scatter(name='Budget Limit', x=monthly_cost['Month'],
                                       y=monthly_cost['Budget (₹)'],
                                       mode='lines+markers',
                                       line=dict(color='red', dash='dash', width=2)))
            fig_b.update_layout(title='Monthly Spend vs Budget', barmode='group')
            st.plotly_chart(fig_b, width='stretch')
            st.dataframe(monthly_cost, width='stretch', hide_index=True)

    # ── Threshold alerts ──────────────────────────────────────────────────────
    with tab_threshold:
        st.subheader("Power Threshold Alerts")
        new_threshold = st.number_input(
            "Alert when power exceeds (W)",
            min_value=100.0,
            value=st.session_state['alert_threshold_w'],
            step=100.0
        )
        if st.button("💾 Save Threshold"):
            st.session_state['alert_threshold_w'] = new_threshold
            st.success(f"Threshold set to {new_threshold:.0f}W")

        if df is not None:
            exceeded = df[df['power_w'] > st.session_state['alert_threshold_w']]
            st.markdown(f"### Readings exceeding {st.session_state['alert_threshold_w']:.0f}W")
            c1, c2, c3 = st.columns(3)
            c1.metric("Exceedances",     len(exceeded))
            c2.metric("% of readings",   f"{len(exceeded)/len(df)*100:.1f}%")
            c3.metric("Max recorded",    f"{df['power_w'].max():.0f}W")

            if len(exceeded) > 0:
                st.warning(f"⚠️ {len(exceeded)} readings exceeded your threshold!")
                fig_exc = px.scatter(
                    df.sample(min(2000, len(df))).sort_values('timestamp'),
                    x='timestamp', y='power_w',
                    color=(df.sample(min(2000,len(df))).sort_values('timestamp')['power_w']
                           > st.session_state['alert_threshold_w']).map({True:'Exceeded', False:'Normal'}),
                    color_discrete_map={'Normal':'steelblue', 'Exceeded':'red'},
                    title='Power Threshold Exceedances'
                )
                fig_exc.add_hline(y=st.session_state['alert_threshold_w'],
                                  line_dash='dash', line_color='red',
                                  annotation_text='Threshold')
                st.plotly_chart(fig_exc, width='stretch')

    # ── Email alerts ──────────────────────────────────────────────────────────
    with tab_email:
        st.subheader("📧 Email Alert Configuration")
        st.info("Uses Gmail SMTP. Enable 'App Passwords' in your Google account settings.")

        with st.form("email_form"):
            col1, col2 = st.columns(2)
            with col1:
                sender_email   = st.text_input("Your Gmail", placeholder="you@gmail.com")
                app_password   = st.text_input("App Password", type="password",
                                               placeholder="16-char app password")
            with col2:
                to_email       = st.text_input("Send alert to", placeholder="recipient@email.com")
                alert_type     = st.selectbox("Alert Type", [
                    "Daily Summary",
                    "Budget Exceeded",
                    "High Anomaly Count",
                    "Test Email",
                ])
            send_btn = st.form_submit_button("📤 Send Alert Email", type="primary")

        if send_btn:
            if not sender_email or not app_password or not to_email:
                st.error("Please fill all email fields.")
            else:
                # Build email body
                if df is not None:
                    total_kwh  = df['energy_kwh'].sum()
                    total_cost = df['cost_inr'].sum()
                    anom_count = int(df['anomaly_pred'].sum()) if 'anomaly_pred' in df.columns else 0
                else:
                    total_kwh = total_cost = anom_count = 0

                body = f"""
                <html><body>
                <h2 style='color:#FF6B35'>⚡ Smart Energy Meter — {alert_type}</h2>
                <hr>
                <table border='1' cellpadding='8' style='border-collapse:collapse'>
                  <tr><td><b>Total Energy</b></td><td>{total_kwh:.2f} kWh</td></tr>
                  <tr><td><b>Total Cost</b></td><td>₹{total_cost:.2f}</td></tr>
                  <tr><td><b>Anomalies</b></td><td>{anom_count}</td></tr>
                  <tr><td><b>Monthly Budget</b></td><td>₹{st.session_state['budget']:.2f}</td></tr>
                  <tr><td><b>Budget Status</b></td>
                      <td>{'🔴 EXCEEDED' if total_cost > st.session_state['budget'] else '🟢 OK'}</td></tr>
                  <tr><td><b>Generated</b></td><td>{datetime.now().strftime('%Y-%m-%d %H:%M')}</td></tr>
                </table>
                <br>
                <p style='color:gray;font-size:12px'>
                Smart IoT Energy Meter — ESP32 + ACS712 + ZMPT101B</p>
                </body></html>
                """
                with st.spinner("Sending email..."):
                    ok, msg = send_email_alert(
                        to_email, f"⚡ Energy Alert: {alert_type}",
                        body, "smtp.gmail.com", 465, sender_email, app_password
                    )
                if ok:
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ Failed: {msg}")
                    st.info("Make sure you're using an **App Password**, not your regular Gmail password. "
                            "Go to Google Account → Security → 2-Step Verification → App Passwords.")

    # ── Recommendations ────────────────────────────────────────────────────────
    with tab_recs:
        st.subheader("💡 Smart Recommendations")

        if df is not None:
            avg_power   = df['power_w'].mean()
            total_cost  = df['cost_inr'].sum()
            anom_count  = int(df['anomaly_pred'].sum()) if 'anomaly_pred' in df.columns else 0
            budget      = st.session_state['budget']
            night_usage = float(df[df['hour'].between(0,6)]['power_w'].mean())
            peak_usage  = float(df[df['hour'].between(18,22)]['power_w'].mean())

            recs = []

            if total_cost > budget * 6 and budget > 0:
                saving = total_cost - budget * 6
                recs.append(("🔴 Budget Alert",
                              f"You exceeded 6-month budget by ₹{saving:.0f}. "
                              "Reduce AC and geyser usage during peak hours."))

            if peak_usage > avg_power * 1.5:
                recs.append(("🌆 Peak Hour Load",
                              f"Evening peak ({peak_usage:.0f}W avg) is {(peak_usage/avg_power-1)*100:.0f}% "
                              "above average. Shift washing machine / dishwasher to 10 PM–6 AM."))

            if night_usage > avg_power * 0.6:
                recs.append(("🌙 Night Consumption",
                              f"Night usage is high ({night_usage:.0f}W). "
                              "Check if AC, lights or equipment are left running overnight."))

            if anom_count > 100:
                recs.append(("⚡ Electrical Issues",
                              f"{anom_count} anomalies detected. "
                              "Schedule a professional electrical inspection."))

            # Always-on recommendations
            recs += [
                ("💡 LED Lighting",
                 "Replace all bulbs with LED. Saves 80% energy vs incandescent."),
                ("❄️ AC Efficiency",
                 "Set AC to 24°C — each degree below 24 increases energy use by ~6%."),
                ("🧊 Refrigerator",
                 "Keep fridge 3/4 full and away from walls. Clean coils yearly."),
                ("🔌 Standby Power",
                 "Unplug chargers & electronics when not in use — saves 5–10% total energy."),
                ("🌞 Off-Peak Shifting",
                 "Run heavy appliances between 10 PM–6 AM for lower tariff rates."),
            ]

            for title, rec in recs:
                with st.expander(title):
                    st.write(rec)

            # Efficiency score
            efficiency = max(0, min(100, round(100 - (anom_count / max(len(df),1)) * 500, 1)))
            st.markdown("---")
            st.subheader("🏆 Energy Efficiency Score")
            fig_eff = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=efficiency,
                title={'text': "Efficiency Score (0–100)"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': "#2ECC71" if efficiency >= 80 else
                                      "#F39C12" if efficiency >= 60 else "#E74C3C"},
                    'steps': [
                        {'range': [0,  60], 'color': '#fadbd8'},
                        {'range': [60, 80], 'color': '#fef9e7'},
                        {'range': [80,100], 'color': '#d4efdf'},
                    ]
                }
            ))
            st.plotly_chart(fig_eff, width='stretch')
        else:
            st.warning("No data available for recommendations.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AUDIT REPORT (PDF)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Audit Report":
    st.title("📄 Energy Audit Report")
    st.caption("Generate a complete PDF audit report for homes, offices, and industries")

    if df is None:
        st.warning("No data available. Run `python train_models.py` first.")
        st.stop()

    st.subheader("Report Configuration")
    col1, col2 = st.columns(2)
    with col1:
        report_title  = st.text_input("Organization / Location Name",
                                       value="Smart Energy Audit — 2024")
        budget_report = st.number_input("Monthly Budget for Report (₹)",
                                         min_value=0.0,
                                         value=st.session_state['budget'], step=100.0)
    with col2:
        use_case = st.selectbox("Use Case", ["Home", "Office", "Industrial", "Energy Auditing"])
        date_range = st.select_slider("Data Period",
                                      options=["Last 30 days", "Last 90 days", "Full Dataset"],
                                      value="Full Dataset")

    # Filter data by range
    if date_range == "Last 30 days":
        df_report = df[df['timestamp'] >= df['timestamp'].max() - timedelta(days=30)]
    elif date_range == "Last 90 days":
        df_report = df[df['timestamp'] >= df['timestamp'].max() - timedelta(days=90)]
    else:
        df_report = df.copy()

    # Preview metrics
    st.markdown("---")
    st.subheader("📊 Report Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total kWh",    f"{df_report['energy_kwh'].sum():.2f}")
    c2.metric("Total Cost",   f"₹{df_report['cost_inr'].sum():.2f}")
    c3.metric("Anomalies",    str(int(df_report['anomaly_pred'].sum()))
                               if 'anomaly_pred' in df_report.columns else "N/A")
    efficiency = max(0, min(100, round(
        100 - (int(df_report.get('anomaly_pred', pd.Series([0])).sum()) / max(len(df_report),1)) * 500, 1
    )))
    c4.metric("Efficiency Score", f"{efficiency}/100")

    # Appliance breakdown preview
    st.markdown("**Estimated Appliance Distribution**")
    app_counts = {}
    for pw in df_report['power_w']:
        for app in detect_appliances(pw):
            app_counts[app['name']] = app_counts.get(app['name'], 0) + 1
    app_prev = pd.DataFrame([{"Appliance": k, "% Time": round(v/len(df_report)*100,1)}
                              for k,v in sorted(app_counts.items(), key=lambda x:-x[1])])
    fig_prev = px.pie(app_prev, names='Appliance', values='% Time',
                      title='Time-based Appliance Distribution',
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_prev, width='stretch')

    st.markdown("---")
    st.subheader("📥 Generate & Download PDF Report")

    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_pdf_report(df_report, budget_report)
                filename  = f"energy_audit_{use_case.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                )
                st.balloons()
            except Exception as e:
                st.error(f"Failed to generate PDF: {e}")

    st.markdown("---")
    st.markdown("""
    **Report includes:**
    - Executive summary (kWh, cost, anomalies, efficiency score)
    - Monthly breakdown table
    - Estimated appliance usage (NILM)
    - Shift-wise analysis
    - Personalized energy-saving recommendations
    - Budget status
    """)
