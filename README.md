# Smart IoT Energy Meter — AI/ML Dashboard

> **Group 6 | Problem Statement:** Consumers often lack visibility into electricity consumption of individual appliances, making energy conservation difficult.

**Objective:** Monitor power consumption and generate alerts for excessive energy usage.

**Hardware:** ESP32 + ACS712 (current) + ZMPT101B (voltage) + Blynk 2.0 + Telegram Bot

---

## Features

| Page | Description |
|------|-------------|
| 📊 Dashboard | Live metrics, appliance cards, budget banner, threshold alerts, 7-day trend |
| 🔌 Appliance Detector | Rule-based NILM — detects Fan, AC, Geyser, TV etc. from power reading |
| 🚨 Anomaly Detector | Isolation Forest ML — voltage spikes, current surges, power dropouts |
| 🔮 Energy Forecast | XGBoost ML — predicts next 15-min energy (kWh), R² = 0.9982 |
| 📈 Analytics | Monthly summary, anomaly timeline, cost analysis, shift-wise breakdown |
| 🔔 Alerts & Budget | Set ₹ budget, power threshold, email alerts, efficiency score, recommendations |
| 📄 Audit Report | One-click PDF — executive summary, appliance usage, recommendations |

---

## ML Models

| Model | Purpose | Performance |
|-------|---------|-------------|
| XGBoost Regressor | Predict energy consumption (kWh) per 15-min interval | R² = 0.9982, MAE = 0.00067 kWh |
| Isolation Forest | Detect anomalies in sensor readings | 2% contamination rate |

---

## Dataset

- **Source:** Synthetically generated to match ESP32 + ACS712 + ZMPT101B sensor specs
- **Period:** 2025-01-01 to 2025-06-29 (6 months)
- **Frequency:** Every 15 minutes
- **Total records:** 17,280 rows
- **Features:** voltage_v, current_a, power_w, energy_kwh, power_factor, temperature_c, cost_inr

---

## Project Structure

```
Smart Energy/
├── streamlit_app.py       # Main Streamlit dashboard (7 pages)
├── app.py                 # FastAPI backend (optional)
├── train_models.py        # Generate data + train + save models
├── energy_meter_ml.ipynb  # Full ML pipeline notebook
├── requirements.txt       # Python dependencies
└── README.md
```

> `*.pkl` and `*.csv` files are excluded from the repo (generated locally by `train_models.py`)

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/swadhin2006/Smart_Energy.git
cd Smart_Energy
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate data and train models
```bash
python train_models.py
```

### 5. Run Streamlit dashboard
```bash
streamlit run streamlit_app.py
```

Open → **http://localhost:8501**

### 6. (Optional) Run FastAPI backend
```bash
uvicorn app:app --reload
```
API docs → **http://localhost:8000/docs**

---

## Use Cases Covered

- 🏠 **Homes** — Appliance detection, peak hour alerts, monthly cost tracking
- 🏢 **Offices** — Shift-wise (Day/Evening/Night) energy breakdown
- 🏭 **Industries** — Anomaly detection, audit PDF, threshold monitoring
- 📋 **Energy Auditing** — Full PDF report with efficiency score and recommendations

---

## Tech Stack

- **Frontend:** Streamlit
- **ML:** XGBoost, Scikit-learn (Isolation Forest)
- **Backend:** FastAPI + Uvicorn
- **Data:** Pandas, NumPy
- **Charts:** Plotly
- **PDF:** fpdf2
- **Hardware:** ESP32, ACS712, ZMPT101B, Blynk 2.0
