"""
Run this script to generate data and train models.
Usage: python train_models.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import joblib, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
print("Generating data...")

def generate_energy_data(n_days=180, freq_minutes=15):
    periods = int(n_days * 24 * 60 / freq_minutes)
    timestamps = pd.date_range(start='2025-01-01', periods=periods, freq=f'{freq_minutes}min')
    hour  = timestamps.hour.to_numpy()
    dow   = timestamps.dayofweek.to_numpy()
    month = timestamps.month.to_numpy()

    voltage = 220 + np.random.normal(0, 3, periods) + 2 * np.sin(2 * np.pi * hour / 24)
    base_current = (
        0.5
        + 1.5 * ((hour >= 7)  & (hour <= 9)).astype(float)
        + 2.0 * ((hour >= 18) & (hour <= 22)).astype(float)
        + 0.3 * ((dow >= 5)).astype(float)
        + 0.2 * ((month >= 5) & (month <= 8)).astype(float)
    )
    current      = base_current + np.abs(np.random.normal(0, 0.3, periods))
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
        elif atype == 'dropout':
            voltage[idx] = 0.0
            current[idx] = 0.0

    return pd.DataFrame({
        'timestamp':    timestamps,
        'voltage_v':    np.round(voltage, 2),
        'current_a':    np.round(current, 3),
        'power_w':      np.round(power, 2),
        'energy_kwh':   np.round(energy_kwh, 4),
        'power_factor': np.round(power_factor, 3),
        'temperature_c':np.round(temperature, 1),
        'cost_inr':     np.round(cost, 4),
        'is_anomaly':   is_anomaly
    })

def engineer_features(df):
    df = df.copy()
    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month']       = df['timestamp'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_peak_hour']= df['hour'].isin([7,8,9,18,19,20,21,22]).astype(int)
    df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
    df['apparent_power']  = df['voltage_v'] * df['current_a']
    df['reactive_power']  = df['apparent_power'] * np.sqrt(np.maximum(1 - df['power_factor']**2, 0))
    df['power_roll_mean_4']  = df['power_w'].rolling(4,  min_periods=1).mean()
    df['power_roll_mean_24'] = df['power_w'].rolling(24, min_periods=1).mean()
    df['power_roll_std_4']   = df['power_w'].rolling(4,  min_periods=1).std().fillna(0)
    df['energy_lag_1']  = df['energy_kwh'].shift(1).fillna(0)
    df['energy_lag_4']  = df['energy_kwh'].shift(4).fillna(0)
    df['energy_lag_96'] = df['energy_kwh'].shift(96).fillna(0)
    return df

# Generate & engineer
df      = generate_energy_data()
df_feat = engineer_features(df)
df.to_csv('energy_data.csv', index=False)
print(f"Data saved: {df.shape}")

# ── Forecast model ────────────────────────────────────────────────────────────
print("Training XGBoost forecast model...")
FORECAST_FEATURES = [
    'voltage_v','current_a','power_factor','temperature_c',
    'hour_sin','hour_cos','month_sin','month_cos',
    'is_weekend','is_peak_hour','apparent_power','reactive_power',
    'power_roll_mean_4','power_roll_mean_24','power_roll_std_4',
    'energy_lag_1','energy_lag_4','energy_lag_96'
]
X = df_feat[FORECAST_FEATURES].values
y = df_feat['energy_kwh'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                   verbosity=0)
xgb.fit(X_train_s, y_train)

from sklearn.metrics import mean_absolute_error, r2_score
y_pred = xgb.predict(X_test_s)
print(f"  MAE:  {mean_absolute_error(y_test, y_pred):.5f} kWh")
print(f"  R²:   {r2_score(y_test, y_pred):.4f}")

joblib.dump(xgb,              'energy_forecast_model.pkl')
joblib.dump(scaler,           'energy_forecast_scaler.pkl')
joblib.dump(FORECAST_FEATURES,'forecast_features.pkl')
print("  Saved: energy_forecast_model.pkl")

# ── Anomaly model ─────────────────────────────────────────────────────────────
print("Training Isolation Forest anomaly model...")
ANOMALY_FEATURES = ['voltage_v','current_a','power_w','power_factor',
                    'apparent_power','reactive_power','power_roll_std_4']

X_anom    = df_feat[ANOMALY_FEATURES].values
anom_sc   = StandardScaler()
X_anom_s  = anom_sc.fit_transform(X_anom)

iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
iso.fit(X_anom_s)

raw_pred  = iso.predict(X_anom_s)
anom_pred = (raw_pred == -1).astype(int)
scores    = -iso.score_samples(X_anom_s)

df_feat['anomaly_pred']  = anom_pred
df_feat['anomaly_score'] = np.round(scores, 4)
df_feat['year_month']    = df_feat['timestamp'].dt.to_period('M').astype(str)

df_feat.to_csv('energy_data_processed.csv', index=False)

joblib.dump(iso,             'anomaly_model.pkl')
joblib.dump(anom_sc,         'anomaly_scaler.pkl')
joblib.dump(ANOMALY_FEATURES,'anomaly_features.pkl')
print(f"  Anomalies detected: {anom_pred.sum()} ({anom_pred.mean()*100:.1f}%)")
print("  Saved: anomaly_model.pkl")

print("\nAll done! Models and data files ready.")
print("Files created:")
import os
for f in ['energy_data.csv','energy_data_processed.csv',
          'energy_forecast_model.pkl','energy_forecast_scaler.pkl',
          'forecast_features.pkl','anomaly_model.pkl',
          'anomaly_scaler.pkl','anomaly_features.pkl']:
    size = os.path.getsize(f) // 1024
    print(f"  {f}  ({size} KB)")
