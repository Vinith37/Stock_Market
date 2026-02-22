import pandas as pd
import numpy as np

# ===== USER-TUNABLE COEFFICIENTS =====
w_trend = 0.4
w_vol = 0.2
w_green = 0.2
w_vwap = 0.2

trend_scale = 0.10      # expected strong trend ~10%
vol_scale = 0.30        # high vol ~30%
vwap_scale = 0.05       # 5% deviation considered large

# ======================================

df = pd.read_csv("TITAN.csv")

price_cols = [
    'OPEN', 'HIGH', 'LOW', 'CLOSE',
    'PREV. CLOSE', 'LTP', 'VWAP',
    '52W H', '52W L', 'VALUE'
]

for col in price_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '', regex=False)
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)
df = df.sort_values('DATE')

df['SMA50'] = df['CLOSE'].rolling(50).mean()
df['SMA200'] = df['CLOSE'].rolling(200).mean()

latest = df.iloc[-1]

# Raw metrics
trend = (latest['SMA50'] - latest['SMA200']) / latest['CLOSE']
df['returns'] = df['CLOSE'].pct_change()
vol = df['returns'].std() * np.sqrt(252)
green_ratio = (df['CLOSE'] > df['OPEN']).mean()
vwap_dev = (latest['CLOSE'] - latest['VWAP']) / latest['VWAP']

# ===== NORMALIZATION =====
T = trend / trend_scale
V = vol / vol_scale
G = (green_ratio - 0.5) / 0.1
W = vwap_dev / vwap_scale

# ===== SCORE =====
score = (
    w_trend * T +
    w_vol * V +
    w_green * G +
    w_vwap * W
)

# ===== STRIKE MAPPING =====
if score > 1.0:
    decision = "Far OTM (5–8%)"
elif 0.3 < score <= 1.0:
    decision = "Slight OTM (3–5%)"
elif -0.3 <= score <= 0.3:
    decision = "ATM"
else:
    decision = "ITM (2–5%)"

# ===== OUTPUT =====
print("Score:", round(score, 3))
print("Strike Decision:", decision)
