import pandas as pd
import numpy as np

# ===== 1️⃣ Load Data =====
df = pd.read_csv("your_stock_data.csv")

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

# ===== 2️⃣ Core Metrics =====
df['SMA50'] = df['CLOSE'].rolling(50).mean()
df['SMA200'] = df['CLOSE'].rolling(200).mean()

latest = df.iloc[-1]

trend = (latest['SMA50'] - latest['SMA200']) / latest['CLOSE']
df['returns'] = df['CLOSE'].pct_change()
vol = df['returns'].std() * np.sqrt(252)
green_ratio = (df['CLOSE'] > df['OPEN']).mean()

# ===== 3️⃣ VWAP Deviation =====
vwap_dev = (latest['CLOSE'] - latest['VWAP']) / latest['VWAP']

# ===== 4️⃣ Strike Decision Logic =====

if trend > 0.05 and vwap_dev > 0.02:
    strike_type = "Slight OTM"
    strike_distance = "3%–4% OTM"
elif trend > 0.05 and vwap_dev <= 0.02:
    strike_type = "Far OTM"
    strike_distance = "5%–7% OTM"
elif -0.03 <= trend <= 0.03:
    strike_type = "ATM"
    strike_distance = "0%–2% OTM"
elif trend < -0.03:
    strike_type = "ITM"
    strike_distance = "2%–5% ITM"
else:
    strike_type = "Moderate OTM"
    strike_distance = "3%–5% OTM"

# ===== 5️⃣ Risk Alert =====
if latest['CLOSE'] < latest['VWAP']:
    risk_note = "⚠ Price below VWAP — institutional weakness"
else:
    risk_note = "Structure healthy above VWAP"

# ===== 6️⃣ Output =====
print("Trend Strength:", round(trend, 3))
print("Annual Volatility:", round(vol, 3))
print("Green Candle Ratio:", round(green_ratio, 3))
print("VWAP Deviation:", round(vwap_dev, 3))
print("\nRecommended Strike Type:", strike_type)
print("Suggested Distance:", strike_distance)
print("Risk Note:", risk_note)
