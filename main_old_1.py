import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ===== 2️⃣ Trend Strength =====
df['SMA50'] = df['CLOSE'].rolling(50).mean()
df['SMA200'] = df['CLOSE'].rolling(200).mean()

latest = df.iloc[-1]

trend = (latest['SMA50'] - latest['SMA200']) / latest['CLOSE']

# ===== 3️⃣ Realized Volatility =====
df['returns'] = df['CLOSE'].pct_change()
vol = df['returns'].std() * np.sqrt(252)

# ===== 4️⃣ Green Candle Ratio =====
green_ratio = (df['CLOSE'] > df['OPEN']).mean()

# ===== 5️⃣ Decision Logic =====

if trend > 0.03 and green_ratio > 0.55:
    decision = "Far OTM Covered Call"
elif -0.03 <= trend <= 0.03 and vol > 0.20:
    decision = "ATM Covered Call"
elif trend < -0.03:
    decision = "ITM Covered Call"
else:
    decision = "Slight OTM Covered Call"

# ===== 6️⃣ Output =====
print("Trend Strength:", round(trend, 3))
print("Annual Volatility:", round(vol, 3))
print("Green Candle Ratio:", round(green_ratio, 3))
print("Recommendation:", decision)

plt.figure()
plt.bar(["Trend Strength"], [trend])
plt.axhline(0.03)
plt.axhline(-0.03)
plt.title("Trend Strength")
plt.ylabel("Value")
plt.show()

plt.figure()
plt.bar(["Annual Volatility"], [vol])
plt.axhline(0.20)
plt.title("Annual Volatility")
plt.ylabel("Volatility")
plt.show()

plt.figure()
plt.bar(["Green Candle Ratio"], [green_ratio])
plt.axhline(0.55)
plt.axhline(0.45)
plt.title("Green Candle Ratio")
plt.ylabel("Ratio")
plt.show()