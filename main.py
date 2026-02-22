import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ===== LOAD DATA =====
df = pd.read_excel("TITAN.xlsx")

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

# ===== FEATURE ENGINEERING =====
df['SMA50'] = df['CLOSE'].rolling(50).mean()
df['SMA200'] = df['CLOSE'].rolling(200).mean()
df['Trend'] = (df['SMA50'] - df['SMA200']) / df['CLOSE']

df['Return'] = df['CLOSE'].pct_change()
df['Vol20'] = df['Return'].rolling(20).std() * np.sqrt(252)

df['VWAP_Dev'] = (df['CLOSE'] - df['VWAP']) / df['VWAP']

df['Forward_Return_20d'] = df['CLOSE'].shift(-20) / df['CLOSE'] - 1

# ===== REGIME LABEL =====
df['Regime'] = 1  # default sideways

df.loc[df['Trend'] > 0.03, 'Regime'] = 2   # Uptrend
df.loc[df['Trend'] < -0.03, 'Regime'] = 0  # Downtrend

# ===== LOGISTIC TARGET =====
df['Breakout'] = (df['Forward_Return_20d'] > 0.05).astype(int)

df = df.dropna()

# ===== FEATURES =====
features = ['Trend', 'Vol20', 'VWAP_Dev']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== REGIME CLASSIFIER =====
regime_model = RandomForestClassifier(n_estimators=200, random_state=42)
regime_model.fit(X_scaled, df['Regime'])

# ===== LOGISTIC MODEL =====
log_model = LogisticRegression()
log_model.fit(X_scaled, df['Breakout'])

# ===== CURRENT MARKET STATE =====
latest_features = scaler.transform(df[features].iloc[-1:])

current_regime = regime_model.predict(latest_features)[0]
breakout_prob = log_model.predict_proba(latest_features)[0][1]

# ===== STRIKE DECISION ENGINE =====

decision = "No Decision"  # safety default

if current_regime == 2:  # Uptrend
    
    if breakout_prob > df['Breakout'].mean():
        breakout_state = "Elevated"
        decision = "Far OTM (6–8%)"
    else:
        breakout_state = "Below Normal"
        decision = "Slight OTM (3–5%)"

elif current_regime == 1:  # Sideways
    
    breakout_state = "Normal"
    decision = "ATM (0–2%)"

else:  # Downtrend
    
    breakout_state = "N/A"
    decision = "ITM (2–5%)"

# ===== OUTPUT =====
regime_map = {0: "Downtrend", 1: "Sideways", 2: "Uptrend"}

print("Current Regime:", regime_map[current_regime])
print("Breakout Probability (>5% in 20d):", round(breakout_prob, 3))
print("Breakout State:", breakout_state)
print("Strike Recommendation:", decision)
print("Average Breakout Probability (>5% in 20d):", df['Breakout'].mean())