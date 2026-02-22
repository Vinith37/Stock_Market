import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

# --- Trend ---
df['SMA50'] = df['CLOSE'].rolling(50).mean()
df['SMA200'] = df['CLOSE'].rolling(200).mean()
df['Trend'] = (df['SMA50'] - df['SMA200']) / df['CLOSE']

# --- Volatility ---
df['Return'] = df['CLOSE'].pct_change()
df['Vol20'] = df['Return'].rolling(20).std() * np.sqrt(252)

# --- Green Ratio ---
df['Green'] = (df['CLOSE'] > df['OPEN']).astype(int)
df['Green20'] = df['Green'].rolling(20).mean()

# --- VWAP Deviation ---
df['VWAP_Dev'] = (df['CLOSE'] - df['VWAP']) / df['VWAP']

# --- Target Variable ---
df['Forward_Return_20d'] = df['CLOSE'].shift(-20) / df['CLOSE'] - 1

# Drop NA rows
df = df.dropna()


X = df[['Trend', 'VWAP_Dev']]   # choose features
y = df['Forward_Return_20d']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(df.head())


import statsmodels.api as sm

X_scaled = sm.add_constant(X_scaled)

model = sm.OLS(y, X_scaled).fit(cov_type='HC3')
print(model.summary())

