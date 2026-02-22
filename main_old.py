import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Load your CSV file
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


# 2️⃣ Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True)

# 3️⃣ Sort by date (important for FFT)
df = df.sort_values('DATE')

# 4️⃣ Create daily candle movement
print(df.dtypes)
df['OPEN'] = pd.to_numeric(df['OPEN'], errors='coerce')
df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')
df = df.dropna(subset=['OPEN', 'CLOSE'])
df['Candle_Move'] = df['CLOSE'] - df['OPEN']

# 5️⃣ Convert to numpy array
series = df['Candle_Move'].values

# 6️⃣ Remove mean (important for spectrum)
series = series - np.mean(series)

# 7️⃣ FFT
fft_vals = np.fft.fft(series)
energy = np.abs(fft_vals)**2
freq = np.fft.fftfreq(len(series), d=1)

# 8️⃣ Plot only positive frequencies
mask = freq > 0
plt.loglog(freq[mask], energy[mask])
plt.xlabel("Frequency")
plt.ylabel("Energy")
plt.title("Energy Spectrum of Daily Candle Movement")
plt.show()
