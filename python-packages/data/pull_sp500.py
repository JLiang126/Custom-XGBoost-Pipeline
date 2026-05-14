import yfinance as yf
import pandas as pd
import numpy as np

print("Downloading S&P 500 data...")
df = yf.download("^GSPC", start="1986-01-01", end="2017-01-1")

df.columns = df.columns.get_level_values(0)

# 1. Log Returns (Better for ML than raw prices)
df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

# 2. Indicators
df['volatility'] = df['returns'].rolling(window=10).std()
df['ma_20'] = df['Close'].rolling(window=20).mean()
df['dist_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']

# RSI Calculation
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
df['rsi'] = 100 - (100 / (1 + (gain / loss)))

# 3. Features: 5 days of lagged RETURNS + Technicals
feature_cols = []
for i in range(1, 6):
    col = f'ret_lag_{i}'
    df[col] = df['returns'].shift(i)
    feature_cols.append(col)

# Add the indicators to the feature list
feature_cols += ['volatility', 'dist_ma20', 'rsi']

# 4. Target: Tomorrow's Return
df['target'] = df['returns'].shift(-1)

# 5. Cleanup & Save
# We drop the first 20 rows (MA/RSI warmup) and last row (target shift)
final_df = df[feature_cols + ['target']].dropna()
final_df.to_csv('sp500_train.csv', index=False)

print(f"Done! Created 'sp500_test.csv' with {len(final_df)} rows and {len(feature_cols)} features.")
print("Features:", feature_cols)