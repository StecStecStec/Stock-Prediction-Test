import yfinance as yf
import pandas as pd
import pandas_ta as ta
import datetime
import numpy as np

TICKER = "QQQ"
START_DATE = datetime.datetime(2015, 1, 1)
df = yf.download(TICKER, start=START_DATE, interval="1d")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df.rename(columns={'Close': 'Adj Close'}, inplace=True)
df.columns = [col.lower().replace('adj close', 'close') for col in df.columns]

df['previous_close'] = df['close'].shift(1)
df['log_return'] = np.log(df['close'] / df['previous_close'])

THRESHOLD = 0.005

def categorize_move(log_return, threshold):
    if log_return > threshold:
        return 'Up'
    elif log_return < -threshold:
        return 'Down'
    else:
        return 'Neutral'

df['move_category'] = df['log_return'].apply(lambda x: categorize_move(x, THRESHOLD))

df.ta.sma(close='log_return', length=20, append=True, prefix='logr')
df.ta.ema(close='log_return', length=20, append=True, prefix='logr')
df.ta.bbands(close='log_return', length=20, append=True, prefix='logr')
df['LogR_Vol_5'] = df['log_return'].rolling(window=5).std()
df.ta.sma(close='log_return', length=5, append=True, prefix='logr')
df['LogR_vs_MA_5'] = df['log_return'] - df['logr_SMA_5']
df.ta.atr(length=14, append=True)
df.ta.rsi(length=14, append=True)
df.ta.macd(append=True)
df.ta.stoch(append=True)

df["daily_change_pct"] = (df["close"] - df["previous_close"]) / df["previous_close"]
df["interday_change"] = (df["close"] - df["open"]) / df["open"]
df["daily_range_pct"] = (df["high"] - df["low"]) / df["previous_close"]

move_dummies = pd.get_dummies(df['move_category'], prefix='label', dtype=int)
move_dummies.rename(columns={'label_Up': 'up', 'label_Down': 'down', 'label_Neutral': 'neutral'}, inplace=True)
df = pd.concat([df, move_dummies], axis=1)

COLUMNS_TO_DROP = ['open', 'high', 'low', 'close', 'previous_close', 'move_category']
result = df.drop(columns=COLUMNS_TO_DROP).dropna()

result.to_csv("market_data_3class_final.csv", index=False)