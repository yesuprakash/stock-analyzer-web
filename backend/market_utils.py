import pandas as pd
import yfinance as yf

# ------------------------
# SAFETY
# ------------------------
def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (pd.Series, pd.DataFrame)):
            x = x.iloc[-1]
        return float(x)
    except Exception:
        return None

# ------------------------
# PRICE FETCH
# ------------------------
def fetch_price_df(symbol, period="120d", auto_adjust=False):
    try:
        df = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=auto_adjust
        )
        return normalize_price_df(df)
    except Exception:
        return pd.DataFrame()


def normalize_price_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        price_columns = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        flattened = []
        for column in df.columns:
            parts = [str(part) for part in column if part is not None]
            price_column = next(
                (part for part in parts if part in price_columns),
                parts[0] if parts else "",
            )
            flattened.append(price_column)
        df.columns = flattened
        df = df.loc[:, ~df.columns.duplicated(keep="first")]
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df

# ------------------------
# INDICATORS
# ------------------------
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd_hist(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal

def calc_atr(df, period=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1/period, adjust=False).mean()
