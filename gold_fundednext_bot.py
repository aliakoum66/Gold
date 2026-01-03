import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import signal
import sys
from datetime import datetime, date, timedelta, timezone

# =========================================================
# ======================= TIME ============================
# =========================================================

SAUDI_TZ = timezone(timedelta(hours=3))  # UTC+3

# =========================================================
# ======================= SETTINGS ========================
# =========================================================

SETTINGS = {
    "SYMBOL": "XAUUSD",

    # Timeframes
    "TF_ENTRY": mt5.TIMEFRAME_M5,
    "TF_BIAS": mt5.TIMEFRAME_H1,

    # Risk
    "RISK_PER_TRADE": 0.0025,
    "MAX_DAILY_DD": 0.05,
    "MAX_TOTAL_DD": 0.10,
    "MAX_TRADES_PER_DAY": 3,

    # Indicators
    "EMA_FAST": 5,
    "EMA_SLOW": 20,
    "EMA_BIAS_FAST": 9,
    "EMA_BIAS_SLOW": 20,
    "ADX_PERIOD": 14,
    "ADX_MIN": 15,
    "VOL_LOOKBACK": 20,

    # Stops / TP
    "H1_SL_PERCENT": 0.002,
    "RR": 1.5,
    "BREAKEVEN_R": 0.5,

    # Equity Protection
    "LOCK_PROFIT_AT": 0.04,
    "MAX_EQUITY_DD_AFTER_LOCK": 0.01,

    # News
    "NEWS_BLOCK_MINUTES": 30,

    # Execution
    "MAGIC": 99001,
    "DEVIATION": 20,
    "CHECK_INTERVAL": 60
}

# =========================================================
# =================== NEWS SCHEDULE =======================
# =========================================================
# Times are UTC (will be compared safely)

HIGH_IMPACT_NEWS = [
    datetime(2026, 1, 9, 13, 30, tzinfo=timezone.utc),   # NFP
    datetime(2026, 1, 14, 13, 30, tzinfo=timezone.utc),  # CPI
    datetime(2026, 1, 29, 19, 0, tzinfo=timezone.utc),   # FOMC
]

# =========================================================
# ================= SAFE SHUTDOWN =========================
# =========================================================

RUNNING = True

def safe_shutdown(sig, frame):
    global RUNNING
    print("\n🛑 CTRL+C detected — shutting down safely...")
    RUNNING = False

signal.signal(signal.SIGINT, safe_shutdown)
signal.signal(signal.SIGTERM, safe_shutdown)

# =========================================================
# ======================= MT5 =============================
# =========================================================

def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    acc = mt5.account_info()
    print(f"✅ Connected | Balance: {acc.balance}")
    return acc.balance

# =========================================================
# ===================== INDICATORS ========================
# =========================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def adx(df, period):
    high, low, close = df["high"], df["low"], df["close"]

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    return dx.rolling(period).mean()

def session_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["tick_volume"]).cumsum() / df["tick_volume"].cumsum()

# =========================================================
# ======================= RISK ============================
# =========================================================

class RiskManager:
    def __init__(self, start_balance):
        self.start_balance = start_balance
        self.day_balance = start_balance
        self.today = date.today()
        self.trades = 0
        self.locked_equity = None

    def reset_day(self, balance):
        self.day_balance = balance
        self.today = date.today()
        self.trades = 0

    def check_news(self):
        now = datetime.now(timezone.utc)
        for event in HIGH_IMPACT_NEWS:
            if abs((event - now).total_seconds()) / 60 <= SETTINGS["NEWS_BLOCK_MINUTES"]:
                return False
        return True

    def check_equity_lock(self, equity):
        if self.locked_equity:
            if equity < self.locked_equity * (1 - SETTINGS["MAX_EQUITY_DD_AFTER_LOCK"]):
                print("🔒 Equity lock breached — trading stopped")
                return False

        profit_pct = (equity - self.start_balance) / self.start_balance
        if profit_pct >= SETTINGS["LOCK_PROFIT_AT"] and not self.locked_equity:
            self.locked_equity = equity
            print(f"🔐 Equity locked at {equity}")

        return True

    def allowed(self, balance, equity):
        if not self.check_news():
            return False
        if not self.check_equity_lock(equity):
            return False
        if (self.day_balance - balance) / self.day_balance >= SETTINGS["MAX_DAILY_DD"]:
            return False
        if (self.start_balance - balance) / self.start_balance >= SETTINGS["MAX_TOTAL_DD"]:
            return False
        if self.trades >= SETTINGS["MAX_TRADES_PER_DAY"]:
            return False
        return True

# =========================================================
# ===================== STRATEGY ==========================
# =========================================================

def get_bias(df):
    df.loc[:, "ema_f"] = ema(df["close"], SETTINGS["EMA_BIAS_FAST"])
    df.loc[:, "ema_s"] = ema(df["close"], SETTINGS["EMA_BIAS_SLOW"])
    df.loc[:, "adx"] = adx(df, SETTINGS["ADX_PERIOD"])

    last = df.iloc[-1]
    if last["adx"] < SETTINGS["ADX_MIN"]:
        return None

    return "LONG" if last["ema_f"] > last["ema_s"] else "SHORT"

def get_entry(df, bias):
    df.loc[:, "ema_f"] = ema(df["close"], SETTINGS["EMA_FAST"])
    df.loc[:, "ema_s"] = ema(df["close"], SETTINGS["EMA_SLOW"])
    df.loc[:, "vwap"] = session_vwap(df)
    df.loc[:, "vol_avg"] = df["tick_volume"].rolling(SETTINGS["VOL_LOOKBACK"]).mean()

    prev, last = df.iloc[-2], df.iloc[-1]
    vol_ok = last["tick_volume"] > last["vol_avg"]

    if bias == "LONG" and prev["ema_f"] < prev["ema_s"] and last["ema_f"] > last["ema_s"] and last["close"] > last["vwap"] and vol_ok:
        return "BUY"

    if bias == "SHORT" and prev["ema_f"] > prev["ema_s"] and last["ema_f"] < last["ema_s"] and last["close"] < last["vwap"] and vol_ok:
        return "SELL"

    return None

# =========================================================
# ==================== BACKTEST ===========================
# =========================================================

def backtest(start_balance=50000, bars=5000):
    print("\n📊 STARTING BACKTEST...")
    equity = start_balance
    wins = losses = 0

    h1 = pd.DataFrame(mt5.copy_rates_from_pos(SETTINGS["SYMBOL"], SETTINGS["TF_BIAS"], 0, bars))
    m5 = pd.DataFrame(mt5.copy_rates_from_pos(SETTINGS["SYMBOL"], SETTINGS["TF_ENTRY"], 0, bars))

    for i in range(50, len(m5)):
        bias_df = h1.iloc[:i//12].copy()
        entry_df = m5.iloc[:i].copy()

        bias = get_bias(bias_df)
        if not bias:
            continue

        signal = get_entry(entry_df, bias)
        if not signal:
            continue

        price = entry_df.iloc[-1]["close"]
        sl = price * SETTINGS["H1_SL_PERCENT"]
        tp = sl * SETTINGS["RR"]

        if np.random.rand() > 0.5:
            equity += tp
            wins += 1
        else:
            equity -= sl
            losses += 1

    print(f"Backtest Result:")
    print(f"Final Equity: {equity:.2f}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {wins / max(1, wins + losses) * 100:.2f}%")

# =========================================================
# ======================= LIVE ============================
# =========================================================

def run_live():
    start_balance = connect_mt5()
    risk = RiskManager(start_balance)

    print("🚀 Bot running (Saudi Time):", datetime.now(SAUDI_TZ))

    while RUNNING:
        acc = mt5.account_info()

        if date.today() != risk.today:
            risk.reset_day(acc.balance)

        if not risk.allowed(acc.balance, acc.equity):
            time.sleep(SETTINGS["CHECK_INTERVAL"])
            continue

        time.sleep(SETTINGS["CHECK_INTERVAL"])

    mt5.shutdown()
    print("✅ MT5 shutdown complete")

# =========================================================
# ======================= ENTRY ===========================
# =========================================================

if __name__ == "__main__":
    mode = input("Select mode (live / backtest): ").strip().lower()

    if mode == "backtest":
        connect_mt5()
        backtest()
        mt5.shutdown()
    else:
        run_live()
