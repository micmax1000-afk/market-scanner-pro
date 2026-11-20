"""
signals_today.py — "Cosa fare oggi" per le strategie di strategies_pro.py.

Usato sia dall'app (app.py) sia dal messaggio Telegram serale
(scheduled_scan.py), così mostrano sempre le stesse operazioni.

Le posizioni aperte non vengono salvate da nessuna parte: si ricavano
rigiocando le regole sullo storico, che sono deterministiche. Se hai
seguito ogni segnale, quello che vedi qui è esattamente il tuo portafoglio.

Convenzione: il segnale si calcola sulla chiusura di oggi e si esegue
domani all'apertura.
"""

import time

import pandas as pd

from core import TICKER_CATALOG, download_data
from strategies_pro import (
    STRATEGY_INFO, adjust_ohlc, target_rsi2, target_trend10, targets_momentum, monthly_sma_above,
)

REGIME_TICKER = "FTSEMIB.MI"
MOMENTUM_TOP_N = 5


def universe():
    out = {}
    for cat, items in TICKER_CATALOG.items():
        for name, t in items.items():
            out[t] = {"name": name, "category": cat}
    return out


def load_data(tickers, period="2y", retries=3):
    data, missing = {}, []
    for t in tickers:
        df = None
        for attempt in range(retries):
            try:
                raw = download_data(t, period=period, interval="1d")
                if raw is not None and not raw.empty:
                    df = adjust_ohlc(raw)
                    break
            except Exception as e:
                print(f"{t}: {e}")
            time.sleep(2 * (attempt + 1))
        if df is None or len(df) < 260:
            missing.append(t)
        else:
            data[t] = df
    return data, missing


def _entry_info(df, tgt):
    held = tgt.shift(1).fillna(0)
    starts = held.index[(held.diff() == 1).to_numpy()]
    if len(starts) == 0:
        return None, None
    entry = starts[-1]
    price = float(df.loc[entry, "Open"])
    return entry, (float(df["Close"].iloc[-1]) / price - 1) * 100


def _action(tgt):
    """Da target (deciso a chiusura) all'azione per domani."""
    now, before = tgt.iloc[-1], tgt.iloc[-2]
    if now == 1 and before == 0:
        return "COMPRA"
    if now == 0 and before == 1:
        return "VENDI"
    return "TIENI" if now == 1 else None


def signals_rsi2(data, uni):
    rows = []
    for t, df in data.items():
        tgt, ind = target_rsi2(df)
        act = _action(tgt)
        if act is None:
            continue
        last = ind.iloc[-1]
        entry, ret = _entry_info(df, tgt) if act in ("TIENI", "VENDI") else (None, None)
        if act == "COMPRA":
            why = f"RSI(2) a {last['RSI2']:.1f}, sopra la media a 200 giorni"
        elif act == "VENDI":
            why = "chiusura sopra la media a 5 giorni"
        else:
            why = f"vendi alla prima chiusura sopra {last['SMA5']:.2f}"
        rows.append({"strategy": "mean_reversion_rsi2", "ticker": t, "name": uni[t]["name"],
                     "category": uni[t]["category"], "action": act, "why": why,
                     "close": float(df["Close"].iloc[-1]), "date": df.index[-1].date(),
                     "since": entry.date() if entry is not None else None, "open_ret_pct": ret})
    return rows


def signals_trend10(data, uni):
    rows = []
    for t, df in data.items():
        if uni[t]["category"] not in ("Indici Americani", "Indici Europei"):
            continue
        tgt = target_trend10(df)
        if len(tgt) < 2:
            continue
        act = _action(tgt) or "LIQUIDITÀ"
        m_close = df["Close"][monthly_sma_above(df["Close"]).notna()]
        sma = pd.concat([m_close.iloc[-9:], df["Close"].iloc[-1:]]).mean()
        close = float(df["Close"].iloc[-1])
        entry, ret = _entry_info(df, tgt) if act in ("TIENI", "VENDI") else (None, None)
        rows.append({"strategy": "trend_10_mesi", "ticker": t, "name": uni[t]["name"],
                     "category": uni[t]["category"], "action": act,
                     "why": f"chiusura {'sopra' if close > sma else 'sotto'} la media a 10 mesi ({sma:,.0f})",
                     "close": close, "date": df.index[-1].date(),
                     "since": entry.date() if entry is not None else None, "open_ret_pct": ret,
                     "distance_pct": (close / sma - 1) * 100})
    return rows


def signals_momentum(data, uni):
    it = [t for t in data if uni[t]["category"] == "Azioni Italiane"]
    if REGIME_TICKER not in data or len(it) < MOMENTUM_TOP_N:
        return [], {}
    closes = pd.DataFrame({t: data[t]["Close"] for t in it}).sort_index()
    regime = monthly_sma_above(data[REGIME_TICKER]["Close"].reindex(closes.index).ffill())
    tgt, mom = targets_momentum(closes, regime, top_n=MOMENTUM_TOP_N)
    now, before = tgt.iloc[-1], tgt.iloc[-2]
    # classifica con i prezzi di oggi (anteprima della prossima scelta)
    rank = (closes.iloc[-22] / closes.iloc[-253] - 1).dropna().sort_values(ascending=False) if len(closes) > 253 else pd.Series(dtype=float)
    rows = []
    for t in it:
        a = None
        if now[t] == 1 and before[t] == 0:
            a = "COMPRA"
        elif now[t] == 0 and before[t] == 1:
            a = "VENDI"
        elif now[t] == 1:
            a = "TIENI"
        if a is None:
            continue
        df = data[t]
        entry, ret = _entry_info(df, tgt[t].reindex(df.index).ffill().fillna(0)) if a != "COMPRA" else (None, None)
        r12 = rank.get(t)
        rows.append({"strategy": "momentum_12_1", "ticker": t, "name": uni[t]["name"],
                     "category": uni[t]["category"], "action": a,
                     "why": f"rendimento 12-1 mesi {r12 * 100:+.0f}%" if r12 is not None else "",
                     "close": float(df["Close"].iloc[-1]), "date": df.index[-1].date(),
                     "since": entry.date() if entry is not None else None, "open_ret_pct": ret})
    reg_close = data[REGIME_TICKER]["Close"]
    m_close = reg_close[monthly_sma_above(reg_close).notna()]
    sma = pd.concat([m_close.iloc[-9:], reg_close.iloc[-1:]]).mean()
    extra = {
        "regime_ok": bool(reg_close.iloc[-1] > sma),
        "regime_text": f"FTSE MIB {'sopra' if reg_close.iloc[-1] > sma else 'sotto'} la media a 10 mesi",
        "ranking": [{"ticker": t, "name": uni[t]["name"], "mom_pct": float(v) * 100} for t, v in rank.items()],
    }
    return rows, extra


def all_signals(period="2y"):
    uni = universe()
    data, missing = load_data(list(uni), period=period)
    rows = signals_rsi2(data, uni) + signals_trend10(data, uni)
    mrows, mextra = signals_momentum(data, uni)
    rows += mrows
    last_date = max((df.index[-1] for df in data.values()), default=None)
    return {"rows": rows, "momentum": mextra, "missing": missing,
            "last_date": last_date.date() if last_date is not None else None,
            "info": STRATEGY_INFO, "data": data, "universe": uni}


ACTION_ORDER = {"COMPRA": 0, "VENDI": 1, "TIENI": 2, "LIQUIDITÀ": 3}
