"""
backtest_v2.py — Backtest realistico delle strategie di core.py

Differenze rispetto al backtest SL/TP originale (core.backtest_strategy_sl_tp),
che resta disponibile per confronto ("metodo vecchio"):

1. Ingresso all'APERTURA del giorno DOPO il segnale (l'alert arriva a
   mercato chiuso: nella realtà non puoi comprare alla chiusura del
   giorno in cui il segnale si forma).
2. Costi: commissione + slippage su ogni lato del trade (default 0,15%
   per lato, 0,30% andata e ritorno).
3. Gap: se il titolo apre già sotto lo stop (o sopra il target), si esce
   al prezzo di apertura, non al livello teorico.
4. Un solo trade alla volta per titolo: finché un trade è aperto, i nuovi
   segnali sullo stesso titolo vengono ignorati. Prima un segnale che
   restava attivo 5 giorni di fila contava come 5 trade quasi identici.
5. Divisione dello storico in due periodi (in-sample / out-of-sample),
   per vedere se il risultato regge su anni diversi o era fortuna.

Le strategie sono quelle di core.py, chiamate giorno per giorno solo con
i dati disponibili fino a quel giorno (nessun dato futuro).
"""

import math

import numpy as np
import pandas as pd

from core import (
    compute_indicators,
    compute_atr,
    compute_trade_signal,
    ALERT_STRATEGY_CONFIG,
)


def _trade_signal_long(window):
    # il segnale combinato può dare anche VENDITA: qui testiamo solo il lato
    # acquisto, come tutte le altre strategie
    return {"signal": compute_trade_signal(window)["signal"]}


def _always_buy(window):
    return {"signal": "ACQUISTO"}


# chiave -> (nome leggibile, funzione)
STRATEGIES = {
    "trade_signal": ("Segnale combinato (rottura + 3 conferme)", _trade_signal_long),
    # riferimento: entrare a caso (qui: appena possibile, sempre). Con le
    # stesse regole di stop/target/costi mostra quanto vale "nessun
    # vantaggio": una strategia utile deve fare MEGLIO di questa riga.
    "baseline": ("Riferimento: entrata sempre (nessun segnale)", _always_buy),
}

DEFAULT_PARAMS = {
    "atr_sl_mult": 1.5,
    "atr_tp_mult": 3.0,
    "max_holding_days": 20,
    "cost_per_side": 0.0015,  # 0,10% commissione + 0,05% slippage
    "min_history": 200,
}


def compute_signal_series(data_ind, strategy_fn, min_history=200):
    """Per ogni giorno i (da min_history in poi) applica la strategia ai
    soli dati fino a i inclusi. Restituisce un array booleano."""
    n = len(data_ind)
    signals = np.zeros(n, dtype=bool)
    if strategy_fn is _always_buy:
        signals[min_history:] = True
        return signals
    for i in range(min_history, n):
        window = data_ind.iloc[:i + 1]
        try:
            res = strategy_fn(window)
        except Exception:
            continue
        signals[i] = res.get("signal") == "ACQUISTO"
    return signals


def simulate_realistic(data_ind, atr, signals, atr_sl_mult=1.5, atr_tp_mult=3.0,
                       max_holding_days=20, cost_per_side=0.0015, **_):
    o = data_ind["Open"].to_numpy(dtype=float)
    h = data_ind["High"].to_numpy(dtype=float)
    l = data_ind["Low"].to_numpy(dtype=float)
    c = data_ind["Close"].to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)
    idx = data_ind.index
    n = len(c)
    trades = []
    i = 0
    while i < n:
        if not signals[i]:
            i += 1
            continue
        e = i + 1  # giorno di ingresso
        if e >= n or not np.isfinite(a[i]) or a[i] <= 0 or not np.isfinite(o[e]) or o[e] <= 0:
            i += 1
            continue
        entry = o[e]
        risk = atr_sl_mult * a[i]
        stop = entry - risk
        target = entry + atr_tp_mult * a[i]
        last_day = e + max_holding_days - 1
        if last_day >= n:
            break  # trade non concluso a fine dati: non lo contiamo
        outcome, exit_price, exit_day = None, None, None
        for j in range(e, last_day + 1):
            if j > e:
                if o[j] <= stop:
                    outcome, exit_price, exit_day = "stop_loss", o[j], j
                    break
                if o[j] >= target:
                    outcome, exit_price, exit_day = "take_profit", o[j], j
                    break
            if l[j] <= stop:  # se tocca entrambi lo stesso giorno: prudenza, stop
                outcome, exit_price, exit_day = "stop_loss", stop, j
                break
            if h[j] >= target:
                outcome, exit_price, exit_day = "take_profit", target, j
                break
        if outcome is None:
            outcome, exit_price, exit_day = "timeout", c[last_day], last_day
        net_pnl = exit_price * (1 - cost_per_side) - entry * (1 + cost_per_side)
        trades.append({
            "signal_date": idx[i],
            "entry_date": idx[e],
            "exit_date": idx[exit_day],
            "outcome": outcome,
            "r": net_pnl / risk,
            "ret_pct": net_pnl / entry * 100,
            "days": exit_day - e + 1,
        })
        i = exit_day + 1  # nessun nuovo trade finché questo è aperto
    return trades


def simulate_old_method(data_ind, atr, signals, atr_sl_mult=1.5, atr_tp_mult=3.0,
                        max_holding_days=20, **_):
    """Replica il backtest originale: ingresso alla chiusura del giorno del
    segnale, nessun costo, trade sovrapposti (ogni giorno di segnale = un
    trade). Serve solo per mostrare quanto cambiano i numeri."""
    h = data_ind["High"].to_numpy(dtype=float)
    l = data_ind["Low"].to_numpy(dtype=float)
    c = data_ind["Close"].to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)
    idx = data_ind.index
    n = len(c)
    trades = []
    for i in np.flatnonzero(signals):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        entry, risk = c[i], atr_sl_mult * a[i]
        stop, target = entry - risk, entry + atr_tp_mult * a[i]
        outcome = None
        for off in range(1, max_holding_days + 1):
            j = i + off
            if j >= n:
                break
            if l[j] <= stop:
                outcome, exit_price, exit_day = "stop_loss", stop, j
                break
            if h[j] >= target:
                outcome, exit_price, exit_day = "take_profit", target, j
                break
        if outcome is None:
            j = min(i + max_holding_days, n - 1)
            if j <= i:
                continue
            outcome, exit_price, exit_day = "timeout", c[j], j
        trades.append({
            "signal_date": idx[i], "entry_date": idx[i], "exit_date": idx[exit_day],
            "outcome": outcome, "r": (exit_price - entry) / risk,
            "ret_pct": (exit_price - entry) / entry * 100, "days": exit_day - i,
        })
    return trades


def run_ticker(data, strategy_keys=None, params=None):
    """Calcola segnali e trade (metodo nuovo e vecchio) per un titolo."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data_ind = compute_indicators(data)
    atr = compute_atr(data_ind)
    out = {}
    for key in strategy_keys or STRATEGIES:
        fn = STRATEGIES[key][1]
        sig = compute_signal_series(data_ind, fn, params["min_history"])
        out[key] = {
            "n_signal_days": int(sig.sum()),
            "new": simulate_realistic(data_ind, atr, sig, **params),
            "old": simulate_old_method(data_ind, atr, sig, **params),
        }
    return out


def summarize(trades):
    """Metriche su una lista di trade (ordinati per data di uscita)."""
    n = len(trades)
    if n == 0:
        return {"n": 0}
    trades = sorted(trades, key=lambda t: t["exit_date"])
    r = np.array([t["r"] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    max_dd = float((peak - equity).max()) if n else 0.0
    std = r.std(ddof=1) if n > 1 else float("nan")
    t_stat = float(r.mean() / (std / math.sqrt(n))) if n > 1 and std > 0 else float("nan")
    outcomes = pd.Series([t["outcome"] for t in trades]).value_counts(normalize=True) * 100
    return {
        "n": n,
        "win_rate": float((r > 0).mean() * 100),
        "avg_r": float(r.mean()),
        "total_r": float(r.sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else None,
        "max_dd_r": max_dd,
        "t_stat": t_stat,
        "avg_ret_pct": float(np.mean([t["ret_pct"] for t in trades])),
        "avg_days": float(np.mean([t["days"] for t in trades])),
        "tp_pct": float(outcomes.get("take_profit", 0)),
        "sl_pct": float(outcomes.get("stop_loss", 0)),
        "timeout_pct": float(outcomes.get("timeout", 0)),
    }


def verdict(full, oos, ins=None):
    """Giudizio sintetico in italiano, volutamente severo."""
    if full.get("n", 0) < 30:
        return "insufficiente", "Meno di 30 trade: troppo pochi per dire qualcosa."
    pf, t = full.get("profit_factor") or 0, full.get("t_stat")
    oos_r, oos_t = oos.get("avg_r") or 0, oos.get("t_stat") or 0
    ins_r = (ins or {}).get("avg_r") or 0
    solid = full["avg_r"] > 0 and pf > 1.1 and t is not None and t > 2
    # confermata solo se negli anni recenti tiene almeno metà del vantaggio
    # e resta distinguibile dal caso
    oos_ok = oos.get("n", 0) >= 15 and oos_r > 0 and oos_r >= 0.5 * ins_r and oos_t > 1.5
    if solid and oos_ok:
        return "promettente", "Guadagno medio positivo, statisticamente solido e confermato sugli anni recenti."
    if solid and oos_r > 0:
        return "incerta", "Positiva sui 10 anni, ma dal 2022 il vantaggio si è ridotto molto e non è più distinguibile dal caso."
    if full["avg_r"] > 0 and pf > 1.0:
        return "incerta", "Leggermente positiva, ma il risultato potrebbe essere dovuto al caso o non regge su tutti gli anni."
    return "negativa", "Dopo costi e ingresso realistico perde soldi."


def strategy_alert_state(key):
    return bool(ALERT_STRATEGY_CONFIG.get(key, False)) if key != "baseline" else None
