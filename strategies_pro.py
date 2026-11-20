"""
strategies_pro.py — Strategie "da manuale", con regole e parametri presi
dalla letteratura e fissati PRIMA di guardare i risultati (nessuna
ottimizzazione sui dati: è il modo più semplice per non ingannarsi).

1. mean_reversion_rsi2 — Larry Connors, "Short Term Trading Strategies
   That Work" (2008). Compra un titolo in trend rialzista (sopra la media
   a 200 giorni) dopo un calo brusco di breve (RSI a 2 giorni sotto 10).
   Esce alla prima chiusura sopra la media a 5 giorni.
2. trend_10_mesi — Mebane Faber, "A Quantitative Approach to Tactical
   Asset Allocation" (2007). A fine mese: indice sopra la media dei 10
   mesi -> investito il mese dopo, altrimenti liquidità.
3. momentum_12_1 — Jegadeesh & Titman (1993), con filtro di regime
   ("absolute momentum", Antonacci). A fine mese compra le 5 azioni con
   il miglior rendimento da 12 a 1 mese fa, pesi uguali, ma solo se il
   FTSE MIB è sopra la sua media a 10 mesi; altrimenti liquidità.

Esecuzione realistica per tutte: il segnale si calcola sulla chiusura,
l'operazione si fa all'APERTURA del giorno dopo, costi 0,15% per lato.
I prezzi sono rettificati per dividendi e frazionamenti.
"""

import numpy as np
import pandas as pd

COST_PER_SIDE = 0.0015

STRATEGY_INFO = {
    "mean_reversion_rsi2": {
        "name": "Mean reversion RSI(2)",
        "source": "Connors, Short Term Trading Strategies That Work (2008)",
        "universe": "Tutti i titoli e indici",
        "rules": [
            "Il titolo chiude sopra la sua media mobile a 200 giorni (trend di fondo rialzista)",
            "L'RSI a 2 giorni chiude sotto 10 (calo brusco di breve)",
            "Compra il giorno dopo all'apertura",
            "Vendi il giorno dopo la prima chiusura sopra la media mobile a 5 giorni",
        ],
    },
    "trend_10_mesi": {
        "name": "Trend following 10 mesi",
        "source": "Faber, A Quantitative Approach to Tactical Asset Allocation (2007)",
        "universe": "Indici USA ed europei",
        "rules": [
            "L'ultimo giorno di borsa del mese guarda la chiusura dell'indice",
            "Se è sopra la media delle ultime 10 chiusure mensili: investito per tutto il mese dopo",
            "Altrimenti: liquidità per tutto il mese dopo",
            "Le operazioni si fanno all'apertura del primo giorno del mese",
        ],
    },
    "momentum_12_1": {
        "name": "Momentum 12-1 azioni italiane",
        "source": "Jegadeesh & Titman (1993), filtro di regime di Antonacci (2014)",
        "universe": "Azioni italiane",
        "rules": [
            "L'ultimo giorno di borsa del mese calcola per ogni azione il rendimento da 12 mesi fa a 1 mese fa",
            "Se il FTSE MIB è sopra la media delle ultime 10 chiusure mensili: compra le 5 migliori, 20% ciascuna",
            "Altrimenti: tutto in liquidità",
            "Le operazioni si fanno all'apertura del primo giorno del mese; si tiene fino al mese dopo",
        ],
    },
}


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------

def adjust_ohlc(data):
    """OHLC rettificati per dividendi/frazionamenti usando Adj Close."""
    df = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if "Adj Close" in df.columns:
        f = (df["Adj Close"] / df["Close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] * f
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]] if "Volume" in df else df[["Open", "High", "Low", "Close"]]


def rsi_wilder(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(avg_loss != 0, 100.0)


def month_end_mask(index):
    """True sull'ultimo giorno di borsa di ogni mese (non sull'ultimo dato
    se il mese è ancora in corso: lì la decisione non è ancora presa)."""
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).transform("max")
    mask = pd.Series(index == last.values, index=index)
    # l'ultimo mese disponibile è chiuso solo se è finito
    if len(index):
        lastday = index[-1]
        nxt = lastday + pd.offsets.BDay(1)
        if nxt.month == lastday.month:
            mask.iloc[-1] = False
    return mask


def monthly_sma_above(close, months=10):
    """Per ogni fine mese: chiusura sopra la media delle ultime `months`
    chiusure mensili. Serie booleana sull'indice giornaliero, valorizzata
    solo nei giorni di fine mese (NaN altrove)."""
    me = month_end_mask(close.index)
    m_close = close[me]
    sma = m_close.rolling(months).mean()
    above = (m_close > sma).where(sma.notna())
    return above.reindex(close.index)


# ---------------------------------------------------------------------------
# Segnali (target deciso alla chiusura del giorno t, 1 = investito)
# ---------------------------------------------------------------------------

def target_rsi2(df, rsi_entry=10, sma_trend=200, sma_exit=5):
    c = df["Close"]
    sma200 = c.rolling(sma_trend).mean().to_numpy()
    sma5 = c.rolling(sma_exit).mean().to_numpy()
    rsi2 = rsi_wilder(c, 2).to_numpy()
    cv = c.to_numpy()
    tgt = np.zeros(len(c))
    pos = 0
    for t in range(len(c)):
        if pos == 0:
            if np.isfinite(sma200[t]) and cv[t] > sma200[t] and np.isfinite(rsi2[t]) and rsi2[t] < rsi_entry:
                pos = 1
        else:
            if np.isfinite(sma5[t]) and cv[t] > sma5[t]:
                pos = 0
        tgt[t] = pos
    return pd.Series(tgt, index=df.index), pd.DataFrame({"RSI2": rsi2, "SMA200": sma200, "SMA5": sma5}, index=df.index)


def target_trend10(df, months=10):
    above = monthly_sma_above(df["Close"], months)
    # la decisione di fine mese vale fino alla fine del mese successivo
    tgt = above.ffill().fillna(0).astype(float)
    return tgt


def targets_momentum(closes, regime_above, top_n=5, lookback=12, skip=1):
    """closes: DataFrame (date x ticker) di chiusure rettificate.
    regime_above: Serie booleana (fine mese) del filtro di mercato."""
    idx = closes.index
    me = month_end_mask(idx)
    m = closes[me]
    mom = m.shift(skip) / m.shift(lookback) - 1
    reg = regime_above.reindex(m.index)
    picks = pd.DataFrame(0.0, index=m.index, columns=closes.columns)
    for d in m.index:
        if not bool(reg.get(d, False) == True):  # noqa: E712 (NaN -> liquidità)
            continue
        row = mom.loc[d].dropna()
        # serve un prezzo valido oggi per poter comprare
        row = row[m.loc[d, row.index].notna()]
        if len(row) < top_n:
            continue
        picks.loc[d, row.nlargest(top_n).index] = 1.0
    tgt = picks.reindex(idx).ffill().fillna(0.0)
    return tgt, mom


# ---------------------------------------------------------------------------
# Simulazione: dal target ai rendimenti giornalieri e ai trade
# ---------------------------------------------------------------------------

def sleeve_returns(df, target, cost=COST_PER_SIDE):
    """Rendimento giornaliero di una "quota" investita su un titolo seguendo
    il target (deciso a chiusura t, eseguito all'apertura t+1)."""
    o = df["Open"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    held = target.shift(1).fillna(0).to_numpy(float)  # posizione durante il giorno
    prev = np.concatenate([[0.0], held[:-1]])
    pc = np.concatenate([[np.nan], c[:-1]])
    r = np.zeros(len(c))
    stay = (prev == 1) & (held == 1)
    enter = (prev == 0) & (held == 1)
    leave = (prev == 1) & (held == 0)
    r[stay] = c[stay] / pc[stay] - 1
    r[enter] = (c[enter] * (1 - cost)) / (o[enter] * (1 + cost)) - 1
    r[leave] = (o[leave] * (1 - cost)) / pc[leave] - 1
    r = np.nan_to_num(r, nan=0.0)
    return pd.Series(r, index=df.index), pd.Series(held, index=df.index)


def extract_trades(df, target, ticker, cost=COST_PER_SIDE):
    held = target.shift(1).fillna(0).to_numpy(float)
    o = df["Open"].to_numpy(float)
    idx = df.index
    trades = []
    entry_i = None
    for i in range(len(held)):
        if held[i] == 1 and (i == 0 or held[i - 1] == 0):
            entry_i = i
        if held[i] == 0 and i > 0 and held[i - 1] == 1 and entry_i is not None:
            ep, xp = o[entry_i], o[i]
            ret = (xp * (1 - cost)) / (ep * (1 + cost)) - 1
            trades.append({"ticker": ticker, "entry": idx[entry_i], "exit": idx[i],
                           "days": i - entry_i, "ret_pct": ret * 100})
            entry_i = None
    return trades


# ---------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------

def perf_metrics(daily, exposure=None):
    daily = daily.dropna()
    if len(daily) < 20:
        return {"days": len(daily)}
    eq = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else float("nan")
    vol = daily.std() * np.sqrt(252)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else float("nan")
    dd = eq / eq.cummax() - 1
    out = {
        "start": str(daily.index[0].date()), "end": str(daily.index[-1].date()),
        "years": round(years, 1),
        "cagr_pct": cagr * 100, "vol_pct": vol * 100, "sharpe": sharpe,
        "max_dd_pct": dd.min() * 100, "total_pct": (eq.iloc[-1] - 1) * 100,
        "sharpe_t": sharpe * np.sqrt(years) if years > 0 else float("nan"),
    }
    if exposure is not None:
        out["exposure_pct"] = float(exposure.reindex(daily.index).fillna(0).mean() * 100)
    return out


def trade_metrics(trades):
    if not trades:
        return {"n": 0}
    r = np.array([t["ret_pct"] for t in trades])
    wins, losses = r[r > 0], r[r <= 0]
    std = r.std(ddof=1) if len(r) > 1 else float("nan")
    return {
        "n": len(r), "win_rate": float((r > 0).mean() * 100), "avg_pct": float(r.mean()),
        "avg_win_pct": float(wins.mean()) if len(wins) else None,
        "avg_loss_pct": float(losses.mean()) if len(losses) else None,
        "worst_pct": float(r.min()), "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
        "t_stat": float(r.mean() / (std / np.sqrt(len(r)))) if len(r) > 1 and std > 0 else None,
        "avg_days": float(np.mean([t["days"] for t in trades])),
    }


def verdict_pro(full, first, second, bench_full):
    """Regola fissata prima dei risultati. Approvata se: guadagna (dopo i
    costi) in ENTRAMBI i periodi, Sharpe >= 0,4 sull'intero periodo, e fa
    meglio del semplice "compra e tieni" degli stessi titoli per rischio
    (Sharpe più alto oppure calo massimo almeno del 40% più piccolo)."""
    def g(d, k):
        v = d.get(k)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    c1, c2, sh = g(first, "cagr_pct"), g(second, "cagr_pct"), g(full, "sharpe")
    if c1 is None or c2 is None or sh is None:
        return "insufficiente", "Storico troppo corto per giudicare."
    beats = (sh >= (g(bench_full, "sharpe") or 0)) or (
        (g(full, "max_dd_pct") or -100) >= 0.6 * (g(bench_full, "max_dd_pct") or -100))
    if c1 > 0 and c2 > 0 and sh >= 0.4 and beats:
        return "approvata", "Guadagna in entrambi i periodi, dopo i costi, con un rischio migliore del compra e tieni."
    if c1 > 0 and c2 > 0:
        return "incerta", "Guadagna in entrambi i periodi, ma non migliora abbastanza il rapporto rendimento/rischio del compra e tieni."
    return "bocciata", "Non guadagna in modo costante dopo i costi."


# ---------------------------------------------------------------------------
# Stato di oggi (per app e alert): ricostruito rigiocando le regole sullo
# storico, così non serve salvare da nessuna parte le posizioni aperte.
# ---------------------------------------------------------------------------

def today_rsi2(df):
    tgt, ind = target_rsi2(df)
    if len(tgt) < 2:
        return None
    last, prev = tgt.iloc[-1], tgt.iloc[-2]
    held = tgt.shift(1).fillna(0)
    row = ind.iloc[-1]
    close = float(df["Close"].iloc[-1])
    info = {"date": str(df.index[-1].date()), "close": close,
            "rsi2": float(row["RSI2"]), "sma200": float(row["SMA200"]), "sma5": float(row["SMA5"])}
    if last == 1 and prev == 0:
        info["action"] = "COMPRA"
        info["text"] = "Compra domani all'apertura."
    elif last == 0 and prev == 1:
        entry = held[held.diff() == 1].index[-1] if (held.diff() == 1).any() else None
        info["action"] = "VENDI"
        info["text"] = "Vendi domani all'apertura (chiusura sopra la media a 5 giorni)."
        info["since"] = str(entry.date()) if entry is not None else None
    elif last == 1:
        entry = held[held.diff() == 1].index[-1] if (held.diff() == 1).any() else None
        info["action"] = "TIENI"
        info["text"] = f"In posizione: vendi alla prima chiusura sopra {info['sma5']:.2f} (media a 5 giorni)."
        info["since"] = str(entry.date()) if entry is not None else None
        if entry is not None:
            info["open_ret_pct"] = (close / float(df.loc[entry, "Open"]) - 1) * 100
    else:
        info["action"] = "NIENTE"
        info["text"] = "Nessuna operazione."
    return info


def today_trend10(df):
    above = monthly_sma_above(df["Close"], 10)
    decided = above.dropna()
    if decided.empty:
        return None
    me = month_end_mask(df.index)
    is_month_end_today = bool(me.iloc[-1]) or (df.index[-1] + pd.offsets.BDay(1)).month != df.index[-1].month
    # media mensile calcolata con la chiusura di oggi (anticipo di fine mese)
    m_close = df["Close"][month_end_mask(df.index)]
    sma_now = pd.concat([m_close.iloc[-9:], df["Close"].iloc[-1:]]).mean()
    state = bool(decided.iloc[-1])
    info = {"date": str(df.index[-1].date()), "close": float(df["Close"].iloc[-1]),
            "sma10m": float(sma_now), "invested": state, "decided_on": str(decided.index[-1].date())}
    above_now = info["close"] > sma_now
    if is_month_end_today and above_now != state:
        info["action"] = "COMPRA" if above_now else "VENDI"
        info["text"] = ("Fine mese: compra domani all'apertura." if above_now
                        else "Fine mese: vendi domani all'apertura e resta in liquidità.")
    else:
        info["action"] = "TIENI" if state else "LIQUIDITÀ"
        info["text"] = ("Investito fino a fine mese." if state else "In liquidità fino a fine mese.")
        info["preview"] = "sopra" if above_now else "sotto"
    return info
