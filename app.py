# app.py — Market Scanner Pro V3 + ML

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==========================
# CONFIG TELEGRAM
# ==========================
# In locale: variabili d'ambiente TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
# Su Streamlit Community Cloud: sezione "Secrets" dell'app (vedi README),
# lette qui tramite st.secrets. Non mettere mai token/chat id in chiaro
# nel codice: in entrambi i casi restano fuori dal repository.

def _get_secret(key, default=""):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)

BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN")
CHAT_ID = _get_secret("TELEGRAM_CHAT_ID")

# ==========================
# LISTA TITOLI ITALIANI
# ==========================

ITALIAN_TICKERS = [
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STLAM.MI", "LDO.MI", "RACE.MI",
    "MONC.MI", "TRN.MI", "PST.MI", "AMP.MI", "BMED.MI", "BAMI.MI", "BPSO.MI",
    "BPER.MI", "CNHI.MI", "DIA.MI", "ERG.MI", "EXO.MI", "G.MI", "IG.MI",
    "IT.MI", "JUVE.MI", "LR.MI", "MS.MI", "PRY.MI", "REC.MI", "SFER.MI",
    "TIT.MI", "TEN.MI", "UNI.MI", "VIV.MI"
]

# ==========================
# INDICI USA ED EUROPEI
# ==========================

US_INDICES = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq Composite": "^IXIC",
    "Russell 2000": "^RUT",
}

EU_INDICES = {
    "FTSE MIB (Italia)": "FTSEMIB.MI",
    "DAX (Germania)": "^GDAXI",
    "CAC 40 (Francia)": "^FCHI",
    "FTSE 100 (UK)": "^FTSE",
    "Euro Stoxx 50": "^STOXX50E",
}

# ==========================
# CATALOGO PER IL MENU A TENDINA (categoria -> {nome visualizzato: ticker})
# ==========================

TICKER_CATALOG = {
    "Azioni Italiane": {t: t for t in ITALIAN_TICKERS},
    "Indici Americani": US_INDICES,
    "Indici Europei": EU_INDICES,
}

def flatten_catalog():
    """Restituisce una lista di tuple (etichetta con categoria, ticker)
    per popolare un unico selectbox ordinato per categoria."""
    flat = []
    for category, items in TICKER_CATALOG.items():
        for name, ticker in items.items():
            label = f"{category} — {name}" if name != ticker else f"{category} — {ticker}"
            flat.append((label, ticker))
    return flat

# ==========================
# TIMEFRAME SUPPORTATI
# ==========================

TIMEFRAME_OPTIONS = {
    "Giornaliero": {"interval": "1d", "period": "1y"},
    "4 Ore": {"interval": "4h", "period": "60d"},
}

# ==========================
# DOWNLOAD DATI (robusto a MultiIndex, con supporto 4 ore via resample)
# ==========================

def download_data(ticker, period="6mo", interval="1d"):
    """Wrapper su yf.download che normalizza le colonne e aggiunge il
    supporto per l'intervallo '4h', che yfinance non fornisce
    direttamente: scarica le barre orarie e le raggruppa in blocchi da
    4 ore. yfinance limita i dati orari a circa gli ultimi 2 anni, per
    cui il periodo richiesto viene automaticamente limitato."""
    if interval == "4h":
        return _download_4h(ticker, period)

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def _download_4h(ticker, period):
    # yfinance accetta al massimo ~730 giorni di dati orari
    hourly_period = period if period in ("1d", "5d", "7d", "1mo", "3mo", "60d", "730d") else "60d"
    hourly = yf.download(
        ticker,
        period=hourly_period,
        interval="1h",
        auto_adjust=False,
        progress=False
    )
    if hourly.empty:
        return hourly
    if isinstance(hourly.columns, pd.MultiIndex):
        hourly.columns = hourly.columns.get_level_values(0)

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    resampled = hourly.resample("4h", origin="start_day").agg(agg).dropna(subset=["Open"])
    return resampled

# ==========================
# TELEGRAM
# ==========================

def send_telegram_message(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        # Alert Telegram non configurato: lo script continua a funzionare
        # normalmente, semplicemente non manda notifiche.
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Errore Telegram: {e}")

# ==========================
# INDICATORI BASE
# ==========================

def compute_indicators(data):
    data = data.copy()

    # EMA
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()

    # RSI (con guardia contro divisione per zero quando avg_loss = 0)
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))
    data["RSI"] = data["RSI"].fillna(100)  # avg_loss=0 => solo rialzi => RSI=100

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    # Stocastico 10-3-6
    data = compute_stochastic(data)

    # OBV
    data = compute_obv(data)

    # Bollinger Bands
    data = compute_bollinger_bands(data)

    # ADX (forza del trend)
    data = compute_adx(data)

    # VWAP (prezzo medio ponderato per volume, dall'inizio del periodo)
    data = compute_vwap(data)

    return data

# ==========================
# STOCASTICO 10-3-6
# ==========================

def compute_stochastic(data, k_period=10, k_smooth=3, d_period=6):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)

    k = 100 * (close - lowest_low) / denom
    k_smooth_val = k.rolling(k_smooth).mean()
    d = k_smooth_val.rolling(d_period).mean()

    data["Stoch_K"] = k_smooth_val
    data["Stoch_D"] = d

    return data

# ==========================
# OBV (vettorizzato, era un loop Python riga per riga)
# ==========================

def compute_obv(data):
    direction = np.sign(data["Close"].diff()).fillna(0)
    data["OBV"] = (direction * data["Volume"]).cumsum()
    return data

# ==========================
# BOLLINGER BANDS
# ==========================

def compute_bollinger_bands(data, period=20, num_std=2):
    data["BB_Mid"] = data["Close"].rolling(period).mean()
    std = data["Close"].rolling(period).std()
    data["BB_Upper"] = data["BB_Mid"] + num_std * std
    data["BB_Lower"] = data["BB_Mid"] - num_std * std
    band_width = (data["BB_Upper"] - data["BB_Lower"]).replace(0, np.nan)
    data["BB_PercentB"] = (data["Close"] - data["BB_Lower"]) / band_width
    return data

def bollinger_score(data):
    last = data.iloc[-1]
    percent_b = last.get("BB_PercentB", np.nan)
    close = last["Close"]
    upper = last.get("BB_Upper", np.nan)
    if pd.isna(percent_b) or pd.isna(upper):
        return 0
    if close > upper:
        return 8  # chiusura sopra la banda superiore: forte momentum/breakout di volatilità
    elif percent_b > 0.8:
        return 4  # vicino alla banda superiore
    return 0

# ==========================
# ADX (forza del trend)
# ==========================

def compute_adx(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=data.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=data.index
    )

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothing di Wilder (equivalente a ewm con alpha=1/period)
    atr_wilder = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_wilder.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_wilder.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    data["Plus_DI"] = plus_di
    data["Minus_DI"] = minus_di
    data["ADX"] = adx
    return data

def adx_score(data):
    last = data.iloc[-1]
    adx = last.get("ADX", np.nan)
    plus_di = last.get("Plus_DI", np.nan)
    minus_di = last.get("Minus_DI", np.nan)
    if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
        return 0
    if adx > 25 and plus_di > minus_di:
        return 10  # trend rialzista forte e confermato
    elif adx > 20 and plus_di > minus_di:
        return 5   # trend rialzista, forza moderata
    return 0

# ==========================
# VWAP
# ==========================

def compute_vwap(data):
    """VWAP 'ancorato' dall'inizio della serie caricata (non è il VWAP
    intraday classico che riparte ogni giorno, dato che qui lavoriamo
    su barre giornaliere)."""
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    cum_vol = data["Volume"].cumsum()
    cum_tp_vol = (typical_price * data["Volume"]).cumsum()
    data["VWAP"] = cum_tp_vol / cum_vol.replace(0, np.nan)
    return data

def vwap_score(data):
    last = data.iloc[-1]
    vwap = last.get("VWAP", np.nan)
    if pd.isna(vwap):
        return 0
    return 5 if last["Close"] > vwap else 0

# ==========================
# ROTTURA TREND LINE (pivot su minimi/massimi, come si traccia a mano)
# ==========================

def find_pivots(series, window=3, kind="low"):
    """Un punto è un pivot minimo se è il più basso in una finestra di
    `window` barre prima e dopo (pivot massimo: analogo sui massimi).
    Restituisce le posizioni intere (offset) dentro `series`."""
    idxs = []
    n = len(series)
    if n < 2 * window + 1:
        return idxs
    for i in range(window, n - window):
        seg = series.iloc[i - window:i + window + 1]
        val = series.iloc[i]
        if kind == "low" and val == seg.min():
            idxs.append(i)
        elif kind == "high" and val == seg.max():
            idxs.append(i)
    return idxs

def compute_swing_trendline(data, lookback=90, pivot_window=3, kind="low", num_pivots=3):
    """Trend line 'classica': una retta che passa per gli ultimi pivot
    (minimi = supporto rialzista, massimi = resistenza ribassista),
    esattamente come si traccia manualmente su una piattaforma di
    trading — non una regressione su tutte le chiusure."""
    window = data.tail(lookback)
    series = window["Low"] if kind == "low" else window["High"]

    pivots = find_pivots(series, window=pivot_window, kind=kind)
    if len(pivots) < 2:
        return None

    chosen = pivots[-num_pivots:] if len(pivots) >= num_pivots else pivots[-2:]
    xs = np.array(chosen, dtype=float)
    ys = series.iloc[chosen].values
    m, q = np.polyfit(xs, ys, 1)

    # una trend line di supporto deve salire, una di resistenza deve
    # scendere: se la pendenza va nel verso sbagliato, i pivot scelti
    # non formano una trend line valida in questo momento
    expected_sign = 1 if kind == "low" else -1
    if m == 0 or np.sign(m) != expected_sign:
        return None

    last_offset = len(window) - 1
    value_today = m * last_offset + q
    value_yesterday = m * (last_offset - 1) + q

    line_offsets = np.arange(chosen[0], len(window))
    line_values = m * line_offsets + q

    return {
        "slope": m,
        "intercept": q,
        "value_today": value_today,
        "value_yesterday": value_yesterday,
        "pivot_dates": window.index[chosen],
        "pivot_values": ys,
        "line_dates": window.index[chosen[0]:],
        "line_values": line_values,
    }

def detect_swing_trendline_break(data, lookback=90, pivot_window=3):
    """Controlla sia la rottura del supporto (ribassista, allarme) sia
    la rottura della resistenza (rialzista, segnale di forza)."""
    close_today = data["Close"].iloc[-1]
    close_yesterday = data["Close"].iloc[-2]

    out = {"support_break": False, "resistance_break": False, "support": None, "resistance": None}

    support = compute_swing_trendline(data, lookback, pivot_window, kind="low")
    if support:
        out["support"] = support
        out["support_break"] = bool((close_yesterday >= support["value_yesterday"]) and (close_today < support["value_today"]))

    resistance = compute_swing_trendline(data, lookback, pivot_window, kind="high")
    if resistance:
        out["resistance"] = resistance
        out["resistance_break"] = bool((close_yesterday <= resistance["value_yesterday"]) and (close_today > resistance["value_today"]))

    return out

def trendline_break_score(data, lookback=90, pivot_window=3):
    brk = detect_swing_trendline_break(data, lookback, pivot_window)
    if brk["resistance_break"]:
        return 10  # rottura rialzista della trend line ribassista (sui massimi): segnale di forza
    if brk["support_break"]:
        return 0   # rottura ribassista del supporto: nessun bonus (ma viene comunque segnalata)
    if brk["support"] is not None and data["Close"].iloc[-1] > brk["support"]["value_today"]:
        return 3   # prezzo ancora sopra un supporto rialzista sano
    return 0

def detect_trendline_break(data, lookback=90, pivot_window=3):
    """Booleano usato per l'alert 'rottura oggi': True solo per la
    rottura rialzista della resistenza esattamente nell'ultima barra."""
    brk = detect_swing_trendline_break(data, lookback, pivot_window)
    return bool(brk["resistance_break"])

# ==========================
# SEGNALE ACQUISTO/VENDITA (rottura trend line + conferme)
# ==========================
# Logica: la rottura di una trend line da sola genera troppi falsi
# segnali. Qui il segnale scatta solo se, oltre alla rottura, almeno
# `min_confirmations` altri indicatori vanno nella stessa direzione.

def compute_trade_signal(data, min_confirmations=3):
    last = data.iloc[-1]
    swing_brk = detect_swing_trendline_break(data)

    bullish, bearish = [], []

    if not pd.isna(last.get("RSI", np.nan)):
        (bullish if last["RSI"] > 50 else bearish).append(f"RSI {'>' if last['RSI'] > 50 else '<'} 50 ({round(last['RSI'],1)})")

    if not pd.isna(last.get("MACD", np.nan)) and not pd.isna(last.get("Signal", np.nan)):
        (bullish if last["MACD"] > last["Signal"] else bearish).append("MACD sopra Signal" if last["MACD"] > last["Signal"] else "MACD sotto Signal")

    adx = last.get("ADX", np.nan)
    plus_di = last.get("Plus_DI", np.nan)
    minus_di = last.get("Minus_DI", np.nan)
    if not pd.isna(adx) and adx > 20 and not pd.isna(plus_di) and not pd.isna(minus_di):
        (bullish if plus_di > minus_di else bearish).append(f"ADX {round(adx,1)} con {'+DI>-DI' if plus_di > minus_di else '-DI>+DI'}")

    vwap = last.get("VWAP", np.nan)
    if not pd.isna(vwap):
        (bullish if last["Close"] > vwap else bearish).append("Sopra VWAP" if last["Close"] > vwap else "Sotto VWAP")

    stoch_k, stoch_d = last.get("Stoch_K", np.nan), last.get("Stoch_D", np.nan)
    if not pd.isna(stoch_k) and not pd.isna(stoch_d):
        (bullish if stoch_k > stoch_d else bearish).append("Stocastico K>D" if stoch_k > stoch_d else "Stocastico K<D")

    signal = "NEUTRALE"
    confirmations = []

    if swing_brk["resistance_break"] and len(bullish) >= min_confirmations:
        signal = "ACQUISTO"
        confirmations = bullish
    elif swing_brk["support_break"] and len(bearish) >= min_confirmations:
        signal = "VENDITA"
        confirmations = bearish

    return {
        "signal": signal,
        "resistance_break": swing_brk["resistance_break"],
        "support_break": swing_brk["support_break"],
        "bullish_confirmations": bullish,
        "bearish_confirmations": bearish,
        "confirmations_used": confirmations,
        "min_confirmations": min_confirmations,
    }

# ==========================
# ATR
# ==========================

def compute_atr(data, period=14):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr

# ==========================
# SCORE MODULI
# ==========================

def trend_score(last):
    if last["EMA20"] > last["EMA50"] > last["EMA200"]:
        return 20
    elif last["EMA20"] > last["EMA50"]:
        return 12
    elif last["EMA20"] > last["EMA200"]:
        return 6
    return 0

def momentum_score(last):
    if last["RSI"] > 60:
        return 15
    elif last["RSI"] > 50:
        return 8
    return 0

def macd_score(last):
    if last["MACD"] > last["Signal"]:
        return 15
    return 0

def volume_spike_score(data):
    vol20 = data["Volume"].rolling(20).mean().iloc[-1]
    last_vol = data["Volume"].iloc[-1]
    if not vol20 or pd.isna(vol20):
        return 0
    ratio = last_vol / vol20

    if ratio > 2:
        return 8
    elif ratio > 1.5:
        return 4
    elif ratio > 1.2:
        return 2
    return 0

def volume_trend_score(data):
    recent = data.tail(30)
    if len(recent) < 2:
        return 0
    x = np.arange(len(recent))
    y = recent["Volume"].values
    m, q = np.polyfit(x, y, 1)
    if m > 0:
        return 4
    return 0

def obv_score(data):
    recent = data.tail(20)
    if len(recent) < 2:
        return 0
    x = np.arange(len(recent))
    y = recent["OBV"].values
    m, q = np.polyfit(x, y, 1)
    if m > 0:
        return 8
    return 0

def volume_score_advanced(data):
    # OBV già calcolato in compute_indicators: non ricalcolarlo qui
    # (evita di sovrascrivere la colonna e sprecare tempo ad ogni score)
    score = 0
    score += obv_score(data)
    score += volume_spike_score(data)
    score += volume_trend_score(data)
    return score

def breakout_score_advanced(data):
    recent = data.tail(60)
    resistance = recent["High"].max()
    last_close = data["Close"].iloc[-1]
    atr = compute_atr(data).iloc[-1]

    if pd.isna(atr) or atr == 0:
        return 0

    breakout_ratio = (last_close - resistance) / atr

    if breakout_ratio > 2:
        return 15
    elif breakout_ratio > 1:
        return 10
    elif breakout_ratio > 0.5:
        return 5
    elif breakout_ratio > 0:
        return 2
    return 0

def trendline_score_advanced(data):
    recent = data.tail(60)
    if len(recent) < 2:
        return 0
    x = np.arange(len(recent))
    y = recent["Close"].values
    m, q = np.polyfit(x, y, 1)
    slope_norm = m / y[-1]

    if slope_norm > 0.02:
        return 15
    elif slope_norm > 0.01:
        return 10
    elif slope_norm > 0.005:
        return 5
    elif slope_norm > 0:
        return 2
    return 0

def stochastic_score(data):
    last_k = data["Stoch_K"].iloc[-1]
    last_d = data["Stoch_D"].iloc[-1]
    if pd.isna(last_k) or pd.isna(last_d):
        return 0
    if last_k > last_d and last_k < 80:
        return 10
    if last_k > last_d:
        return 5
    return 0

# ==========================
# SCORE PESATO V3
# ==========================
# NB: queste funzioni guardano SEMPRE "adesso" (data.iloc[-1] o
# data.tail(60)). Per uso live va bene; per training ML servono invece
# gli score calcolati punto-per-punto nel tempo (vedi
# build_ml_dataset più sotto, che NON usa questa funzione com'era
# nell'originale per evitare lookahead bias).
#
# NB2: ogni componente viene normalizzata sul proprio massimo prima di
# applicare il peso, così lo score finale usa davvero l'intera scala
# 0-100 (nella versione originale la somma pesata dei punteggi grezzi
# non superava mai ~17, mentre altrove nel codice si confrontava lo
# score con soglie come "score >= 75": quella soglia non scattava
# quasi mai).

def compute_weighted_score(data):
    last = data.iloc[-1]

    # (punteggio_grezzo, punteggio_massimo_possibile, peso)
    components = {
        "trend":            (trend_score(last), 20, 0.16),
        "momentum":         (momentum_score(last), 15, 0.06),
        "macd":             (macd_score(last), 15, 0.06),
        "volume":           (volume_score_advanced(data), 20, 0.13),
        "breakout":         (breakout_score_advanced(data), 15, 0.13),
        "trendline_slope":  (trendline_score_advanced(data), 15, 0.06),
        "stochastic":       (stochastic_score(data), 10, 0.04),
        "trendline_break":  (trendline_break_score(data), 10, 0.10),
        "adx":              (adx_score(data), 10, 0.10),
        "bollinger":        (bollinger_score(data), 8, 0.10),
        "vwap":             (vwap_score(data), 5, 0.06),
    }

    weighted = 0.0
    for raw_score, max_score, weight in components.values():
        normalized = (raw_score / max_score) if max_score else 0
        weighted += normalized * weight * 100

    return round(min(weighted, 100), 2)

def compute_score(data):
    return compute_weighted_score(data)

# ==========================
# FILTRI SCANNER V2
# ==========================

def filter_atr(data, max_atr_ratio=0.03):
    atr = compute_atr(data).iloc[-1]
    price = data["Close"].iloc[-1]
    if pd.isna(atr):
        return False
    return atr <= price * max_atr_ratio

def filter_volatility(data, max_volatility=0.04):
    recent = data.tail(20)
    high = recent["High"].max()
    low = recent["Low"].min()
    volatility = (high - low) / recent["Close"].iloc[-1]
    return volatility <= max_volatility

def filter_volume_spike(data):
    vol20 = data["Volume"].rolling(20).mean().iloc[-1]
    last_vol = data["Volume"].iloc[-1]
    if not vol20 or pd.isna(vol20):
        return False
    return last_vol > vol20 * 1.2

def filter_trendline(data):
    recent = data.tail(60)
    if len(recent) < 2:
        return False
    x = np.arange(len(recent))
    y = recent["Close"].values
    m, q = np.polyfit(x, y, 1)
    return m > 0

def filter_breakout(data):
    recent = data.tail(60)
    resistance = recent["High"].max()
    close = data["Close"].iloc[-1]
    atr = compute_atr(data).iloc[-1]
    if atr == 0 or pd.isna(atr):
        return False
    breakout_ratio = (close - resistance) / atr
    return breakout_ratio > 0.5

def filter_ema(data):
    last = data.iloc[-1]
    return last["EMA20"] > last["EMA50"] > last["EMA200"]

def filter_stochastic(data):
    last_k = data["Stoch_K"].iloc[-1]
    last_d = data["Stoch_D"].iloc[-1]
    if pd.isna(last_k) or pd.isna(last_d):
        return False
    return last_k > last_d and last_k < 80

# ==========================
# ALERT V2
# ==========================

def send_alert_v2(ticker, data, score, timeframe_label="Giornaliero"):
    last = data.iloc[-1]

    breakout = breakout_score_advanced(data)
    vol_spike = volume_spike_score(data)
    trendline = trendline_score_advanced(data)
    stoch_k = data["Stoch_K"].iloc[-1]
    stoch_d = data["Stoch_D"].iloc[-1]
    trade = compute_trade_signal(data)
    resistance_break = trade["resistance_break"]
    support_break = trade["support_break"]
    adx_val = last.get("ADX", np.nan)
    plus_di = last.get("Plus_DI", np.nan)
    minus_di = last.get("Minus_DI", np.nan)
    above_vwap = (not pd.isna(last.get("VWAP", np.nan))) and last["Close"] > last["VWAP"]

    strong_score = score >= 75
    strong_breakout = breakout >= 10
    strong_volume = vol_spike >= 4
    strong_trendline = trendline >= 10
    stoch_signal = stoch_k > stoch_d and stoch_k < 80
    strong_adx = (not pd.isna(adx_val)) and adx_val > 25 and plus_di > minus_di
    has_trade_signal = trade["signal"] != "NEUTRALE"

    if not (strong_score or strong_breakout or strong_volume or strong_trendline
            or stoch_signal or resistance_break or support_break or strong_adx or has_trade_signal):
        return

    adx_str = f"{round(adx_val, 1)}" if not pd.isna(adx_val) else "n/d"

    trendline_line = "📐 Trend Line: "
    if resistance_break:
        trendline_line += "🔀⬆️ rottura rialzista (sopra resistenza sui massimi)"
    elif support_break:
        trendline_line += "🔀⬇️ rottura ribassista (sotto supporto sui minimi)"
    else:
        trendline_line += "nessuna rottura oggi"

    signal_header = ""
    if trade["signal"] == "ACQUISTO":
        signal_header = f"🟢 <b>SEGNALE ACQUISTO</b> ({len(trade['confirmations_used'])} conferme)\n\n"
    elif trade["signal"] == "VENDITA":
        signal_header = f"🔴 <b>SEGNALE VENDITA</b> ({len(trade['confirmations_used'])} conferme)\n\n"

    msg = (
        f"🚨 <b>ALERT V2: {ticker}</b> ({timeframe_label})\n"
        f"{signal_header}"
        f"Score: {score}/100\n"
        f"Prezzo: {round(last['Close'], 2)}\n\n"
        f"📈 Breakout ATR: {breakout}\n"
        f"📊 Volume Spike: {vol_spike}\n"
        f"{trendline_line}\n"
        f"🎯 Stocastico K/D: {round(stoch_k,2)} / {round(stoch_d,2)}\n"
        f"💪 ADX: {adx_str}\n"
        f"💰 Sopra VWAP: {'sì' if above_vwap else 'no'}\n\n"
        f"⚠️ Segnale algoritmico, non è un consiglio di investimento."
    )

    send_telegram_message(BOT_TOKEN, CHAT_ID, msg)

# ==========================
# SCANNER V2
# ==========================

def run_scanner_v2(tickers=None, interval="1d", period=None, timeframe_label="Giornaliero"):
    if tickers is None:
        tickers = ITALIAN_TICKERS
    if period is None:
        period = "6mo" if interval == "1d" else "60d"

    results = []

    for ticker in tickers:
        try:
            data = download_data(ticker, period=period, interval=interval)

            min_bars = 200 if interval == "1d" else 80
            if data.empty or len(data) < min_bars:
                continue

            data = compute_indicators(data)
            score = compute_score(data)

            if not filter_ema(data):
                continue
            if not filter_trendline(data):
                continue
            if not filter_volume_spike(data):
                continue
            if not filter_breakout(data):
                continue
            if not filter_atr(data):
                continue
            if not filter_volatility(data):
                continue
            if not filter_stochastic(data):
                continue

            send_alert_v2(ticker, data, score, timeframe_label=timeframe_label)

            last = data.iloc[-1]
            atr_val = compute_atr(data).iloc[-1]
            resistance = data.tail(60)["High"].max()
            breakout_ratio = (last["Close"] - resistance) / atr_val if atr_val and not pd.isna(atr_val) and atr_val != 0 else 0
            adx_val = last.get("ADX", np.nan)
            above_vwap = (not pd.isna(last.get("VWAP", np.nan))) and last["Close"] > last["VWAP"]
            swing_brk = detect_swing_trendline_break(data)
            if swing_brk["resistance_break"]:
                trendline_flag = "🔀⬆️"
            elif swing_brk["support_break"]:
                trendline_flag = "🔀⬇️"
            else:
                trendline_flag = ""
            trade_signal = compute_trade_signal(data)["signal"]

            results.append({
                "Ticker": ticker,
                "Timeframe": timeframe_label,
                "Prezzo": round(last["Close"], 2),
                "Score": score,
                "Segnale": trade_signal,
                "RSI": round(last["RSI"], 2),
                "MACD": round(last["MACD"], 4),
                "ATR": round(atr_val, 4) if not pd.isna(atr_val) else None,
                "Breakout Ratio": round(breakout_ratio, 2),
                "ADX": round(adx_val, 1) if not pd.isna(adx_val) else None,
                "Rottura Trendline": trendline_flag,
                "Sopra VWAP": "✔️" if above_vwap else ""
            })

        except Exception as e:
            print(f"Errore su {ticker}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Score", ascending=False)
    return df

# ==========================
# BACKTEST BREAKOUT
# ==========================

def backtest_breakouts(data):
    results = []
    atr = compute_atr(data)

    for i in range(60, len(data)):
        window = data.iloc[i-60:i]
        resistance = window["High"].max()

        close_today = data["Close"].iloc[i]
        atr_today = atr.iloc[i]

        if pd.isna(atr_today) or atr_today == 0:
            continue

        breakout_ratio = (close_today - resistance) / atr_today

        if breakout_ratio > 1:
            date = data.index[i]

            perf_5 = perf_10 = perf_20 = None

            if i + 5 < len(data):
                perf_5 = (data["Close"].iloc[i+5] - close_today) / close_today * 100
            if i + 10 < len(data):
                perf_10 = (data["Close"].iloc[i+10] - close_today) / close_today * 100
            if i + 20 < len(data):
                perf_20 = (data["Close"].iloc[i+20] - close_today) / close_today * 100

            results.append({
                "Data": date,
                "Breakout Ratio": round(breakout_ratio, 2),
                "Close": round(close_today, 2),
                "Perf 5g": round(perf_5, 2) if perf_5 is not None else None,
                "Perf 10g": round(perf_10, 2) if perf_10 is not None else None,
                "Perf 20g": round(perf_20, 2) if perf_20 is not None else None,
                "Successo": "✔️" if perf_20 is not None and perf_20 > 0 else "❌"
            })

    return pd.DataFrame(results)

# ==========================
# BACKTEST SCORE
# ==========================

def backtest_score(data):
    results = []

    data = compute_indicators(data)
    scores = []

    for i in range(len(data)):
        if i < 200:
            scores.append(None)
            continue
        window = data.iloc[:i+1]
        score = compute_score(window)
        scores.append(score)

    data = data.copy()
    data["Score"] = scores

    for i in range(len(data)):
        score_today = data["Score"].iloc[i]
        if score_today is None or score_today < 75:
            continue

        close_today = data["Close"].iloc[i]

        perf_5 = perf_10 = perf_20 = None

        if i + 5 < len(data):
            perf_5 = (data["Close"].iloc[i+5] - close_today) / close_today * 100
        if i + 10 < len(data):
            perf_10 = (data["Close"].iloc[i+10] - close_today) / close_today * 100
        if i + 20 < len(data):
            perf_20 = (data["Close"].iloc[i+20] - close_today) / close_today * 100

        results.append({
            "Data": data.index[i],
            "Score": score_today,
            "Close": round(close_today, 2),
            "Perf 5g": round(perf_5, 2) if perf_5 is not None else None,
            "Perf 10g": round(perf_10, 2) if perf_10 is not None else None,
            "Perf 20g": round(perf_20, 2) if perf_20 is not None else None,
            "Successo": "✔️" if perf_20 is not None and perf_20 > 0 else "❌"
        })

    return pd.DataFrame(results)

# ==========================
# SCORE STORICO PUNTO-PER-PUNTO (per il dataset ML, senza lookahead)
# ==========================

def compute_pointwise_scores(data, min_history=200):
    """Calcola TrendlineScore/BreakoutScore/ScoreV3 usando, per ogni
    riga, solo i dati disponibili fino a quel giorno incluso. Sostituisce
    l'approccio originale che applicava un unico valore (calcolato
    sull'intero dataset) a tutte le righe, causando lookahead bias."""
    n = len(data)
    trendline_scores = [np.nan] * n
    breakout_scores = [np.nan] * n
    total_scores = [np.nan] * n

    for i in range(min_history, n):
        window = data.iloc[:i+1]
        trendline_scores[i] = trendline_score_advanced(window)
        breakout_scores[i] = breakout_score_advanced(window)
        total_scores[i] = compute_score(window)

    return (
        pd.Series(trendline_scores, index=data.index),
        pd.Series(breakout_scores, index=data.index),
        pd.Series(total_scores, index=data.index),
    )

# ==========================
# ML DATASET + MODEL
# ==========================

def build_ml_dataset(data, min_history=200):
    data = compute_indicators(data)

    data["Future_Close"] = data["Close"].shift(-10)
    data["Target"] = (data["Future_Close"] > data["Close"]).astype(int)

    tl_scores, brk_scores, v3_scores = compute_pointwise_scores(data, min_history=min_history)

    features = pd.DataFrame({
        "RSI": data["RSI"],
        "MACD": data["MACD"],
        "Signal": data["Signal"],
        "EMA20": data["EMA20"],
        "EMA50": data["EMA50"],
        "EMA200": data["EMA200"],
        "ATR": compute_atr(data),
        "Stoch_K": data["Stoch_K"],
        "Stoch_D": data["Stoch_D"],
        "OBV": data["OBV"],
        "Volume": data["Volume"],
        "BB_PercentB": data["BB_PercentB"],
        "ADX": data["ADX"],
        "Plus_DI": data["Plus_DI"],
        "Minus_DI": data["Minus_DI"],
        "VWAP_Distance": (data["Close"] - data["VWAP"]) / data["VWAP"],
        "TrendlineScore": tl_scores,
        "BreakoutScore": brk_scores,
        "ScoreV3": v3_scores
    })

    # Scarta anche le ultime 10 righe: non hanno un Target valido
    # perché Future_Close guarda 10 giorni avanti (che non esistono)
    valid = features.dropna().index.intersection(data.dropna(subset=["Target"]).index)
    features = features.loc[valid]
    target = data.loc[valid, "Target"]

    return features, target

def train_ml_model(features, target):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, target)

    feature_names = features.columns.tolist()
    return model, scaler, feature_names

def compute_ml_score(model, scaler, features):
    X_scaled = scaler.transform(features.tail(1))
    prob = model.predict_proba(X_scaled)[0][1]
    return round(prob * 100, 2)

def get_feature_importance(model, feature_names):
    coefs = model.coef_[0]
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Peso": coefs,
        "Importanza Assoluta": abs(coefs)
    }).sort_values("Importanza Assoluta", ascending=False)
    return importance

# ==========================
# SCANNER V3 (ML)
# ==========================

def run_scanner_v3(tickers=None):
    if tickers is None:
        tickers = ITALIAN_TICKERS

    results = []

    for ticker in tickers:
        try:
            data = download_data(ticker, period="1y", interval="1d")

            if data.empty:
                continue

            features, target = build_ml_dataset(data)
            if len(features) < 200:
                continue

            model, scaler, feature_names = train_ml_model(features, target)
            ml_score = compute_ml_score(model, scaler, features)
            score_v3 = compute_score(compute_indicators(data))
            final_score = round((ml_score * 0.6) + (score_v3 * 0.4), 2)

            last = data.iloc[-1]

            results.append({
                "Ticker": ticker,
                "Prezzo": round(last["Close"], 2),
                "Score ML": ml_score,
                "Score V3": score_v3,
                "Score Finale": final_score
            })

        except Exception as e:
            print(f"Errore su {ticker}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Score Finale", ascending=False)
    return df

# ==========================
# STREAMLIT UI
# ==========================

st.set_page_config(page_title="Market Scanner Pro V3 + ML", layout="wide")

st.title("📈 Market Scanner Pro V3 + Machine Learning")
st.caption("⚠️ Strumento a scopo informativo/didattico. Gli score e le probabilità del modello ML NON sono consigli di investimento.")

ticker_labels = [label for label, _ in flatten_catalog()]
ticker_map = dict(flatten_catalog())

col_a, col_b = st.columns([2, 1])
with col_a:
    selected_label = st.selectbox("Titolo / Indice", ["Personalizzato..."] + ticker_labels)
with col_b:
    timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_OPTIONS.keys()))

if selected_label == "Personalizzato...":
    ticker = st.text_input("Ticker personalizzato (es. AAPL, TSLA, BTC-USD)", "ENI.MI")
else:
    ticker = ticker_map[selected_label]
    st.caption(f"Ticker: `{ticker}`")

timeframe = TIMEFRAME_OPTIONS[timeframe_label]
period = st.selectbox("Periodo storico", ["1mo", "3mo", "6mo", "1y", "2y"] if timeframe["interval"] == "1d" else ["5d", "1mo", "3mo", "60d"], index=2 if timeframe["interval"] == "1d" else 0)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Grafico", "📊 Indicatori", "🔢 Score", "🧮 Scanner", "📜 Backtest"]
)

# --------------------------
# TAB 1: GRAFICO
# --------------------------

with tab1:
    st.header("📈 Grafico del titolo")

    if st.button("Mostra Grafico"):
        data = download_data(ticker, period=period, interval=timeframe["interval"])

        if data.empty:
            st.error("Nessun dato trovato.")
        else:
            data = compute_indicators(data)

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Prezzo"
            ))

            fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], line=dict(color="blue"), name="EMA20"))
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], line=dict(color="orange"), name="EMA50"))
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA200"], line=dict(color="green"), name="EMA200"))

            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Upper"], line=dict(color="gray", width=1),
                name="Bollinger Upper"
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data["BB_Lower"], line=dict(color="gray", width=1),
                name="Bollinger Lower", fill="tonexty", fillcolor="rgba(150,150,150,0.1)"
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data["VWAP"], line=dict(color="magenta", width=1.5, dash="dot"),
                name="VWAP"
            ))

            recent = data.tail(60)
            support = compute_swing_trendline(data, lookback=min(90, len(data)), pivot_window=3, kind="low")
            resistance_tl = compute_swing_trendline(data, lookback=min(90, len(data)), pivot_window=3, kind="high")

            if support:
                fig.add_trace(go.Scatter(
                    x=support["line_dates"], y=support["line_values"], mode="lines",
                    line=dict(color="green", width=2), name="Trend Line rialzista (supporto)"
                ))
                fig.add_trace(go.Scatter(
                    x=support["pivot_dates"], y=support["pivot_values"], mode="markers",
                    marker=dict(color="green", size=9, symbol="triangle-up"), name="Pivot minimi"
                ))

            if resistance_tl:
                fig.add_trace(go.Scatter(
                    x=resistance_tl["line_dates"], y=resistance_tl["line_values"], mode="lines",
                    line=dict(color="red", width=2), name="Trend Line ribassista (resistenza)"
                ))
                fig.add_trace(go.Scatter(
                    x=resistance_tl["pivot_dates"], y=resistance_tl["pivot_values"], mode="markers",
                    marker=dict(color="red", size=9, symbol="triangle-down"), name="Pivot massimi"
                ))

            if not support and not resistance_tl:
                st.caption("Nessuna trend line valida rilevata sugli ultimi pivot: servono almeno 2 minimi (o massimi) consecutivi con la giusta inclinazione.")

            fig.update_layout(height=600, title=f"Grafico {ticker}")
            st.plotly_chart(fig, use_container_width=True)

# --------------------------
# TAB 2: INDICATORI
# --------------------------

with tab2:
    st.header("📊 Indicatori Tecnici")

    if st.button("Mostra Indicatori"):
        data = download_data(ticker, period=period, interval=timeframe["interval"])

        if data.empty:
            st.error("Nessun dato trovato.")
        else:
            data = compute_indicators(data)

            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI"], line=dict(color="blue"), name="RSI"))
            fig_rsi.add_hline(y=70, line=dict(color="red", dash="dot"))
            fig_rsi.add_hline(y=30, line=dict(color="green", dash="dot"))
            fig_rsi.update_layout(title="RSI", height=250)
            st.subheader("RSI")
            st.plotly_chart(fig_rsi, use_container_width=True)

            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], line=dict(color="blue"), name="MACD"))
            fig_macd.add_trace(go.Scatter(x=data.index, y=data["Signal"], line=dict(color="orange"), name="Signal"))
            fig_macd.update_layout(title="MACD", height=250)
            st.subheader("MACD")
            st.plotly_chart(fig_macd, use_container_width=True)

            fig_volume = go.Figure()
            fig_volume.add_trace(go.Bar(x=data.index, y=data["Volume"], name="Volume"))
            fig_volume.update_layout(title="Volume", height=250)
            st.subheader("Volume")
            st.plotly_chart(fig_volume, use_container_width=True)

            fig_stoch = go.Figure()
            fig_stoch.add_trace(go.Scatter(x=data.index, y=data["Stoch_K"], line=dict(color="blue"), name="%K (10-3)"))
            fig_stoch.add_trace(go.Scatter(x=data.index, y=data["Stoch_D"], line=dict(color="orange"), name="%D (6)"))
            fig_stoch.add_hline(y=80, line=dict(color="red", dash="dot"))
            fig_stoch.add_hline(y=20, line=dict(color="green", dash="dot"))
            fig_stoch.update_layout(title="Stocastico 10-3-6", height=250, yaxis=dict(range=[0, 100]))
            st.subheader("Stocastico 10-3-6")
            st.plotly_chart(fig_stoch, use_container_width=True)

            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=data.index, y=data["ADX"], line=dict(color="black"), name="ADX"))
            fig_adx.add_trace(go.Scatter(x=data.index, y=data["Plus_DI"], line=dict(color="green"), name="+DI"))
            fig_adx.add_trace(go.Scatter(x=data.index, y=data["Minus_DI"], line=dict(color="red"), name="-DI"))
            fig_adx.add_hline(y=25, line=dict(color="gray", dash="dot"))
            fig_adx.update_layout(title="ADX — forza del trend (>25 = trend forte)", height=250)
            st.subheader("ADX")
            st.caption("Non indica la direzione, solo quanto è forte il trend in corso. +DI > -DI = pressione rialzista prevalente.")
            st.plotly_chart(fig_adx, use_container_width=True)

# --------------------------
# TAB 3: SCORE
# --------------------------

with tab3:
    st.header("🔢 Score del Titolo")

    if st.button("Calcola Score"):
        data = download_data(ticker, period=period, interval=timeframe["interval"])

        if data.empty:
            st.error("Nessun dato trovato.")
        else:
            data = compute_indicators(data)
            score = compute_score(data)
            st.metric("Score Finale Pesato V3", f"{score}/100")

            st.subheader("🚦 Segnale (rottura trend line + conferme)")
            trade = compute_trade_signal(data)
            if trade["signal"] == "ACQUISTO":
                st.success(f"📈 ACQUISTO — rottura rialzista della trend line confermata da {len(trade['confirmations_used'])} indicatori")
            elif trade["signal"] == "VENDITA":
                st.error(f"📉 VENDITA — rottura ribassista della trend line confermata da {len(trade['confirmations_used'])} indicatori")
            else:
                st.info("⏸️ NEUTRALE — nessuna rottura di trend line oggi, oppure rottura senza abbastanza conferme")

            if trade["confirmations_used"]:
                st.write("**Indicatori a conferma:** " + ", ".join(trade["confirmations_used"]))
            with st.expander("Dettaglio di tutti gli indicatori"):
                st.write(f"Rottura resistenza (ribassista → rialzista) oggi: {trade['resistance_break']}")
                st.write(f"Rottura supporto (rialzista → ribassista) oggi: {trade['support_break']}")
                st.write(f"Segnali rialzisti presenti: {trade['bullish_confirmations']}")
                st.write(f"Segnali ribassisti presenti: {trade['bearish_confirmations']}")
                st.caption(f"Soglia richiesta: almeno {trade['min_confirmations']} conferme nella stessa direzione della rottura.")

            st.caption("⚠️ Segnale algoritmico basato su regole tecniche, non un consiglio di investimento.")

# --------------------------
# TAB 4: SCANNER
# --------------------------

with tab4:
    st.header("🧮 Scanner Automatico")

    st.subheader("Scanner V2 (Filtri avanzati)")

    universe_choice = st.multiselect(
        "Universo da scansionare",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"]
    )
    scan_timeframes = st.multiselect(
        "Timeframe da scansionare (invia un alert Telegram separato per ciascuno)",
        list(TIMEFRAME_OPTIONS.keys()),
        default=["Giornaliero"]
    )

    scan_tickers = []
    for cat in universe_choice:
        scan_tickers += list(TICKER_CATALOG[cat].values())
    scan_tickers = list(dict.fromkeys(scan_tickers))  # rimuove duplicati mantenendo l'ordine

    st.caption(f"{len(scan_tickers)} ticker selezionati, su {len(scan_timeframes) or 0} timeframe.")

    if st.button("Avvia Scanner V2"):
        if not scan_tickers:
            st.warning("Seleziona almeno un universo di titoli.")
        elif not scan_timeframes:
            st.warning("Seleziona almeno un timeframe.")
        else:
            all_results = []
            for tf_label in scan_timeframes:
                tf = TIMEFRAME_OPTIONS[tf_label]
                scan_period = "6mo" if tf["interval"] == "1d" else "60d"
                with st.spinner(f"Scansione {tf_label} in corso ({len(scan_tickers)} ticker)..."):
                    df_tf = run_scanner_v2(
                        tickers=scan_tickers,
                        interval=tf["interval"],
                        period=scan_period,
                        timeframe_label=tf_label
                    )
                if not df_tf.empty:
                    all_results.append(df_tf)

            if not all_results:
                st.warning("Nessun titolo soddisfa i criteri V2 su nessuno dei timeframe selezionati.")
            else:
                df = pd.concat(all_results, ignore_index=True).sort_values("Score", ascending=False)
                st.dataframe(df, use_container_width=True)

    st.subheader("Scanner V3 (Machine Learning)")
    st.caption("Lavora sempre su dati giornalieri: il training ML richiede più storia di quanta ne offra il timeframe a 4 ore.")

    if st.button("Avvia Scanner V3"):
        if not scan_tickers:
            st.warning("Seleziona almeno un universo di titoli qui sopra.")
        else:
            with st.spinner(f"Scansione + training modelli in corso su {len(scan_tickers)} ticker (può richiedere qualche minuto)..."):
                df3 = run_scanner_v3(tickers=scan_tickers)
            if df3.empty:
                st.warning("Nessun titolo disponibile per Scanner V3.")
            else:
                st.dataframe(df3, use_container_width=True)

    st.subheader("📊 Importanza delle Feature (Scanner V3)")

    if st.button("Mostra Feature Importance"):
        data_ref = download_data("ENI.MI", period="1y", interval="1d")
        if data_ref.empty:
            st.error("Nessun dato per ENI.MI.")
        else:
            features_ref, target_ref = build_ml_dataset(data_ref)
            if len(features_ref) < 200:
                st.warning("Dati insufficienti per ENI.MI.")
            else:
                model_ref, scaler_ref, feature_names_ref = train_ml_model(features_ref, target_ref)
                importance_df = get_feature_importance(model_ref, feature_names_ref)
                st.dataframe(importance_df, use_container_width=True)

# --------------------------
# TAB 5: BACKTEST
# --------------------------

with tab5:
    st.header("📜 Backtest")
    st.caption("Risultati storici passati. Non garantiscono risultati futuri. Il backtest lavora sempre su dati giornalieri (serve più storia di quanta ne offra il 4 ore), indipendentemente dal timeframe selezionato sopra.")

    st.subheader("Backtest dei Breakout")

    if st.button("Esegui Backtest Breakout"):
        data = download_data(ticker, period="2y", interval="1d")
        if data.empty:
            st.error("Nessun dato trovato.")
        else:
            df_bt = backtest_breakouts(data)
            st.dataframe(df_bt, use_container_width=True)
            if len(df_bt) > 0:
                success_rate = (df_bt["Successo"] == "✔️").mean() * 100
                st.subheader(f"📈 Percentuale di breakout riusciti: {success_rate:.2f}%")
            else:
                st.warning("Nessun breakout rilevato negli ultimi 2 anni.")

    st.subheader("🧪 Backtest dello Score")

    if st.button("Esegui Backtest Score"):
        data = download_data(ticker, period="2y", interval="1d")
        if data.empty:
            st.error("Nessun dato trovato.")
        else:
            df_bs = backtest_score(data)
            st.dataframe(df_bs, use_container_width=True)
            if len(df_bs) > 0:
                success_rate = (df_bs["Successo"] == "✔️").mean() * 100
                avg_perf = df_bs["Perf 20g"].mean()
                med_perf = df_bs["Perf 20g"].median()
                st.metric("Percentuale di Successo", f"{success_rate:.2f}%")
                st.metric("Profitto Medio (20g)", f"{avg_perf:.2f}%")
                st.metric("Profitto Mediano (20g)", f"{med_perf:.2f}%")
            else:
                st.warning("Nessun segnale di score ≥ 75 negli ultimi 2 anni.")
