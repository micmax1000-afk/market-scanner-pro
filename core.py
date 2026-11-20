# core.py — Market Scanner Pro V3 + ML: logica di calcolo condivisa
# Riusato sia da app.py (interfaccia Streamlit) sia da scheduled_scan.py
# (script eseguito da GitHub Actions per gli scanner automatici).

import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

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
# QUALI STRATEGIE MANDANO ALERT SU TELEGRAM
# ==========================
# Cambia qui True/False per attivare o disattivare l'alert di ogni
# strategia, senza toccare nient'altro nel codice. Vale sia per l'uso
# manuale dell'app sia per gli scanner automatici (GitHub Actions):
# è lo stesso file per entrambi.
#
# Attive solo le strategie che hanno passato il backtest realistico
# (vedi results/ e la pagina dei risultati).

ALERT_STRATEGY_CONFIG = {
    "trade_signal": False,          # Segnale combinato: positivo su 10 anni ma dal 2022 non distinguibile dal caso
    "trendline_break_raw": False,   # Rottura trend line senza conferme: mai testata da sola, solo informativa
    # Strategie di strategies_pro.py (vedi scheduled_scan.py)
    "mean_reversion_rsi2": True,
    "trend_10_mesi": True,
    "momentum_12_1": True,
}

# ==========================
# LISTA TITOLI ITALIANI
# ==========================

ITALIAN_TICKERS = [
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STLAM.MI", "LDO.MI", "RACE.MI",
    "MONC.MI", "TRN.MI", "PST.MI", "AMP.MI", "BMED.MI", "BAMI.MI", "BPSO.MI",
    "BPE.MI", "CNHI.MI", "DIA.MI", "ERG.MI", "EXO.MI", "G.MI", "IG.MI",
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

    # VWAP mobile 20 barre (prezzo medio ponderato per volume)
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
    # larghezza normalizzata sul prezzo medio: usata per rilevare lo "squeeze"
    # (bande vicine tra loro = bassa volatilità, spesso precursore di un breakout)
    data["BB_Width"] = band_width / data["BB_Mid"]
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

def compute_vwap(data, window=20):
    """VWAP mobile sulle ultime `window` barre. Prima era 'ancorato'
    all'inizio dei dati scaricati: il suo valore cambiava a seconda di
    quanto storico si scaricava (6 mesi, 2 anni, 10 anni), quindi lo
    stesso giorno poteva dare segnali diversi nell'app, nello scanner
    automatico e nel backtest. Con una finestra mobile il valore
    dipende solo dalle ultime `window` barre. Se il volume è sempre 0
    (alcuni indici su Yahoo) resta NaN e la conferma viene ignorata."""
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    vol = data["Volume"]
    sum_vol = vol.rolling(window).sum()
    sum_tp_vol = (typical_price * vol).rolling(window).sum()
    data["VWAP"] = sum_tp_vol / sum_vol.replace(0, np.nan)
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

def _trade_signal_components(data):
    """Calcola rotture e conferme una sola volta (indipendentemente
    dalla soglia). Usato sia da compute_trade_signal (soglia singola,
    uso live) sia dal backtest multi-soglia, per non ripetere 3 volte
    lo stesso calcolo costoso (ricerca dei pivot)."""
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

    return {
        "resistance_break": swing_brk["resistance_break"],
        "support_break": swing_brk["support_break"],
        "bullish": bullish,
        "bearish": bearish,
    }

def _signal_from_components(comp, min_confirmations):
    if comp["resistance_break"] and len(comp["bullish"]) >= min_confirmations:
        return "ACQUISTO"
    if comp["support_break"] and len(comp["bearish"]) >= min_confirmations:
        return "VENDITA"
    return "NEUTRALE"

def compute_trade_signal(data, min_confirmations=3):
    comp = _trade_signal_components(data)
    signal = _signal_from_components(comp, min_confirmations)
    confirmations = comp["bullish"] if signal == "ACQUISTO" else (comp["bearish"] if signal == "VENDITA" else [])

    return {
        "signal": signal,
        "resistance_break": comp["resistance_break"],
        "support_break": comp["support_break"],
        "bullish_confirmations": comp["bullish"],
        "bearish_confirmations": comp["bearish"],
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
# SUPERTREND
# ==========================
# Indicatore trend-following basato sull'ATR: traccia una linea di
# supporto sotto il prezzo quando il trend è rialzista (la linea sale
# seguendo il prezzo ma non scende mai finché il trend regge), e sopra
# quando è ribassista. Usato qui SOLO come stop loss dinamico
# (trailing) alternativo a quello fisso a multiplo di ATR nel backtest
# SL/TP: invece di uno stop fermo dal giorno di ingresso, "segue" il
# prezzo verso l'alto stringendosi man mano che il trend si sviluppa.

def compute_supertrend(data, period=10, multiplier=3.0):
    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    atr = compute_atr(data, period=period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    n = len(data)
    supertrend = pd.Series(index=data.index, dtype=float)
    direction = pd.Series(index=data.index, dtype=float)  # 1 = rialzista, -1 = ribassista

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, n):
        # bootstrap: finché il valore precedente è NaN (riscaldamento ATR),
        # la banda parte semplicemente dal valore di oggi, altrimenti la
        # regola ricorsiva confrontata con un NaN non si aggiornerebbe mai
        # più (resterebbe bloccata a NaN per sempre)
        if pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = upper_band.iloc[i]
        elif upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if pd.isna(final_lower.iloc[i - 1]):
            final_lower.iloc[i] = lower_band.iloc[i]
        elif lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if pd.isna(final_upper.iloc[i]) or pd.isna(final_lower.iloc[i]):
            continue  # ancora in riscaldamento: direction/supertrend restano NaN

        if pd.isna(supertrend.iloc[i - 1]):
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
        elif direction.iloc[i - 1] == 1:
            direction.iloc[i] = -1 if close.iloc[i] < final_lower.iloc[i] else 1
        else:
            direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1

        supertrend.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    data = data.copy()
    data["Supertrend"] = supertrend
    data["Supertrend_Direction"] = direction
    return data

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
    # BUGFIX: la resistenza deve essere il massimo dei 60 giorni PRIMA di
    # oggi, non inclusi oggi. Includendo oggi, resistance >= High di oggi
    # >= Close di oggi sempre per definizione, quindi breakout_ratio non
    # può quasi mai essere positivo: il componente "breakout" dello score
    # e questo filtro erano di fatto sempre a zero.
    recent = data.iloc[-61:-1] if len(data) > 60 else data.iloc[:-1]
    if recent.empty:
        return 0
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
# SCORE PESATO V4 (sperimentale: consolida la ridondanza del trend)
# ==========================
# Nello Score V3, 4 componenti su 11 misurano tutte, da angolazioni
# diverse, la stessa cosa: "il titolo è in un trend rialzista forte"
# (allineamento EMA, pendenza della regressione a 60gg, rottura della
# trend line sui pivot, ADX) — insieme pesano il 42% dello score
# finale. Un titolo in trend pulito guadagna punti 4 volte per lo
# stesso fenomeno, non 4 conferme davvero indipendenti.
#
# Lo Score V4 le consolida in UNA sola componente "forza del trend"
# (media delle 4 normalizzate), pesata una sola volta, e ridistribuisce
# il peso liberato su volume, breakout e Bollinger: dimensioni più
# indipendenti dal trend puro (liquidità/partecipazione, estensione di
# prezzo, volatilità). Non sostituisce lo Score V3 (che resta invariato
# ovunque nell'app): è pensato per essere confrontato fianco a fianco.

def compute_trend_strength_score(data, last):
    """Media di 4 misure di forza del trend, ciascuna normalizzata sul
    proprio massimo (0-1), poi riportata in scala 0-100 come le altre
    componenti dello score."""
    components_norm = [
        trend_score(last) / 20,
        trendline_score_advanced(data) / 15,
        trendline_break_score(data) / 10,
        adx_score(data) / 10,
    ]
    return round(sum(components_norm) / len(components_norm) * 100, 2)

def compute_weighted_score_v4(data):
    last = data.iloc[-1]

    trend_strength = compute_trend_strength_score(data, last)

    # (punteggio_grezzo, punteggio_massimo_possibile, peso)
    components = {
        "trend_strength":   (trend_strength, 100, 0.20),  # consolidata (era 0.16+0.06+0.10+0.10=0.42)
        "momentum":         (momentum_score(last), 15, 0.06),
        "macd":             (macd_score(last), 15, 0.06),
        "volume":           (volume_score_advanced(data), 20, 0.20),   # era 0.13
        "breakout":         (breakout_score_advanced(data), 15, 0.20),  # era 0.13
        "stochastic":       (stochastic_score(data), 10, 0.04),
        "bollinger":        (bollinger_score(data), 8, 0.18),          # era 0.10
        "vwap":             (vwap_score(data), 5, 0.06),
    }

    weighted = 0.0
    for raw_score, max_score, weight in components.values():
        normalized = (raw_score / max_score) if max_score else 0
        weighted += normalized * weight * 100

    return round(min(weighted, 100), 2)

def compute_score_v4(data):
    return compute_weighted_score_v4(data)

# ==========================
# FILTRI SCANNER V2
# ==========================

def filter_atr(data, max_atr_ratio=0.03):
    atr = compute_atr(data).iloc[-1]
    price = data["Close"].iloc[-1]
    if pd.isna(atr):
        return False
    return atr <= price * max_atr_ratio

def filter_volatility(data, max_volatility=0.08):
    recent = data.tail(20)
    high = recent["High"].max()
    low = recent["Low"].min()
    volatility = (high - low) / recent["Close"].iloc[-1]
    return volatility <= max_volatility

def filter_volume_spike(data, min_ratio=1.0):
    vol20 = data["Volume"].rolling(20).mean().iloc[-1]
    last_vol = data["Volume"].iloc[-1]
    if not vol20 or pd.isna(vol20):
        return False
    return last_vol > vol20 * min_ratio

def filter_trendline(data):
    recent = data.tail(60)
    if len(recent) < 2:
        return False
    x = np.arange(len(recent))
    y = recent["Close"].values
    m, q = np.polyfit(x, y, 1)
    return m > 0

def filter_breakout(data, min_breakout_ratio=0.2):
    # Stesso bugfix di breakout_score_advanced: resistenza sui 60gg
    # PRIMA di oggi, non inclusi oggi.
    recent = data.iloc[-61:-1] if len(data) > 60 else data.iloc[:-1]
    if recent.empty:
        return False
    resistance = recent["High"].max()
    close = data["Close"].iloc[-1]
    atr = compute_atr(data).iloc[-1]
    if atr == 0 or pd.isna(atr):
        return False
    breakout_ratio = (close - resistance) / atr
    return breakout_ratio > min_breakout_ratio

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


    ref_index = get_reference_index(ticker)
    market_regime = compute_market_regime(ref_index) if ref_index else None

    has_trade_signal = trade["signal"] != "NEUTRALE"

    # L'alert scatta solo per le strategie attivate in ALERT_STRATEGY_CONFIG
    # (in cima al file): cambia True/False lì per decidere quali strategie
    # devono generare notifica, senza toccare il resto del codice.
    strategy_triggers = {
        "trade_signal": has_trade_signal,
        "trendline_break_raw": resistance_break or support_break,
    }
    if not any(ALERT_STRATEGY_CONFIG.get(key, False) and fired for key, fired in strategy_triggers.items()):
        return False

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

    strategy_lines = ""
    # Rottura trend line senza conferme: mostrata solo se non è già coperta
    # dal segnale combinato (che include la stessa rottura MA richiede
    # anche le conferme) — evita di duplicare la stessa informazione con
    # due diciture diverse nello stesso messaggio.
    if (resistance_break or support_break) and not has_trade_signal:
        direction = "rialzista (sopra resistenza)" if resistance_break else "ribassista (sotto supporto)"
        strategy_lines += f"🔀 <b>ROTTURA TREND LINE</b> — {direction}, senza conferme aggiuntive (solo informativo)\n"
    if strategy_lines:
        strategy_lines += "\n"

    regime_str = {"Rialzista": "🟢 Rialzista", "Ribassista": "🔴 Ribassista"}.get(market_regime, "n/d")

    msg = (
        f"🚨 <b>ALERT V2: {ticker}</b> ({timeframe_label})\n"
        f"{signal_header}"
        f"{strategy_lines}"
        f"Score: {score}/100\n"
        f"Prezzo: {round(last['Close'], 2)}\n\n"
        f"📈 Breakout ATR: {breakout}\n"
        f"📊 Volume Spike: {vol_spike}\n"
        f"{trendline_line}\n"
        f"🎯 Stocastico K/D: {round(stoch_k,2)} / {round(stoch_d,2)}\n"
        f"💪 ADX: {adx_str}\n"
        f"💰 Sopra VWAP: {'sì' if above_vwap else 'no'}\n"
        f"🌍 Regime di mercato: {regime_str}\n\n"
        f"⚠️ Segnale algoritmico, non è un consiglio di investimento."
    )

    send_telegram_message(BOT_TOKEN, CHAT_ID, msg)
    return True

# ==========================
# SCANNER V2
# ==========================

# ==========================
# REGIME DI MERCATO (informativo, non filtra nulla automaticamente)
# ==========================
# Un titolo può dare un segnale tecnico valido sul suo grafico e ciò
# nonostante muoversi contro l'onda di fondo se il mercato generale è
# in un trend ribassista. Questa colonna mostra, per riferimento, se
# l'indice associato al titolo è sopra o sotto la propria media mobile
# a 200 giorni (una misura standard e grezza di "il mercato di fondo è
# rialzista o ribassista"). È solo un'informazione in più da valutare,
# NON filtra o blocca nessun segnale.

def get_reference_index(ticker):
    """Associa a ogni ticker l'indice più adatto a rappresentare il
    'mercato generale' in cui si muove."""
    if ticker in ITALIAN_TICKERS or ticker == "FTSEMIB.MI":
        return "FTSEMIB.MI"
    if ticker in US_INDICES.values():
        return "^GSPC"  # S&P 500 come benchmark ampio USA
    if ticker in EU_INDICES.values():
        return "^STOXX50E"  # Euro Stoxx 50 come benchmark ampio Europa
    return None  # ticker personalizzato non riconosciuto: nessun riferimento

def compute_market_regime(reference_ticker, period="2y"):
    """'Rialzista' se l'indice di riferimento chiude sopra la propria
    media mobile semplice a 200 giorni, 'Ribassista' altrimenti, None
    se non ci sono abbastanza dati."""
    if reference_ticker is None:
        return None
    data = download_data(reference_ticker, period=period, interval="1d")
    if data.empty or len(data) < 200:
        return None
    sma200 = data["Close"].rolling(200).mean()
    last_close = data["Close"].iloc[-1]
    last_sma = sma200.iloc[-1]
    if pd.isna(last_sma):
        return None
    return "Rialzista" if last_close > last_sma else "Ribassista"

# ==========================
# MOMENTUM RANKING (fattore cross-sezionale, stile fondi sistematici)
# ==========================
# A differenza delle strategie sopra (segnale sì/no sul singolo
# titolo), questo è un fattore: classifica TUTTO l'universo per
# rendimento storico su un periodo (default 6 mesi) e mostra i
# migliori. È il fattore "momentum" documentato in letteratura
# accademica (Jegadeesh-Titman e centinaia di studi successivi) e
# usato su vasta scala da fondi sistematici/CTA — con la differenza
# che loro applicano anche position sizing su volatilità, gestione
# del rischio e diversificazione su decine di mercati, cose che qui
# non replichiamo: è solo il ranking di base.

def compute_momentum_ranking(tickers, lookback_days=126, period="1y", interval="1d"):
    """Per ogni ticker calcola il rendimento totale sugli ultimi
    `lookback_days` giorni di trading (default ~126 = 6 mesi borsistici)
    ed ordina dal migliore al peggiore."""
    results = []
    for ticker in tickers:
        try:
            data = download_data(ticker, period=period, interval=interval)
            if data.empty or len(data) <= lookback_days:
                continue
            close_now = data["Close"].iloc[-1]
            close_then = data["Close"].iloc[-(lookback_days + 1)]
            if pd.isna(close_now) or pd.isna(close_then) or close_then == 0:
                continue
            momentum_return = (close_now - close_then) / close_then * 100
            results.append({
                "Ticker": ticker,
                "Prezzo": round(close_now, 2),
                f"Rendimento {lookback_days}gg %": round(momentum_return, 2),
            })
        except Exception as e:
            print(f"Errore momentum ranking su {ticker}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(f"Rendimento {lookback_days}gg %", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", range(1, len(df) + 1))
    return df

def run_scanner_v2(tickers=None, interval="1d", period=None, timeframe_label="Giornaliero",
                    max_volatility=0.08, min_volume_ratio=1.0, min_breakout_ratio=0.2):
    if tickers is None:
        tickers = ITALIAN_TICKERS
    if period is None:
        # 2 anni (~500 barre): servono almeno 200 barre per l'EMA200, con
        # 6 mesi (~125 barre) ogni titolo veniva scartato e non partiva
        # nessun alert
        period = "2y" if interval == "1d" else "60d"

    results = []
    analyzed = 0
    alerts_sent = 0
    skipped_no_data = []
    regime_cache = {}  # evita di riscaricare lo stesso indice per ogni titolo

    for ticker in tickers:
        try:
            data = download_data(ticker, period=period, interval=interval)

            min_bars = 200 if interval == "1d" else 80
            if data.empty or len(data) < min_bars:
                skipped_no_data.append(ticker)
                continue

            analyzed += 1
            data = compute_indicators(data)
            score = compute_score(data)

            # L'alert Telegram viene valutato per OGNI titolo scaricato,
            # indipendentemente dai filtri della tabella qui sotto: i 7
            # filtri sotto sono pensati per un breakout "classico", ma le
            # strategie personalizzate (es. pullback con momentum ancora
            # basso) cercano condizioni opposte e non li passerebbero mai,
            # perdendo l'alert anche quando quella strategia dà ACQUISTO.
            # send_alert_v2() decide da sola, al suo interno, se le
            # condizioni (di una qualsiasi delle strategie) meritano un
            # messaggio.
            if send_alert_v2(ticker, data, score, timeframe_label=timeframe_label):
                alerts_sent += 1

            if not filter_ema(data):
                continue
            if not filter_trendline(data):
                continue
            if not filter_volume_spike(data, min_ratio=min_volume_ratio):
                continue
            if not filter_breakout(data, min_breakout_ratio=min_breakout_ratio):
                continue
            if not filter_atr(data):
                continue
            if not filter_volatility(data, max_volatility=max_volatility):
                continue
            if not filter_stochastic(data):
                continue

            last = data.iloc[-1]
            atr_val = compute_atr(data).iloc[-1]
            resistance_window = data.iloc[-61:-1] if len(data) > 60 else data.iloc[:-1]
            resistance = resistance_window["High"].max() if not resistance_window.empty else np.nan
            breakout_ratio = (last["Close"] - resistance) / atr_val if atr_val and not pd.isna(atr_val) and atr_val != 0 and not pd.isna(resistance) else 0
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

            ref_index = get_reference_index(ticker)
            if ref_index is not None:
                if ref_index not in regime_cache:
                    regime_cache[ref_index] = compute_market_regime(ref_index)
                market_regime = regime_cache[ref_index]
            else:
                market_regime = None

            results.append({
                "Ticker": ticker,
                "Timeframe": timeframe_label,
                "Prezzo": round(last["Close"], 2),
                "Score": score,
                "Score V4": compute_score_v4(data),
                "Segnale": trade_signal,
                "RSI": round(last["RSI"], 2),
                "MACD": round(last["MACD"], 4),
                "ATR": round(atr_val, 4) if not pd.isna(atr_val) else None,
                "Breakout Ratio": round(breakout_ratio, 2),
                "ADX": round(adx_val, 1) if not pd.isna(adx_val) else None,
                "Rottura Trendline": trendline_flag,
                "Sopra VWAP": "✔️" if above_vwap else "",
                "Regime Mercato": market_regime or "n/d"
            })

        except Exception as e:
            print(f"Errore su {ticker}: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("Score", ascending=False)
    df.attrs["stats"] = {
        "analyzed": analyzed,
        "alerts_sent": alerts_sent,
        "skipped_no_data": skipped_no_data,
    }
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

# Le strategie a regole fisse (pullback ipervenduto, pullback EMA20, Turtle,
# candlestick, triangolo, bull flag) sono state rimosse: col backtest
# realistico (backtest_v2.py, 10 anni) perdevano tutte dopo i costi.
# Le strategie attuali sono in strategies_pro.py.

def backtest_strategy_signal(data, strategy_fn, forward_days=(5, 10, 20), min_history=200):
    """Walk-forward generico: ogni giorno vede solo il passato, applica
    `strategy_fn` (una delle funzioni compute_strategy_*) e, se dà
    ACQUISTO, registra la performance reale nei giorni successivi."""
    data_ind = compute_indicators(data)
    n = len(data_ind)
    records = []

    for i in range(min_history, n):
        window = data_ind.iloc[:i + 1]
        if len(window) < 90:
            continue

        result = strategy_fn(window)
        if result["signal"] != "ACQUISTO":
            continue

        close_today = window["Close"].iloc[-1]
        perf = {}
        for fd in forward_days:
            if i + fd < n:
                perf[fd] = (data_ind["Close"].iloc[i + fd] - close_today) / close_today * 100

        records.append({"date": window.index[-1], "perf": perf})

    return records

def summarize_strategy_backtest(records_by_ticker, horizon=10):
    all_records = []
    for per_ticker in records_by_ticker.values():
        all_records.extend(per_ticker)

    perf_values = [r["perf"][horizon] for r in all_records if horizon in r["perf"]]

    if not perf_values:
        return {
            "Segnali totali": 0, "Successo %": None,
            "Perf media %": None, "Perf mediana %": None,
            "Perf migliore %": None, "Perf peggiore %": None,
        }

    return {
        "Segnali totali": len(perf_values),
        "Successo %": round(float(np.mean([p > 0 for p in perf_values])) * 100, 1),
        "Perf media %": round(float(np.mean(perf_values)), 2),
        "Perf mediana %": round(float(np.median(perf_values)), 2),
        "Perf migliore %": round(float(np.max(perf_values)), 2),
        "Perf peggiore %": round(float(np.min(perf_values)), 2),
    }

# ==========================
# BACKTEST REALISTICO CON STOP LOSS / TAKE PROFIT (basati su ATR)
# ==========================
# A differenza dei backtest "a orizzonte fisso" sopra (che guardano solo
# se il prezzo è più alto dopo N giorni, ignorando cosa succede nel
# mezzo), questo simula un trade vero: entra al segnale, esce in perdita
# controllata se tocca lo stop loss, esce in guadagno se tocca il take
# profit, altrimenti chiude alla scadenza del periodo massimo. SL/TP
# sono espressi come multipli dell'ATR (volatilità), non valori fissi,
# così si adattano automaticamente a ogni titolo.
#
# Se nello stesso giorno il prezzo tocca sia lo stop che il target,
# per prudenza si assume che lo stop sia stato colpito per primo
# (non sappiamo l'ordine intra-day dei prezzi).

def backtest_strategy_sl_tp(data, strategy_fn, atr_sl_mult=1.5, atr_tp_mult=3.0,
                             max_holding_days=20, min_history=200):
    data_ind = compute_indicators(data)
    atr_series = compute_atr(data_ind)
    n = len(data_ind)
    trades = []

    for i in range(min_history, n):
        window = data_ind.iloc[:i + 1]
        if len(window) < 90:
            continue

        result = strategy_fn(window)
        if result.get("signal") != "ACQUISTO":
            continue

        entry_price = data_ind["Close"].iloc[i]
        atr_val = atr_series.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        risk_per_share = atr_sl_mult * atr_val
        stop_price = entry_price - risk_per_share
        target_price = entry_price + atr_tp_mult * atr_val

        outcome, exit_price, days_held = None, None, None

        for offset in range(1, max_holding_days + 1):
            j = i + offset
            if j >= n:
                break
            day_high = data_ind["High"].iloc[j]
            day_low = data_ind["Low"].iloc[j]
            hit_sl = day_low <= stop_price
            hit_tp = day_high >= target_price

            if hit_sl:  # copre anche il caso in cui entrambi vengono toccati lo stesso giorno
                outcome, exit_price, days_held = "stop_loss", stop_price, offset
                break
            elif hit_tp:
                outcome, exit_price, days_held = "take_profit", target_price, offset
                break

        if outcome is None:
            j = min(i + max_holding_days, n - 1)
            if j <= i:
                continue
            outcome = "timeout"
            exit_price = data_ind["Close"].iloc[j]
            days_held = j - i

        r_multiple = (exit_price - entry_price) / risk_per_share
        trades.append({
            "date": window.index[-1],
            "outcome": outcome,
            "r_multiple": r_multiple,
            "days_held": days_held,
        })

    return trades

# ==========================
# BACKTEST CON STOP LOSS DINAMICO (SUPERTREND) — variante di quello sopra
# ==========================
# Stessa logica di ingresso/take profit del backtest sopra, ma lo stop
# loss non è più fisso al multiplo di ATR: segue il Supertrend, che si
# stringe verso l'alto mano a mano che il trend si sviluppa (o esce
# subito se il Supertrend "flippa" da rialzista a ribassista). L'idea è
# lasciare correre i guadagni invece di uscire sempre alla stessa
# distanza fissa dall'ingresso.

def backtest_strategy_sl_tp_supertrend(data, strategy_fn, atr_tp_mult=3.0,
                                        supertrend_period=10, supertrend_multiplier=3.0,
                                        max_holding_days=20, min_history=200):
    data_ind = compute_indicators(data)
    data_st = compute_supertrend(data_ind, period=supertrend_period, multiplier=supertrend_multiplier)
    atr_series = compute_atr(data_ind)
    n = len(data_ind)
    trades = []

    for i in range(min_history, n):
        window = data_ind.iloc[:i + 1]
        if len(window) < 90:
            continue

        result = strategy_fn(window)
        if result.get("signal") != "ACQUISTO":
            continue

        entry_price = data_ind["Close"].iloc[i]
        atr_val = atr_series.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        initial_stop = data_st["Supertrend"].iloc[i]
        if pd.isna(initial_stop) or initial_stop >= entry_price:
            continue  # Supertrend non valido o già ribassista: nessun rischio ben definito
        risk_per_share = entry_price - initial_stop
        if risk_per_share <= 0:
            continue

        target_price = entry_price + atr_tp_mult * atr_val

        outcome, exit_price, days_held = None, None, None

        for offset in range(1, max_holding_days + 1):
            j = i + offset
            if j >= n:
                break
            day_high = data_ind["High"].iloc[j]
            day_low = data_ind["Low"].iloc[j]
            st_value = data_st["Supertrend"].iloc[j]
            st_direction = data_st["Supertrend_Direction"].iloc[j]

            hit_tp = day_high >= target_price
            # stop dinamico: il trend è "flippato" ribassista, oppure il
            # prezzo ha rotto sotto la linea del Supertrend di oggi
            flipped_bearish = (not pd.isna(st_direction)) and st_direction == -1
            broke_below = (not pd.isna(st_value)) and day_low <= st_value
            hit_trailing_stop = flipped_bearish or broke_below

            if hit_trailing_stop:  # priorità allo stop, stesso criterio prudente del backtest fisso
                outcome = "stop_loss"
                exit_price = st_value if not pd.isna(st_value) else data_ind["Close"].iloc[j]
                days_held = offset
                break
            elif hit_tp:
                outcome, exit_price, days_held = "take_profit", target_price, offset
                break

        if outcome is None:
            j = min(i + max_holding_days, n - 1)
            if j <= i:
                continue
            outcome = "timeout"
            exit_price = data_ind["Close"].iloc[j]
            days_held = j - i

        r_multiple = (exit_price - entry_price) / risk_per_share
        trades.append({
            "date": window.index[-1],
            "outcome": outcome,
            "r_multiple": r_multiple,
            "days_held": days_held,
        })

    return trades

def summarize_sl_tp_backtest(trades_by_ticker):
    all_trades = []
    for per_ticker in trades_by_ticker.values():
        all_trades.extend(per_ticker)

    if not all_trades:
        return {
            "Trade totali": 0, "Take Profit %": None, "Stop Loss %": None, "Timeout %": None,
            "R medio": None, "Profit Factor": None, "Giorni medi in trade": None,
            "Drawdown massimo (R)": None, "Perdite consecutive max": None,
        }

    n_trades = len(all_trades)
    wins = [t for t in all_trades if t["outcome"] == "take_profit"]
    losses = [t for t in all_trades if t["outcome"] == "stop_loss"]
    timeouts = [t for t in all_trades if t["outcome"] == "timeout"]

    r_values = [t["r_multiple"] for t in all_trades]
    gross_win = sum(r for r in r_values if r > 0)
    gross_loss = -sum(r for r in r_values if r < 0)

    # Drawdown massimo e perdite consecutive: calcolati mettendo in fila
    # TUTTI i trade (di qualsiasi titolo) ordinati per data di ingresso, come
    # se fossero un'unica sequenza di operazioni. È una semplificazione (se
    # due titoli segnalano lo stesso giorno, in realtà li apriresti insieme,
    # non in sequenza), ma dà comunque un'idea concreta della "peggiore
    # striscia" che il sistema ha attraversato nel periodo testato.
    sorted_trades = sorted(all_trades, key=lambda t: t["date"])

    cum_r = 0.0
    peak_r = 0.0
    max_drawdown = 0.0
    for t in sorted_trades:
        cum_r += t["r_multiple"]
        peak_r = max(peak_r, cum_r)
        max_drawdown = max(max_drawdown, peak_r - cum_r)

    max_consecutive_losses = 0
    current_streak = 0
    for t in sorted_trades:
        if t["r_multiple"] < 0:
            current_streak += 1
            max_consecutive_losses = max(max_consecutive_losses, current_streak)
        else:
            current_streak = 0

    return {
        "Trade totali": n_trades,
        "Take Profit %": round(len(wins) / n_trades * 100, 1),
        "Stop Loss %": round(len(losses) / n_trades * 100, 1),
        "Timeout %": round(len(timeouts) / n_trades * 100, 1),
        "R medio": round(float(np.mean(r_values)), 2),
        "Profit Factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
        "Giorni medi in trade": round(float(np.mean([t["days_held"] for t in all_trades])), 1),
        "Drawdown massimo (R)": round(max_drawdown, 2),
        "Perdite consecutive max": max_consecutive_losses,
    }

def backtest_trade_signal(data, thresholds=(2, 3, 4), forward_days=(5, 10, 20), min_history=200):
    """Cammina nel tempo (walk-forward, nessun lookahead: ogni giorno
    vede solo il passato fino a quel momento) e per ogni soglia di
    conferme registra ogni segnale ACQUISTO/VENDITA generato, insieme
    alla performance reale del prezzo nei giorni successivi."""
    data_ind = compute_indicators(data)
    n = len(data_ind)
    records = {th: [] for th in thresholds}

    for i in range(min_history, n):
        window = data_ind.iloc[:i + 1]
        if len(window) < 90:
            continue

        comp = _trade_signal_components(window)
        if not comp["resistance_break"] and not comp["support_break"]:
            continue  # nessuna rottura oggi: irrilevante per qualsiasi soglia

        close_today = window["Close"].iloc[-1]
        perf = {}
        for fd in forward_days:
            if i + fd < n:
                perf[fd] = (data_ind["Close"].iloc[i + fd] - close_today) / close_today * 100

        for th in thresholds:
            sig = _signal_from_components(comp, th)
            if sig == "NEUTRALE":
                continue
            records[th].append({"date": window.index[-1], "signal": sig, "perf": perf})

    return records

def summarize_trade_backtest(records_by_ticker, horizon=10):
    """records_by_ticker: {ticker: {soglia: [record, ...]}} (output di
    backtest_trade_signal per più titoli). Aggrega su tutti i titoli e
    calcola, per ogni soglia, la percentuale di segnali 'riusciti'
    (ACQUISTO seguito da prezzo più alto, VENDITA seguito da prezzo più
    basso dopo `horizon` giorni)."""
    all_thresholds = set()
    for per_ticker in records_by_ticker.values():
        all_thresholds.update(per_ticker.keys())

    summary = []
    for th in sorted(all_thresholds):
        buy, sell = [], []
        for per_ticker in records_by_ticker.values():
            for rec in per_ticker.get(th, []):
                if horizon not in rec["perf"]:
                    continue
                (buy if rec["signal"] == "ACQUISTO" else sell).append(rec["perf"][horizon])

        buy_success = round(np.mean([p > 0 for p in buy]) * 100, 1) if buy else None
        sell_success = round(np.mean([p < 0 for p in sell]) * 100, 1) if sell else None
        buy_avg = round(float(np.mean(buy)), 2) if buy else None
        sell_avg = round(float(np.mean(sell)), 2) if sell else None
        total = len(buy) + len(sell)
        overall_success = None
        if total:
            hits = sum(1 for p in buy if p > 0) + sum(1 for p in sell if p < 0)
            overall_success = round(hits / total * 100, 1)

        summary.append({
            "Soglia conferme": th,
            "Segnali totali": total,
            "Successo complessivo %": overall_success,
            "Segnali ACQUISTO": len(buy),
            "Successo ACQUISTO %": buy_success,
            "Perf media ACQUISTO %": buy_avg,
            "Segnali VENDITA": len(sell),
            "Successo VENDITA %": sell_success,
            "Perf media VENDITA %": sell_avg,
        })

    return pd.DataFrame(summary)

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

