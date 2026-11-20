# Pagina "Grafico": prezzo di un titolo con i periodi in cui la strategia
# lo teneva in portafoglio.
#   - Indici: Trend following 10 mesi
#   - Azioni italiane: Momentum 12-1 (serve scaricare tutte le azioni
#     italiane, perché la scelta dipende dalla classifica)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import TICKER_CATALOG, download_data
from strategies_pro import adjust_ohlc, target_trend10, targets_momentum, monthly_sma_above, extract_trades

st.set_page_config(page_title="Grafico", page_icon="📈", layout="wide")
st.title("📈 Grafico con le operazioni")

REGIME = "FTSEMIB.MI"


@st.cache_data(ttl=3600, show_spinner=False)
def get(ticker, period="10y"):
    for _ in range(3):
        try:
            raw = download_data(ticker, period=period, interval="1d")
            if raw is not None and not raw.empty:
                return adjust_ohlc(raw)
        except Exception:
            pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def momentum_targets():
    stocks = list(TICKER_CATALOG["Azioni Italiane"].values())
    data = {t: get(t) for t in stocks}
    data = {t: d for t, d in data.items() if d is not None and len(d) > 260}
    closes = pd.DataFrame({t: d["Close"] for t, d in data.items()}).sort_index()
    reg = get(REGIME)
    regime = monthly_sma_above(reg["Close"].reindex(closes.index).ffill())
    tgt, _ = targets_momentum(closes, regime)
    return tgt


groups = {"Indici (Trend following 10 mesi)": {**TICKER_CATALOG["Indici Americani"], **TICKER_CATALOG["Indici Europei"]},
          "Azioni italiane (Momentum 12-1)": TICKER_CATALOG["Azioni Italiane"]}
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    group = st.radio("Gruppo", list(groups))
with c2:
    name = st.selectbox("Titolo o indice", list(groups[group]))
with c3:
    years = st.selectbox("Anni", [1, 2, 3, 5, 10], index=2)
ticker = groups[group][name]

with st.spinner("Scarico i dati..." if group.startswith("Indici") else "Scarico tutte le azioni italiane per ricostruire la classifica (circa un minuto la prima volta)..."):
    df = get(ticker)
    if df is None or len(df) < 260:
        st.error("Dati non disponibili per questo titolo.")
        st.stop()
    if group.startswith("Indici"):
        tgt = target_trend10(df)
    else:
        tgt = momentum_targets().get(ticker)
        tgt = tgt.reindex(df.index).ffill().fillna(0) if tgt is not None else pd.Series(0.0, index=df.index)

trades = extract_trades(df, tgt, ticker)
view = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
vt = [t for t in trades if t["exit"] >= view.index[0]]
held = tgt.shift(1).fillna(0)
if held.iloc[-1] == 1:
    entry = held.index[(held.diff() == 1).to_numpy()]
    if len(entry):
        vt.append({"entry": entry[-1], "exit": None, "days": len(df.loc[entry[-1]:]),
                   "ret_pct": (df["Close"].iloc[-1] / df.loc[entry[-1], "Open"] - 1) * 100})

fig = go.Figure()
for t in vt:
    x0 = max(t["entry"], view.index[0])
    x1 = t["exit"] or view.index[-1]
    fig.add_vrect(x0=x0, x1=x1, fillcolor="#1a7f37", opacity=0.10, line_width=0)
fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="Prezzo", line=dict(color="#333", width=1.5)))
if group.startswith("Indici"):
    m_close = df["Close"][monthly_sma_above(df["Close"]).notna()]
    sma = m_close.rolling(10).mean().reindex(df.index).ffill().loc[view.index]
    fig.add_trace(go.Scatter(x=view.index, y=sma, name="Media 10 mesi", line=dict(color="#7b61ff", width=2, shape="hv")))
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.02, x=0),
                  yaxis_title="Prezzo (rettificato per dividendi)")
st.plotly_chart(fig, width="stretch")
st.caption("Le fasce verdi sono i periodi in cui la strategia teneva il titolo: acquisto e vendita all'apertura "
           "del giorno dopo il segnale. Prezzi rettificati per i dividendi, quindi possono differire da quelli del tuo broker.")

if vt:
    r = pd.Series([t["ret_pct"] for t in vt])
    m = st.columns(4)
    m[0].metric("Periodi in portafoglio", len(vt))
    m[1].metric("Positivi", f"{(r > 0).mean() * 100:.0f}%")
    m[2].metric("Risultato medio", f"{r.mean():+.1f}%")
    m[3].metric("Totale (composto)", f"{((1 + r / 100).prod() - 1) * 100:+.1f}%")
    st.dataframe(pd.DataFrame([{"Acquisto": t["entry"].strftime("%d/%m/%Y"),
                                "Vendita": t["exit"].strftime("%d/%m/%Y") if t["exit"] is not None else "in corso",
                                "Giorni": t["days"], "Risultato": f"{t['ret_pct']:+.1f}%"} for t in reversed(vt)]),
                 hide_index=True, width="stretch")
else:
    st.info("La strategia non ha tenuto questo titolo nel periodo mostrato.")
