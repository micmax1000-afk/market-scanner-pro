# app.py — Market Scanner Pro V3 + ML (interfaccia Streamlit)
#
# Tutta la logica di calcolo (indicatori, strategie, scanner, catalogo
# ticker, invio Telegram) vive in core.py, così può essere riusata sia
# qui sia dallo script schedulato (scheduled_scan.py) eseguito da
# GitHub Actions per gli scanner automatici Lun-Ven.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from core import *  # noqa: F401,F403 — tutte le funzioni/costanti di calcolo

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

            st.subheader("📉 Strategia: Rottura resistenza discendente con momentum basso")
            pullback_os = compute_strategy_pullback_oversold(data)
            if pullback_os["signal"] == "ACQUISTO":
                st.success("📈 ACQUISTO — tutte le condizioni soddisfatte")
            else:
                st.info("⏸️ Condizioni non tutte soddisfatte")
            for desc, met in pullback_os["conditions"]:
                st.write(f"{'✅' if met else '❌'} {desc}")

            st.subheader("📈 Strategia: Pullback alla EMA20 in trend forte")
            pullback_trend = compute_strategy_trend_pullback(data)
            if pullback_trend["signal"] == "ACQUISTO":
                st.success("📈 ACQUISTO — tutte le condizioni soddisfatte")
            else:
                st.info("⏸️ Condizioni non tutte soddisfatte")
            for desc, met in pullback_trend["conditions"]:
                st.write(f"{'✅' if met else '❌'} {desc}")

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

    st.subheader("🚦 Backtest del Segnale ACQUISTO/VENDITA (confronto soglie)")
    st.caption("Testa il segnale su più titoli contemporaneamente con soglie di conferma diverse (2, 3, 4), per capire quale funziona meglio storicamente.")

    bt_universe = st.multiselect(
        "Universo per il backtest del segnale",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="bt_signal_universe"
    )
    bt_horizon = st.selectbox("Orizzonte per valutare il successo (giorni dopo il segnale)", [5, 10, 20], index=1)

    if st.button("Esegui Backtest Segnale"):
        bt_tickers = []
        for cat in bt_universe:
            bt_tickers += list(TICKER_CATALOG[cat].values())
        bt_tickers = list(dict.fromkeys(bt_tickers))

        if not bt_tickers:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            records_by_ticker = {}
            progress = st.progress(0.0)
            for idx, tkr in enumerate(bt_tickers):
                try:
                    hist = download_data(tkr, period="2y", interval="1d")
                    if not hist.empty and len(hist) >= 250:
                        records_by_ticker[tkr] = backtest_trade_signal(hist, thresholds=(2, 3, 4))
                except Exception as e:
                    print(f"Errore backtest segnale su {tkr}: {e}")
                progress.progress((idx + 1) / len(bt_tickers))

            if not records_by_ticker:
                st.warning("Nessun dato sufficiente per il backtest sui titoli selezionati.")
            else:
                summary_df = summarize_trade_backtest(records_by_ticker, horizon=bt_horizon)
                st.dataframe(summary_df, use_container_width=True)
                st.caption(
                    f"'Successo' = il prezzo si è mosso nella direzione prevista dal segnale entro {bt_horizon} giorni "
                    "(per ACQUISTO: prezzo più alto; per VENDITA: prezzo più basso). "
                    "Un numero basso di 'Segnali totali' rende la percentuale poco affidabile: preferisci soglie con più segnali a parità di successo."
                )

    st.subheader("📉 Backtest: Rottura resistenza discendente con momentum basso (la tua)")
    pb_os_universe = st.multiselect(
        "Universo per il backtest",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="bt_pullback_os_universe"
    )
    pb_os_horizon = st.selectbox("Orizzonte (giorni)", [5, 10, 20], index=1, key="bt_pullback_os_horizon")

    if st.button("Esegui Backtest Pullback Ipervenduto"):
        tickers_pb = list(dict.fromkeys(t for cat in pb_os_universe for t in TICKER_CATALOG[cat].values()))
        if not tickers_pb:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            records_by_ticker = {}
            progress = st.progress(0.0)
            for idx, tkr in enumerate(tickers_pb):
                try:
                    hist = download_data(tkr, period="2y", interval="1d")
                    if not hist.empty and len(hist) >= 250:
                        records_by_ticker[tkr] = backtest_strategy_signal(hist, compute_strategy_pullback_oversold)
                except Exception as e:
                    print(f"Errore backtest pullback ipervenduto su {tkr}: {e}")
                progress.progress((idx + 1) / len(tickers_pb))

            summary = summarize_strategy_backtest(records_by_ticker, horizon=pb_os_horizon)
            st.write(summary)
            if summary["Segnali totali"] == 0:
                st.warning("Nessun segnale trovato in questo universo/periodo: la strategia richiede molte condizioni simultanee, è normale che sia rara.")

    st.subheader("📈 Backtest: Pullback alla EMA20 in trend forte (proposta)")
    pb_tr_universe = st.multiselect(
        "Universo per il backtest",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="bt_pullback_trend_universe"
    )
    pb_tr_horizon = st.selectbox("Orizzonte (giorni)", [5, 10, 20], index=1, key="bt_pullback_trend_horizon")

    if st.button("Esegui Backtest Pullback Trend"):
        tickers_pt = list(dict.fromkeys(t for cat in pb_tr_universe for t in TICKER_CATALOG[cat].values()))
        if not tickers_pt:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            records_by_ticker = {}
            progress = st.progress(0.0)
            for idx, tkr in enumerate(tickers_pt):
                try:
                    hist = download_data(tkr, period="2y", interval="1d")
                    if not hist.empty and len(hist) >= 250:
                        records_by_ticker[tkr] = backtest_strategy_signal(hist, compute_strategy_trend_pullback)
                except Exception as e:
                    print(f"Errore backtest pullback trend su {tkr}: {e}")
                progress.progress((idx + 1) / len(tickers_pt))

            summary = summarize_strategy_backtest(records_by_ticker, horizon=pb_tr_horizon)
            st.write(summary)
            if summary["Segnali totali"] == 0:
                st.warning("Nessun segnale trovato in questo universo/periodo.")

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
