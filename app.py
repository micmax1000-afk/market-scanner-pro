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
            score_v4 = compute_score_v4(data)

            col_v3, col_v4 = st.columns(2)
            with col_v3:
                st.metric("Score V3 (attuale)", f"{score}/100")
            with col_v4:
                delta = round(score_v4 - score, 1)
                st.metric("Score V4 (sperimentale)", f"{score_v4}/100", delta=delta)
            st.caption(
                "Lo Score V4 consolida 4 componenti dello V3 che misuravano tutte, da "
                "angolazioni diverse, la stessa 'forza del trend' (allineamento EMA, "
                "pendenza, rottura trend line, ADX) in una sola misura, e ridistribuisce "
                "il peso liberato su volume/breakout/Bollinger. Se i due punteggi divergono "
                "molto, probabilmente il titolo deve gran parte del suo Score V3 alla "
                "ripetizione dello stesso segnale di trend, non a conferme indipendenti."
            )

            ref_index = get_reference_index(ticker)
            if ref_index is not None:
                market_regime = compute_market_regime(ref_index)
                if market_regime == "Rialzista":
                    st.caption(f"🌍 Regime di mercato ({ref_index}): 🟢 Rialzista — sopra la media mobile a 200gg")
                elif market_regime == "Ribassista":
                    st.caption(f"🌍 Regime di mercato ({ref_index}): 🔴 Ribassista — sotto la media mobile a 200gg. Informativo: comprare rialzi controcorrente è statisticamente meno affidabile, ma non è un blocco automatico.")
                else:
                    st.caption(f"🌍 Regime di mercato ({ref_index}): n/d (dati insufficienti)")

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

            st.subheader("🐢 Strategia: Turtle Trading (Donchian Breakout)")
            st.caption("⚠️ Non ancora collegata agli alert: testala con il Backtest SL/TP prima.")
            turtle = compute_strategy_turtle_breakout(data)
            if turtle["signal"] == "ACQUISTO":
                st.success("📈 ACQUISTO — rottura del canale di Donchian")
            else:
                st.info("⏸️ Nessuna rottura")
            for desc, met in turtle["conditions"]:
                st.write(f"{'✅' if met else '❌'} {desc}")

            st.subheader("🕯️ Strategia: Pattern di inversione (Candlestick) su supporto")
            st.caption("⚠️ Non ancora collegata agli alert: testala con il Backtest SL/TP prima.")
            candle = compute_strategy_candlestick_reversal(data)
            if candle["signal"] == "ACQUISTO":
                st.success("📈 ACQUISTO — pattern di inversione sul supporto")
            else:
                st.info("⏸️ Condizioni non tutte soddisfatte")
            for desc, met in candle["conditions"]:
                st.write(f"{'✅' if met else '❌'} {desc}")

            st.subheader("🔺 Strategia: Triangolo simmetrico con breakout")
            st.caption("⚠️ Non ancora collegata agli alert: testala con il Backtest SL/TP prima.")
            triangle = compute_strategy_triangle_breakout(data)
            if triangle["signal"] == "ACQUISTO":
                st.success("📈 ACQUISTO — rottura del triangolo")
            else:
                st.info("⏸️ Condizioni non tutte soddisfatte")
            for desc, met in triangle["conditions"]:
                st.write(f"{'✅' if met else '❌'} {desc}")

            st.subheader("🚩 Strategia: Bandiera rialzista (Bull Flag)")
            st.caption("⚠️ Non ancora collegata agli alert: testala con il Backtest SL/TP prima.")
            flag = compute_strategy_bull_flag(data)
            if flag["signal"] == "ACQUISTO":
                st.success("📈 ACQUISTO — rottura della bandiera")
            else:
                st.info("⏸️ Condizioni non tutte soddisfatte")
            for desc, met in flag["conditions"]:
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

    with st.expander("⚙️ Sensibilità dei filtri (di default già allentati rispetto all'originale)"):
        st.caption(
            "Con tutti e 7 i filtri obbligatori insieme, lo scanner può dare spesso tabella "
            "vuota su un universo piccolo: qui puoi allentare o restringere le tre soglie più "
            "severe. Valori più alti/permissivi = più titoli passano; valori originali stretti "
            "(volatilità 4%, volume 1.2x, breakout 0.5x ATR) = meno titoli ma più selettivi."
        )
        max_volatility_pct = st.slider(
            "Volatilità massima 20gg consentita (%)", min_value=2, max_value=15, value=8, step=1,
            key="scan_max_volatility"
        )
        min_volume_ratio = st.slider(
            "Volume minimo richiesto (x media 20gg)", min_value=0.8, max_value=2.0, value=1.0, step=0.1,
            key="scan_min_volume_ratio"
        )
        min_breakout_ratio = st.slider(
            "Breakout minimo richiesto (x ATR sopra il massimo 60gg)", min_value=0.0, max_value=1.5, value=0.2, step=0.1,
            key="scan_min_breakout_ratio"
        )

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
                        timeframe_label=tf_label,
                        max_volatility=max_volatility_pct / 100,
                        min_volume_ratio=min_volume_ratio,
                        min_breakout_ratio=min_breakout_ratio
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

    st.markdown("---")
    st.subheader("🏆 Momentum Ranking (fattore stile fondi sistematici)")
    st.caption(
        "A differenza dello Scanner sopra (segnale sì/no sul singolo titolo), qui classifica "
        "TUTTO l'universo scelto per rendimento storico e mostra i migliori — è il fattore "
        "'momentum' usato su vasta scala da fondi sistematici (senza il position sizing e la "
        "gestione del rischio che loro applicano in più: qui è solo il ranking di base)."
    )

    momentum_universe = st.multiselect(
        "Universo da classificare",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="momentum_universe"
    )
    momentum_lookback = st.selectbox(
        "Periodo di lookback",
        [("1 mese (~21gg)", 21), ("3 mesi (~63gg)", 63), ("6 mesi (~126gg)", 126), ("12 mesi (~252gg)", 252)],
        format_func=lambda x: x[0],
        index=2,
        key="momentum_lookback"
    )

    if st.button("Calcola Momentum Ranking"):
        momentum_tickers = list(dict.fromkeys(t for cat in momentum_universe for t in TICKER_CATALOG[cat].values()))
        if not momentum_tickers:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            with st.spinner(f"Calcolo ranking su {len(momentum_tickers)} titoli..."):
                df_momentum = compute_momentum_ranking(momentum_tickers, lookback_days=momentum_lookback[1])
            if df_momentum.empty:
                st.warning("Dati insufficienti per calcolare il ranking su questo universo/periodo.")
            else:
                st.dataframe(df_momentum, use_container_width=True, hide_index=True)
                st.caption(
                    "I titoli in cima hanno avuto il rendimento migliore nel periodo scelto. "
                    "Il fattore momentum scommette che chi ha performato meglio di recente "
                    "tenda a continuare nel breve-medio termine — non è garantito, e funziona "
                    "in media su portafogli diversificati, non sul singolo titolo isolato."
                )

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

    st.subheader("🎯 Backtest realistico con Stop Loss / Take Profit")
    st.caption(
        "A differenza dei backtest sopra (che guardano solo se il prezzo è più alto "
        "dopo N giorni fissi), questo simula un trade vero: entra al segnale, esce in "
        "perdita controllata se tocca lo stop, in guadagno se tocca il target, o "
        "altrimenti alla scadenza. Stop e target sono multipli dell'ATR (volatilità), "
        "così si adattano automaticamente a ogni titolo."
    )

    sl_tp_strategy_choice = st.selectbox(
        "Strategia da testare",
        ["Segnale combinato (rottura + conferme)", "Rottura resistenza + momentum basso (la tua)", "Pullback trend EMA20 (proposta)", "Turtle Trading (Donchian Breakout)", "Pattern candlestick su supporto", "Triangolo simmetrico con breakout", "Bandiera rialzista (Bull Flag)"],
        key="sl_tp_strategy_choice"
    )
    sl_tp_universe = st.multiselect(
        "Universo per il backtest",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="sl_tp_universe"
    )
    col_sl, col_tp, col_hold = st.columns(3)
    with col_sl:
        sl_mult = st.number_input("Stop Loss (x ATR)", min_value=0.5, max_value=5.0, value=1.5, step=0.25, key="sl_mult")
    with col_tp:
        tp_mult = st.number_input("Take Profit (x ATR)", min_value=0.5, max_value=10.0, value=3.0, step=0.25, key="tp_mult")
    with col_hold:
        max_hold = st.number_input("Giorni massimi in trade", min_value=5, max_value=60, value=20, step=5, key="max_hold")

    if st.button("Esegui Backtest Stop Loss / Take Profit"):
        tickers_sltp = list(dict.fromkeys(t for cat in sl_tp_universe for t in TICKER_CATALOG[cat].values()))
        if not tickers_sltp:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            if sl_tp_strategy_choice == "Segnale combinato (rottura + conferme)":
                strategy_fn = lambda d: compute_trade_signal(d, min_confirmations=3)
            elif sl_tp_strategy_choice == "Rottura resistenza + momentum basso (la tua)":
                strategy_fn = compute_strategy_pullback_oversold
            elif sl_tp_strategy_choice == "Pullback trend EMA20 (proposta)":
                strategy_fn = compute_strategy_trend_pullback
            elif sl_tp_strategy_choice == "Turtle Trading (Donchian Breakout)":
                strategy_fn = compute_strategy_turtle_breakout
            elif sl_tp_strategy_choice == "Pattern candlestick su supporto":
                strategy_fn = compute_strategy_candlestick_reversal
            elif sl_tp_strategy_choice == "Triangolo simmetrico con breakout":
                strategy_fn = compute_strategy_triangle_breakout
            else:
                strategy_fn = compute_strategy_bull_flag

            trades_by_ticker = {}
            progress = st.progress(0.0)
            for idx, tkr in enumerate(tickers_sltp):
                try:
                    hist = download_data(tkr, period="2y", interval="1d")
                    if not hist.empty and len(hist) >= 250:
                        trades_by_ticker[tkr] = backtest_strategy_sl_tp(
                            hist, strategy_fn,
                            atr_sl_mult=sl_mult, atr_tp_mult=tp_mult, max_holding_days=int(max_hold)
                        )
                except Exception as e:
                    print(f"Errore backtest SL/TP su {tkr}: {e}")
                progress.progress((idx + 1) / len(tickers_sltp))

            summary = summarize_sl_tp_backtest(trades_by_ticker)
            st.write(summary)

            if summary["Trade totali"] == 0:
                st.warning("Nessun segnale trovato in questo universo/periodo.")
            else:
                r_medio = summary["R medio"]
                pf = summary["Profit Factor"]
                if r_medio is not None:
                    if r_medio > 0 and (pf is None or pf > 1):
                        st.success(
                            f"R medio positivo ({r_medio}) e Profit Factor "
                            f"{'n/d' if pf is None else pf}: nel periodo testato il sistema "
                            f"avrebbe guadagnato più di quanto perso, considerando SL/TP."
                        )
                    else:
                        st.warning(
                            f"R medio {r_medio}, Profit Factor {'n/d' if pf is None else pf}: "
                            f"con questi SL/TP il sistema non risulta profittevole nel periodo testato."
                        )
                st.caption(
                    "'R medio' = guadagno/perdita medio per trade espresso in multipli del "
                    "rischio (1R = distanza tra ingresso e stop loss). Un R medio positivo con "
                    "Profit Factor > 1 significa che i guadagni superano le perdite in totale, "
                    "non solo che vinci più spesso di quanto perdi. 'Drawdown massimo (R)' e "
                    "'Perdite consecutive max' dicono invece quanto puoi 'soffrire' nel mezzo, "
                    "anche in un sistema profittevole in media."
                )

    st.markdown("---")
    st.subheader("⚖️ Confronta tutte e tre le strategie (stessi parametri SL/TP)")
    st.caption(
        "Testa il segnale combinato e le due strategie di solo acquisto con gli stessi "
        "Stop Loss/Take Profit/universo, scaricando i dati di ogni titolo una sola volta "
        "(non tre), e le mette in ordine dalla più alla meno profittevole per Profit Factor."
    )

    compare_universe = st.multiselect(
        "Universo per il confronto",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="compare_universe"
    )
    col_csl, col_ctp, col_chold = st.columns(3)
    with col_csl:
        compare_sl = st.number_input("Stop Loss (x ATR)", min_value=0.5, max_value=5.0, value=1.5, step=0.25, key="compare_sl")
    with col_ctp:
        compare_tp = st.number_input("Take Profit (x ATR)", min_value=0.5, max_value=10.0, value=3.0, step=0.25, key="compare_tp")
    with col_chold:
        compare_hold = st.number_input("Giorni massimi in trade", min_value=5, max_value=60, value=20, step=5, key="compare_hold")

    if st.button("Confronta le 3 strategie"):
        tickers_cmp = list(dict.fromkeys(t for cat in compare_universe for t in TICKER_CATALOG[cat].values()))
        if not tickers_cmp:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            strategies = {
                "Segnale combinato (rottura + conferme)": lambda d: compute_trade_signal(d, min_confirmations=3),
                "Rottura resistenza + momentum basso (la tua)": compute_strategy_pullback_oversold,
                "Pullback trend EMA20 (proposta)": compute_strategy_trend_pullback,
                "Turtle Trading (Donchian Breakout)": compute_strategy_turtle_breakout,
                "Pattern candlestick su supporto": compute_strategy_candlestick_reversal,
                "Triangolo simmetrico con breakout": compute_strategy_triangle_breakout,
                "Bandiera rialzista (Bull Flag)": compute_strategy_bull_flag,
            }
            trades_by_strategy = {name: {} for name in strategies}

            progress = st.progress(0.0)
            for idx, tkr in enumerate(tickers_cmp):
                try:
                    hist = download_data(tkr, period="2y", interval="1d")
                    if not hist.empty and len(hist) >= 250:
                        for name, fn in strategies.items():
                            trades_by_strategy[name][tkr] = backtest_strategy_sl_tp(
                                hist, fn,
                                atr_sl_mult=compare_sl, atr_tp_mult=compare_tp, max_holding_days=int(compare_hold)
                            )
                except Exception as e:
                    print(f"Errore confronto strategie su {tkr}: {e}")
                progress.progress((idx + 1) / len(tickers_cmp))

            rows = []
            for name in strategies:
                s = summarize_sl_tp_backtest(trades_by_strategy[name])
                s["Strategia"] = name
                rows.append(s)

            comparison_df = pd.DataFrame(rows)[
                ["Strategia", "Trade totali", "Take Profit %", "Stop Loss %", "Timeout %",
                 "R medio", "Profit Factor", "Drawdown massimo (R)", "Perdite consecutive max",
                 "Giorni medi in trade"]
            ]
            comparison_df = comparison_df.sort_values(
                "Profit Factor", ascending=False, na_position="last"
            )
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.caption(
                "'Drawdown massimo (R)' = la peggiore serie di perdite non ancora recuperate, "
                "mettendo in fila tutti i trade per data (approssimazione: se più titoli segnalano "
                "lo stesso giorno, in realtà li apriresti insieme, non in sequenza). "
                "'Perdite consecutive max' = quante sconfitte di fila ha attraversato il sistema "
                "nel periodo testato: utile per capire cosa aspettarsi psicologicamente, anche in "
                "un sistema che alla fine è profittevole."
            )

            valid_rows = comparison_df.dropna(subset=["Profit Factor"])
            if not valid_rows.empty:
                best = valid_rows.iloc[0]
                st.info(
                    f"In questo universo e periodo, **{best['Strategia']}** ha il Profit Factor "
                    f"più alto ({best['Profit Factor']}) su {int(best['Trade totali'])} trade. "
                    f"Con un campione piccolo questo può cambiare cambiando universo o parametri: "
                    f"prova a ripetere il confronto su un altro sottoinsieme di titoli prima di "
                    f"fidartene."
                )
            else:
                st.warning("Nessuna strategia ha generato trade sufficienti in questo universo/periodo.")

    st.markdown("---")
    st.subheader("🔀 Backtest con Stop Loss dinamico (Supertrend)")
    st.caption(
        "Stessa logica di ingresso/take profit del backtest SL/TP sopra, ma lo stop non è "
        "più fisso: segue il Supertrend, che si stringe verso l'alto mano a mano che il trend "
        "si sviluppa (esce anche se il Supertrend 'flippa' da rialzista a ribassista). L'idea "
        "è lasciare correre i guadagni invece di uscire sempre alla stessa distanza fissa "
        "dall'ingresso. Confronta il Profit Factor con quello del backtest a stop fisso sopra."
    )

    st_strategy_choice = st.selectbox(
        "Strategia da testare",
        ["Segnale combinato (rottura + conferme)", "Rottura resistenza + momentum basso (la tua)", "Pullback trend EMA20 (proposta)", "Turtle Trading (Donchian Breakout)", "Pattern candlestick su supporto", "Triangolo simmetrico con breakout", "Bandiera rialzista (Bull Flag)"],
        key="st_strategy_choice"
    )
    st_universe = st.multiselect(
        "Universo per il backtest",
        ["Azioni Italiane", "Indici Americani", "Indici Europei"],
        default=["Azioni Italiane"],
        key="st_universe"
    )
    col_sttp, col_stperiod, col_stmult, col_sthold = st.columns(4)
    with col_sttp:
        st_tp_mult = st.number_input("Take Profit (x ATR)", min_value=0.5, max_value=10.0, value=3.0, step=0.25, key="st_tp_mult")
    with col_stperiod:
        st_period = st.number_input("Periodo Supertrend", min_value=5, max_value=30, value=10, step=1, key="st_period")
    with col_stmult:
        st_mult = st.number_input("Moltiplicatore Supertrend (x ATR)", min_value=1.0, max_value=6.0, value=3.0, step=0.5, key="st_mult")
    with col_sthold:
        st_max_hold = st.number_input("Giorni massimi in trade", min_value=5, max_value=60, value=20, step=5, key="st_max_hold")

    if st.button("Esegui Backtest Supertrend"):
        tickers_st = list(dict.fromkeys(t for cat in st_universe for t in TICKER_CATALOG[cat].values()))
        if not tickers_st:
            st.warning("Seleziona almeno un universo di titoli.")
        else:
            if st_strategy_choice == "Segnale combinato (rottura + conferme)":
                st_strategy_fn = lambda d: compute_trade_signal(d, min_confirmations=3)
            elif st_strategy_choice == "Rottura resistenza + momentum basso (la tua)":
                st_strategy_fn = compute_strategy_pullback_oversold
            elif st_strategy_choice == "Pullback trend EMA20 (proposta)":
                st_strategy_fn = compute_strategy_trend_pullback
            elif st_strategy_choice == "Turtle Trading (Donchian Breakout)":
                st_strategy_fn = compute_strategy_turtle_breakout
            elif st_strategy_choice == "Pattern candlestick su supporto":
                st_strategy_fn = compute_strategy_candlestick_reversal
            elif st_strategy_choice == "Triangolo simmetrico con breakout":
                st_strategy_fn = compute_strategy_triangle_breakout
            else:
                st_strategy_fn = compute_strategy_bull_flag

            trades_by_ticker_st = {}
            progress = st.progress(0.0)
            for idx, tkr in enumerate(tickers_st):
                try:
                    hist = download_data(tkr, period="2y", interval="1d")
                    if not hist.empty and len(hist) >= 250:
                        trades_by_ticker_st[tkr] = backtest_strategy_sl_tp_supertrend(
                            hist, st_strategy_fn,
                            atr_tp_mult=st_tp_mult, supertrend_period=int(st_period),
                            supertrend_multiplier=st_mult, max_holding_days=int(st_max_hold)
                        )
                except Exception as e:
                    print(f"Errore backtest Supertrend su {tkr}: {e}")
                progress.progress((idx + 1) / len(tickers_st))

            summary_st = summarize_sl_tp_backtest(trades_by_ticker_st)
            st.write(summary_st)

            if summary_st["Trade totali"] == 0:
                st.warning("Nessun segnale trovato in questo universo/periodo.")
            else:
                st.caption(
                    "Confronta 'Profit Factor' e 'Drawdown massimo (R)' con lo stesso "
                    "backtest a stop fisso qui sopra sulla stessa strategia/universo: se lo "
                    "stop dinamico fa meglio su entrambi, è un miglioramento robusto; se "
                    "migliora uno e peggiora l'altro, è un trade-off da valutare."
                )

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
