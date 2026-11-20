# app.py — Market Scanner Pro: pagina principale "Cosa fare oggi".
#
# Avvio:  streamlit run app.py
# Altre pagine (menu a sinistra): Strategie (risultati dei test),
# Grafico (segnali sul grafico di un titolo), Strumenti avanzati (la
# vecchia app completa).

import json
import os

import pandas as pd
import streamlit as st

from core import ALERT_STRATEGY_CONFIG
from signals_today import all_signals, ACTION_ORDER
from strategies_pro import STRATEGY_INFO

st.set_page_config(page_title="Market Scanner Pro", page_icon="📈", layout="wide")

RESULTS = os.path.join(os.path.dirname(__file__), "results", "pro_results.json")


@st.cache_data(ttl=3600, show_spinner=False)
def load_results():
    try:
        with open(RESULTS) as fh:
            return json.load(fh)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_signals():
    s = all_signals()
    s.pop("data", None)  # i prezzi servono solo alla pagina Grafico
    return s


VERDICT_BADGE = {"approvata": "🟢 Approvata", "incerta": "🟡 Incerta", "bocciata": "🔴 Bocciata",
                 "insufficiente": "⚪ Dati insufficienti"}
ACTION_ICON = {"COMPRA": "🟢 COMPRA", "VENDI": "🔴 VENDI", "TIENI": "⏳ TIENI", "LIQUIDITÀ": "💤 LIQUIDITÀ"}

results = load_results()
active = [k for k in STRATEGY_INFO if ALERT_STRATEGY_CONFIG.get(k, False)]

st.title("📅 Cosa fare oggi")

with st.sidebar:
    st.header("💶 Il tuo capitale")
    st.caption("Quanto vuoi dedicare a ogni strategia. Serve solo a calcolare gli importi qui a destra: "
               "non viene salvato da nessuna parte.")
    capital = {}
    for k in active:
        capital[k] = st.number_input(STRATEGY_INFO[k]["name"], min_value=0, value=10000, step=1000, key=f"cap_{k}")
    if st.button("🔄 Aggiorna i dati"):
        load_signals.clear()

with st.spinner("Scarico i prezzi e calcolo i segnali (circa un minuto la prima volta)..."):
    sig = load_signals()

rows = pd.DataFrame(sig["rows"])
last_date = sig["last_date"]
st.markdown(
    f"Segnali calcolati sulla chiusura del **{last_date:%d/%m/%Y}**. "
    "Le operazioni **COMPRA** e **VENDI** vanno fatte **il giorno di borsa successivo, all'apertura**."
)

if not active:
    st.warning("Nessuna strategia è attiva. Le strategie si attivano in `ALERT_STRATEGY_CONFIG` in core.py.")

# Quante "quote" usa ogni strategia (stesse ipotesi del backtest)
n_slots = {
    "mean_reversion_rsi2": max(1, len(sig["universe"]) - len(sig["missing"])),
    "trend_10_mesi": max(1, sum(1 for t, u in sig["universe"].items()
                                if u["category"] in ("Indici Americani", "Indici Europei") and t not in sig["missing"])),
    "momentum_12_1": 5,
}

# riepilogo in alto
cols = st.columns(max(1, len(active)))
for col, k in zip(cols, active):
    sub = rows[rows["strategy"] == k] if not rows.empty else rows
    n_buy = int((sub["action"] == "COMPRA").sum()) if not sub.empty else 0
    n_sell = int((sub["action"] == "VENDI").sum()) if not sub.empty else 0
    with col:
        st.metric(STRATEGY_INFO[k]["name"], f"{n_buy} compra · {n_sell} vendi")

for k in active:
    info = STRATEGY_INFO[k]
    res = (results or {}).get("strategies", {}).get(k, {})
    st.divider()
    head = f"### {info['name']}"
    if res:
        head += f"  \n{VERDICT_BADGE.get(res.get('verdict'), '')}"
    st.markdown(head)
    if res:
        f, b = res["strategy"]["full"], res["bench"]["full"]
        st.caption(
            f"Test su {f.get('years', 0):.0f} anni: {f.get('cagr_pct', 0):+.1f}% l'anno, calo massimo {f.get('max_dd_pct', 0):.0f}%. "
            f"Compra e tieni degli stessi titoli: {b.get('cagr_pct', 0):+.1f}% l'anno, calo massimo {b.get('max_dd_pct', 0):.0f}%. "
            "Dettagli nella pagina Strategie."
        )
    with st.expander("Regole della strategia"):
        for r in info["rules"]:
            st.markdown(f"- {r}")
        st.caption(f"Fonte: {info['source']}. Universo: {info['universe']}.")

    sub = rows[rows["strategy"] == k].copy() if not rows.empty else pd.DataFrame()
    per_slot = capital.get(k, 0) / n_slots[k]

    if k == "momentum_12_1":
        m = sig["momentum"]
        if m:
            st.markdown(("✅ " if m["regime_ok"] else "⛔ ") + m["regime_text"] +
                        (": a fine mese si investe." if m["regime_ok"] else ": a fine mese si resta in liquidità."))
    if k == "trend_10_mesi":
        st.caption("Gli indici non si comprano direttamente: si usa un ETF che li replica (per esempio un ETF sull'S&P 500 o sul DAX).")

    if sub.empty:
        st.info("Nessuna operazione e nessuna posizione aperta.")
        continue

    sub["ord"] = sub["action"].map(ACTION_ORDER)
    sub = sub.sort_values(["ord", "ticker"])
    todo = sub[sub["action"].isin(["COMPRA", "VENDI"])]
    if todo.empty:
        st.success("Nessuna operazione da fare domani.")
    else:
        for _, r in todo.iterrows():
            amount = f" — circa **{per_slot:,.0f} €**".replace(",", ".") if r["action"] == "COMPRA" and per_slot else ""
            extra = ""
            if r["action"] == "VENDI" and pd.notna(r.get("open_ret_pct")):
                extra = f" (in posizione dal {r['since']:%d/%m}, {r['open_ret_pct']:+.1f}%)"
            st.markdown(f"**{ACTION_ICON[r['action']]} {r['ticker']}** {r['name'] if r['name'] != r['ticker'] else ''}"
                        f"{amount}{extra}  \n<small>{r['why']}</small>", unsafe_allow_html=True)

    hold = sub[sub["action"].isin(["TIENI", "LIQUIDITÀ"])]
    if not hold.empty:
        table = pd.DataFrame({
            "Stato": hold["action"].map(ACTION_ICON),
            "Titolo": hold["ticker"],
            "Nome": hold["name"],
            "Dal": hold["since"].map(lambda d: d.strftime("%d/%m/%Y") if pd.notna(d) and d else ""),
            "Risultato": hold["open_ret_pct"].map(lambda v: f"{v:+.1f}%" if pd.notna(v) else ""),
            "Note": hold["why"],
        })
        st.markdown("**Posizioni in corso**" if k != "trend_10_mesi" else "**Stato degli indici**")
        st.dataframe(table, hide_index=True, width="stretch")

    if k == "momentum_12_1" and sig["momentum"].get("ranking"):
        with st.expander("Classifica momentum con i prezzi di oggi (anteprima della prossima scelta)"):
            rk = pd.DataFrame(sig["momentum"]["ranking"])
            rk.index = range(1, len(rk) + 1)
            rk["mom_pct"] = rk["mom_pct"].map(lambda v: f"{v:+.1f}%")
            st.dataframe(rk.rename(columns={"ticker": "Titolo", "name": "Nome", "mom_pct": "Rendimento 12-1 mesi"}),
                         width="stretch")

disabled = [k for k in STRATEGY_INFO if not ALERT_STRATEGY_CONFIG.get(k, False)]
if disabled:
    st.divider()
    st.caption("Strategie testate ma non attive: " + ", ".join(STRATEGY_INFO[k]["name"] for k in disabled) +
               ". Vedi la pagina Strategie per il perché.")

if sig["missing"]:
    st.caption("Titoli senza dati oggi (esclusi): " + ", ".join(sig["missing"]))

st.caption("⚠️ Strumento informativo. I risultati passati non garantiscono quelli futuri: "
           "non è un consiglio di investimento.")
