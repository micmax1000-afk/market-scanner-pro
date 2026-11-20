# Pagina "Strategie": risultati dei test su 20 anni (results/pro_results.json).

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import ALERT_STRATEGY_CONFIG

st.set_page_config(page_title="Strategie", page_icon="📊", layout="wide")
ROOT = os.path.dirname(os.path.dirname(__file__))


@st.cache_data(ttl=3600)
def load(name):
    try:
        with open(os.path.join(ROOT, "results", name)) as fh:
            return json.load(fh)
    except Exception:
        return None


R = load("pro_results.json")
st.title("📊 Strategie: come sono andate nei test")
if not R:
    st.error("Risultati non trovati: lancia il workflow 'Backtest realistico' su GitHub.")
    st.stop()

st.markdown(
    f"Test su **{R['period']}** di dati giornalieri, aggiornato il {R['generated_at'][:10]}. "
    f"Ingresso e uscita all'apertura del giorno dopo il segnale, costi {R['cost_per_side'] * 100:.2f}% per lato, "
    "prezzi rettificati per i dividendi. Ogni strategia è confrontata con il **compra e tieni** degli stessi titoli. "
    f"Lo storico è diviso in due periodi (prima e dopo il {R['split'][:4]}) per vedere se il risultato regge nel tempo."
)

S = R["strategies"]
VERDICT = {"approvata": "🟢 Approvata", "incerta": "🟡 Incerta", "bocciata": "🔴 Bocciata", "insufficiente": "⚪ Dati insufficienti"}


def pct(v, sign=True):
    return "—" if v is None else (f"{v:+.1f}%" if sign else f"{v:.1f}%")


def num(v):
    return "—" if v is None else f"{v:.2f}"


summary = []
for k, s in S.items():
    f, b = s["strategy"]["full"], s["bench"]["full"]
    summary.append({
        "Strategia": s["name"], "Giudizio": VERDICT.get(s["verdict"], s["verdict"]),
        "Alert": "attivo" if ALERT_STRATEGY_CONFIG.get(k) else "spento",
        "Rendimento annuo": pct(f.get("cagr_pct")), "Calo massimo": pct(f.get("max_dd_pct")),
        "Sharpe": num(f.get("sharpe")),
        "Compra e tieni: annuo": pct(b.get("cagr_pct")), "Compra e tieni: calo max": pct(b.get("max_dd_pct")),
        "Compra e tieni: Sharpe": num(b.get("sharpe")),
    })
st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
st.caption("Sharpe = rendimento diviso per il rischio (volatilità). Più alto è meglio; sopra 0,5 è buono per una strategia su azioni. "
           "Calo massimo = la perdita peggiore da un massimo al minimo successivo.")

key = st.selectbox("Strategia da vedere nel dettaglio", list(S), format_func=lambda k: S[k]["name"])
s = S[key]
st.header(s["name"])
st.markdown(f"**{VERDICT.get(s['verdict'])}** — {s['verdict_text']}")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Regole**")
    for r in s["rules"]:
        st.markdown(f"- {r}")
    st.caption(f"Fonte: {s['source']}. Universo: {s['universe']}.")
with c2:
    st.markdown("**Rendimento e rischio**")
    tbl = []
    for label, part in (("Tutto il periodo", "full"), (f"Fino al {int(R['split'][:4]) - 1}", "first"), (f"Dal {R['split'][:4]}", "second")):
        a, b = s["strategy"][part], s["bench"][part]
        tbl.append({"Periodo": f"{label} ({a.get('start', '')[:4]}–{a.get('end', '')[:4]})",
                    "Strategia annuo": pct(a.get("cagr_pct")), "Strategia calo max": pct(a.get("max_dd_pct")),
                    "Strategia Sharpe": num(a.get("sharpe")),
                    "Compra e tieni annuo": pct(b.get("cagr_pct")), "C&T calo max": pct(b.get("max_dd_pct")),
                    "C&T Sharpe": num(b.get("sharpe"))})
    st.dataframe(pd.DataFrame(tbl), hide_index=True, width="stretch")
    if s["strategy"]["full"].get("exposure_pct") is not None:
        st.caption(f"Tempo investito in media: {s['strategy']['full']['exposure_pct']:.0f}% del capitale. Il resto è in liquidità (nel test non rende nulla).")
    if s.get("note"):
        st.caption(s["note"])

eq = pd.DataFrame(s["equity"]).assign(serie="Strategia")
be = pd.DataFrame(s["bench_equity"]).assign(serie="Compra e tieni")
fig = go.Figure()
fig.add_trace(go.Scatter(x=be["date"], y=be["value"] * 10000, name="Compra e tieni", line=dict(color="#9aa0a6")))
fig.add_trace(go.Scatter(x=eq["date"], y=eq["value"] * 10000, name="Strategia", line=dict(width=3)))
fig.update_layout(title="Valore di 10.000 € investiti (scala logaritmica)", yaxis_type="log", height=420,
                  margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.02, x=0))
st.plotly_chart(fig, width="stretch")

yr = pd.DataFrame(s["by_year"])
if not yr.empty:
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=yr["year"], y=yr["bench_pct"], name="Compra e tieni", marker_color="#c7cbd1"))
    fig2.add_trace(go.Bar(x=yr["year"], y=yr["strategy_pct"], name="Strategia",
                          marker_color=["#1a7f37" if (v or 0) >= 0 else "#c4302b" for v in yr["strategy_pct"]]))
    fig2.update_layout(title="Rendimento per anno (%)", barmode="group", height=360,
                       margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig2, width="stretch")

t = s["trades"]
if t.get("n"):
    st.subheader("Operazioni")
    m = st.columns(5)
    m[0].metric("Operazioni", f"{t['n']:,}".replace(",", "."))
    m[1].metric("Vincenti", pct(t.get("win_rate"), sign=False))
    m[2].metric("Guadagno medio", pct(t.get("avg_pct")))
    m[3].metric("Giorni medi", f"{t.get('avg_days', 0):.0f}")
    m[4].metric("Operazione peggiore", pct(t.get("worst_pct")))
    with st.expander("Risultato per titolo"):
        st.dataframe(pd.DataFrame(s["per_ticker"]), hide_index=True, width="stretch")
    with st.expander("Ultime operazioni simulate"):
        st.dataframe(pd.DataFrame(s["recent_trades"]), hide_index=True, width="stretch")

old = load("backtest_results.json")
if old:
    st.divider()
    with st.expander("Strategie eliminate (test del 19/11/2026)"):
        st.markdown("Queste strategie sono state tolte dall'app: con ingresso realistico e costi perdevano soldi, "
                    "oppure il vantaggio è sparito negli anni recenti.")
        rows = []
        for k2, s2 in old["strategies"].items():
            if k2 == "baseline":
                continue
            f2, o2 = s2["full"], s2["out_of_sample"]
            rows.append({"Strategia": s2["name"], "Operazioni": f2.get("n"),
                         "R medio per operazione": num(f2.get("avg_r")), "R medio dal 2022": num(o2.get("avg_r")),
                         "Giudizio": s2["verdict"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Nel test del 19/11 c'erano anche pullback ipervenduto, pullback EMA20, Turtle, candlestick, triangolo e "
                   "bull flag: tutte con R medio negativo o nullo dopo i costi.")

st.caption("⚠️ I risultati passati non garantiscono quelli futuri. Non è un consiglio di investimento.")
