"""
run_pro_backtest.py — Testa le strategie di strategies_pro.py su 20 anni
di dati e le confronta col "compra e tieni" degli stessi titoli.
Salva results/pro_results.json. Gira su GitHub Actions
(.github/workflows/backtest.yml).

    python run_pro_backtest.py --period 20y
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core import TICKER_CATALOG, download_data
from strategies_pro import (
    STRATEGY_INFO, COST_PER_SIDE, adjust_ohlc, target_rsi2, target_trend10, targets_momentum,
    monthly_sma_above, sleeve_returns, extract_trades, perf_metrics, trade_metrics, verdict_pro,
    today_rsi2, today_trend10,
)

SPLIT = pd.Timestamp("2016-01-01")
REGIME_TICKER = "FTSEMIB.MI"


def fetch(ticker, period):
    for attempt in range(4):
        try:
            d = download_data(ticker, period=period, interval="1d")
            if d is not None and not d.empty:
                return adjust_ohlc(d)
        except Exception as e:
            print(f"  {ticker}: tentativo {attempt + 1} fallito ({e})", flush=True)
        time.sleep(4 * (attempt + 1))
    return None


def split_metrics(daily, exposure=None):
    return (perf_metrics(daily, exposure),
            perf_metrics(daily[daily.index < SPLIT], exposure),
            perf_metrics(daily[daily.index >= SPLIT], exposure))


def monthly_equity(daily):
    eq = (1 + daily.fillna(0)).cumprod()
    m = eq.resample("ME").last()
    return [{"date": str(d.date()), "value": round(float(v), 4)} for d, v in m.items()]


def yearly(daily):
    y = (1 + daily.fillna(0)).groupby(daily.index.year).prod() - 1
    return {int(k): round(float(v) * 100, 2) for k, v in y.items()}


def portfolio_mean(sleeves):
    """Pesi uguali tra i titoli disponibili quel giorno (prima della
    quotazione un titolo non conta)."""
    df = pd.DataFrame(sleeves)
    return df.mean(axis=1, skipna=True)


def package(key, strat_daily, bench_daily, exposure, trades, per_ticker, today, bench_label, extra=None):
    full, first, second = split_metrics(strat_daily, exposure)
    bfull, bfirst, bsecond = split_metrics(bench_daily)
    v, vt = verdict_pro(full, first, second, bfull)
    ys, yb = yearly(strat_daily), yearly(bench_daily)
    recent = sorted(trades, key=lambda t: t["entry"], reverse=True)[:60]
    out = {
        "key": key, **STRATEGY_INFO[key],
        "verdict": v, "verdict_text": vt,
        "benchmark": bench_label,
        "strategy": {"full": full, "first": first, "second": second},
        "bench": {"full": bfull, "first": bfirst, "second": bsecond},
        "trades": trade_metrics(trades),
        "trades_first": trade_metrics([t for t in trades if t["entry"] < SPLIT]),
        "trades_second": trade_metrics([t for t in trades if t["entry"] >= SPLIT]),
        "equity": monthly_equity(strat_daily),
        "bench_equity": monthly_equity(bench_daily),
        "by_year": [{"year": y, "strategy_pct": ys.get(y), "bench_pct": yb.get(y)} for y in sorted(set(ys) | set(yb))],
        "per_ticker": per_ticker,
        "recent_trades": [{**t, "entry": str(t["entry"].date()), "exit": str(t["exit"].date()),
                           "ret_pct": round(t["ret_pct"], 2)} for t in recent],
        "today": today,
    }
    if extra:
        out.update(extra)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="20y")
    ap.add_argument("--out", default="results/pro_results.json")
    args = ap.parse_args()
    t0 = time.time()

    universe = {}
    for cat, items in TICKER_CATALOG.items():
        for name, t in items.items():
            universe[t] = {"name": name, "category": cat}

    data, skipped = {}, []
    for t in universe:
        d = fetch(t, args.period)
        if d is None or len(d) < 260:
            skipped.append(t)
            print(f"[skip] {t}", flush=True)
        else:
            data[t] = d
            print(f"[ok]   {t}: {d.index[0].date()} -> {d.index[-1].date()} ({len(d)} giorni)", flush=True)

    results = {}

    # ---------------- 1. Mean reversion RSI(2) su tutto -------------------
    sleeves, bench, held, trades, per_ticker, today = {}, {}, {}, [], [], []
    for t, df in data.items():
        tgt, _ = target_rsi2(df)
        r, h = sleeve_returns(df, tgt)
        start = df.index[200]  # serve la media a 200 giorni
        sleeves[t], held[t] = r[r.index >= start], h[h.index >= start]
        b, _ = sleeve_returns(df, pd.Series(1.0, index=df.index))
        bench[t] = b[b.index >= start]
        tr = [x for x in extract_trades(df, tgt, t) if x["entry"] >= start]
        trades += tr
        tm = trade_metrics(tr)
        per_ticker.append({"ticker": t, "name": universe[t]["name"], "category": universe[t]["category"],
                           "n": tm["n"], "avg_pct": tm.get("avg_pct"), "win_rate": tm.get("win_rate"),
                           "total_pct": float(((1 + pd.Series([x["ret_pct"] / 100 for x in tr])).prod() - 1) * 100) if tr else 0.0})
        st = today_rsi2(df)
        if st:
            today.append({"ticker": t, "name": universe[t]["name"], **st})
    strat = portfolio_mean(sleeves)
    exposure = pd.DataFrame(held).mean(axis=1)
    results["mean_reversion_rsi2"] = package(
        "mean_reversion_rsi2", strat, portfolio_mean(bench), exposure, trades,
        sorted(per_ticker, key=lambda x: -(x["avg_pct"] or -99)), today,
        "Compra e tieni degli stessi titoli, pesi uguali",
        {"note": "Ogni titolo ha una quota fissa di capitale, in liquidità quando non è in posizione: per questo il rendimento annuo del portafoglio è basso rispetto al guadagno per trade."})

    # ---------------- 2. Trend following 10 mesi sugli indici -------------
    idx_tickers = [t for t in data if universe[t]["category"] in ("Indici Americani", "Indici Europei")]
    sleeves, bench, held, trades, per_ticker, today = {}, {}, {}, [], [], []
    for t in idx_tickers:
        df = data[t]
        tgt = target_trend10(df)
        first_dec = monthly_sma_above(df["Close"]).dropna().index
        if len(first_dec) == 0:
            continue
        start = first_dec[0]
        r, h = sleeve_returns(df, tgt)
        sleeves[t], held[t] = r[r.index > start], h[h.index > start]
        b, _ = sleeve_returns(df, pd.Series(1.0, index=df.index))
        bench[t] = b[b.index > start]
        tr = [x for x in extract_trades(df, tgt, t) if x["entry"] > start]
        trades += tr
        s_m, b_m = perf_metrics(sleeves[t]), perf_metrics(bench[t])
        per_ticker.append({"ticker": t, "name": universe[t]["name"], "category": universe[t]["category"],
                           "n": len(tr), "cagr_pct": s_m.get("cagr_pct"), "max_dd_pct": s_m.get("max_dd_pct"),
                           "bench_cagr_pct": b_m.get("cagr_pct"), "bench_max_dd_pct": b_m.get("max_dd_pct")})
        st = today_trend10(df)
        if st:
            today.append({"ticker": t, "name": universe[t]["name"], **st})
    results["trend_10_mesi"] = package(
        "trend_10_mesi", portfolio_mean(sleeves), portfolio_mean(bench), pd.DataFrame(held).mean(axis=1),
        trades, sorted(per_ticker, key=lambda x: -(x["cagr_pct"] or -99)), today,
        "Compra e tieni degli stessi indici, pesi uguali")

    # ---------------- 3. Momentum 12-1 azioni italiane --------------------
    it = [t for t in data if universe[t]["category"] == "Azioni Italiane"]
    if REGIME_TICKER in data and it:
        closes = pd.DataFrame({t: data[t]["Close"] for t in it}).sort_index()
        regime = monthly_sma_above(data[REGIME_TICKER]["Close"].reindex(closes.index).ffill())
        tgt_all, mom = targets_momentum(closes, regime)
        top_n = 5
        sleeves, held, trades, per_ticker = {}, {}, [], []
        first_pick = tgt_all.index[(tgt_all.sum(axis=1) > 0).to_numpy()]
        # inizio: primo fine mese con 12 mesi di storia (anche se in liquidità)
        start = mom.dropna(how="all").index[0] if not mom.dropna(how="all").empty else closes.index[0]
        for t in it:
            df = data[t]
            tgt = tgt_all[t].reindex(df.index).ffill().fillna(0)
            r, h = sleeve_returns(df, tgt)
            sleeves[t] = (r / top_n).reindex(closes.index).fillna(0)
            held[t] = h.reindex(closes.index).fillna(0)
            tr = [x for x in extract_trades(df, tgt, t) if x["entry"] > start]
            trades += tr
            tm = trade_metrics(tr)
            per_ticker.append({"ticker": t, "name": universe[t]["name"], "category": universe[t]["category"],
                               "n": tm["n"], "avg_pct": tm.get("avg_pct"), "win_rate": tm.get("win_rate"),
                               "months_held": int(round(held[t][held[t].index > start].sum() / 21))})
        strat = pd.DataFrame(sleeves).sum(axis=1)
        strat = strat[strat.index > start]
        bench = portfolio_mean({t: sleeve_returns(data[t], pd.Series(1.0, index=data[t].index))[0] for t in it})
        bench = bench[bench.index > start]
        invested = (pd.DataFrame(held).sum(axis=1) > 0).astype(float)
        # selezione attuale (ultima decisione di fine mese) e anteprima con i prezzi di oggi
        last_me = tgt_all.index[(tgt_all.diff().abs().sum(axis=1) > 0).to_numpy()]
        cur = tgt_all.iloc[-1]
        last_mom = (closes.iloc[-22] / closes.iloc[-253] - 1) if len(closes) > 253 else pd.Series(dtype=float)
        reg_now = data[REGIME_TICKER]["Close"].iloc[-1] > pd.concat([
            data[REGIME_TICKER]["Close"][monthly_sma_above(data[REGIME_TICKER]["Close"]).notna()].iloc[-9:],
            data[REGIME_TICKER]["Close"].iloc[-1:]]).mean()
        today = [{"ticker": t, "name": universe[t]["name"], "held": bool(cur.get(t, 0) == 1),
                  "momentum_pct": float(last_mom.get(t)) * 100 if t in last_mom and pd.notna(last_mom.get(t)) else None}
                 for t in it]
        today.sort(key=lambda x: -(x["momentum_pct"] if x["momentum_pct"] is not None else -1e9))
        for i, row in enumerate(today):
            row["rank"] = i + 1
        results["momentum_12_1"] = package(
            "momentum_12_1", strat, bench, invested, trades,
            sorted(per_ticker, key=lambda x: -(x["months_held"] or 0)), today,
            "Compra e tieni di tutte le azioni italiane della lista, pesi uguali",
            {"regime_ok_now": bool(reg_now),
             "note": "Usa solo le azioni quotate oggi: quelle fallite o uscite dalla borsa mancano, e questo favorisce il momentum più del compra e tieni."})

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": args.period, "split": str(SPLIT.date()), "cost_per_side": COST_PER_SIDE,
        "universe": {t: {**universe[t], "start": str(d.index[0].date()), "end": str(d.index[-1].date()), "days": len(d)}
                     for t, d in data.items()},
        "skipped": skipped,
        "runtime_seconds": round(time.time() - t0),
        "strategies": results,
    }

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else round(float(x), 4)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, pd.Timestamp):
            return str(x.date())
        return x

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(clean(out), fh, ensure_ascii=False, indent=1)

    print(f"\nSalvato {args.out} — {len(data)} titoli, saltati: {skipped}\n")
    for k, s in out["strategies"].items():
        f, b = s["strategy"]["full"], s["bench"]["full"]
        print(f"{s['name']:32s} CAGR {f.get('cagr_pct') or 0:6.2f}% Sharpe {f.get('sharpe') or 0:5.2f} "
              f"DD {f.get('max_dd_pct') or 0:6.1f}% | 1° {s['strategy']['first'].get('cagr_pct') or 0:6.2f}% "
              f"2° {s['strategy']['second'].get('cagr_pct') or 0:6.2f}% | B&H CAGR {b.get('cagr_pct') or 0:6.2f}% "
              f"Sharpe {b.get('sharpe') or 0:5.2f} DD {b.get('max_dd_pct') or 0:6.1f}% | trade {s['trades'].get('n')} "
              f"medio {s['trades'].get('avg_pct') or 0:.2f}% -> {s['verdict']}")


if __name__ == "__main__":
    main()
