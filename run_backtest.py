"""
run_backtest.py — Esegue il backtest realistico (backtest_v2.py) su tutto
l'universo e salva i risultati in un file JSON, letto dalla pagina dei
risultati. Pensato per girare su GitHub Actions
(.github/workflows/backtest.yml), dove Yahoo Finance è raggiungibile.

Uso:
    python run_backtest.py                       # 10 anni, tutto l'universo
    python run_backtest.py --period 5y --out results/bt.json
    python run_backtest.py --tickers ENI.MI ISP.MI
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core import TICKER_CATALOG, download_data
from backtest_v2 import (
    STRATEGIES, DEFAULT_PARAMS, run_ticker, summarize, verdict, strategy_alert_state,
)

SPLIT_DATE = "2022-01-01"  # prima: in-sample, da qui in poi: out-of-sample


def build_universe():
    tickers = {}
    for category, items in TICKER_CATALOG.items():
        for name, t in items.items():
            tickers[t] = {"name": name, "category": category}
    return tickers


def _worker(ticker, period):
    try:
        data = download_data(ticker, period=period, interval="1d")
    except Exception as e:
        return ticker, None, f"download fallito: {e}"
    if data is None or data.empty or len(data) < 300:
        return ticker, None, f"dati insufficienti ({0 if data is None else len(data)} barre)"
    t0 = time.time()
    res = run_ticker(data)
    info = {"first": str(data.index[0].date()), "last": str(data.index[-1].date()),
            "bars": len(data), "seconds": round(time.time() - t0, 1)}
    return ticker, {"info": info, "res": res}, None


def _json_trade(t, ticker):
    return {
        "ticker": ticker,
        "entry": str(pd.Timestamp(t["entry_date"]).date()),
        "exit": str(pd.Timestamp(t["exit_date"]).date()),
        "outcome": t["outcome"],
        "r": round(float(t["r"]), 3),
        "ret_pct": round(float(t["ret_pct"]), 2),
        "days": int(t["days"]),
    }


def _clean(x):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def aggregate(per_ticker, universe):
    split = pd.Timestamp(SPLIT_DATE)
    out = {}
    for key, (label, _) in STRATEGIES.items():
        new_all, old_all, signal_days = [], [], 0
        by_ticker = []
        for tkr, payload in per_ticker.items():
            r = payload["res"][key]
            signal_days += r["n_signal_days"]
            new_all += [dict(t, ticker=tkr) for t in r["new"]]
            old_all += r["old"]
            s = summarize(r["new"])
            by_ticker.append({"ticker": tkr, "name": universe[tkr]["name"],
                              "category": universe[tkr]["category"],
                              "n": s["n"], "avg_r": s.get("avg_r"), "total_r": s.get("total_r"),
                              "win_rate": s.get("win_rate")})
        full = summarize(new_all)
        ins = summarize([t for t in new_all if pd.Timestamp(t["entry_date"]) < split])
        oos = summarize([t for t in new_all if pd.Timestamp(t["entry_date"]) >= split])
        old = summarize(old_all)
        v, v_text = verdict(full, oos)

        # curva dei guadagni: R cumulati a fine mese (tutti i titoli insieme)
        equity = []
        if new_all:
            s = pd.Series([t["r"] for t in new_all],
                          index=pd.to_datetime([t["exit_date"] for t in new_all])).sort_index()
            monthly = s.cumsum().resample("ME").last().ffill()
            equity = [{"date": str(d.date()), "r": round(float(v_), 2)} for d, v_ in monthly.items()]

        by_year = {}
        for t in new_all:
            y = pd.Timestamp(t["exit_date"]).year
            b = by_year.setdefault(y, {"year": y, "n": 0, "total_r": 0.0})
            b["n"] += 1
            b["total_r"] += float(t["r"])
        years = sorted(by_year.values(), key=lambda b: b["year"])
        for b in years:
            b["total_r"] = round(b["total_r"], 2)

        recent = sorted(new_all, key=lambda t: t["entry_date"], reverse=True)[:40]
        out[key] = {
            "key": key,
            "name": label,
            "alert_on": strategy_alert_state(key),
            "signal_days": signal_days,
            "verdict": v,
            "verdict_text": v_text,
            "full": full, "in_sample": ins, "out_of_sample": oos, "old_method": old,
            "equity": equity,
            "by_year": years,
            "by_ticker": sorted(by_ticker, key=lambda b: (b["total_r"] or 0), reverse=True),
            "recent_trades": [_json_trade(t, t["ticker"]) for t in recent],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="10y")
    ap.add_argument("--out", default="results/backtest_results.json")
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    universe = build_universe()
    if args.tickers:
        universe = {t: universe.get(t, {"name": t, "category": "Personalizzato"}) for t in args.tickers}

    per_ticker, skipped, info = {}, [], {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t, args.period): t for t in universe}
        for f in as_completed(futs):
            tkr, payload, err = f.result()
            if err:
                skipped.append({"ticker": tkr, "reason": err})
                print(f"[skip] {tkr}: {err}", flush=True)
            else:
                per_ticker[tkr] = payload
                info[tkr] = payload["info"]
                print(f"[ok]   {tkr}: {payload['info']}", flush=True)

    if not per_ticker:
        print("Nessun titolo scaricato: impossibile fare il backtest.")
        sys.exit(1)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": args.period,
        "split_date": SPLIT_DATE,
        "params": DEFAULT_PARAMS,
        "universe": {t: dict(universe[t], **info[t]) for t in sorted(info)},
        "skipped": skipped,
        "runtime_seconds": round(time.time() - t0),
        "strategies": aggregate(per_ticker, universe),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(_clean(result), fh, ensure_ascii=False, indent=1, default=str)

    print(f"\nSalvato {args.out} ({len(per_ticker)} titoli, {len(skipped)} saltati)\n")
    for key, s in result["strategies"].items():
        f, o, old = s["full"], s["out_of_sample"], s["old_method"]
        print(f"{s['name'][:45]:45s} n={f.get('n',0):5d} avgR={f.get('avg_r') or 0:+.3f} "
              f"PF={f.get('profit_factor') or 0:.2f} t={f.get('t_stat') or 0:+.2f} "
              f"OOS avgR={o.get('avg_r') or 0:+.3f} | vecchio n={old.get('n',0)} "
              f"avgR={old.get('avg_r') or 0:+.3f}  -> {s['verdict']}")


if __name__ == "__main__":
    main()
