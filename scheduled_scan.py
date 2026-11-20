"""
scheduled_scan.py — Messaggio Telegram serale con le operazioni da fare
il giorno dopo, eseguito da GitHub Actions (.github/workflows/scan_daily.yml)
dal lunedì al venerdì dopo la chiusura di Wall Street, quando le candele
giornaliere di Europa e USA sono complete.

Mostra solo le strategie attive in ALERT_STRATEGY_CONFIG (core.py) e usa
lo stesso calcolo della pagina "Cosa fare oggi" dell'app (signals_today.py).

Uso:
    python scheduled_scan.py            # calcola e invia
    python scheduled_scan.py --dry-run  # stampa soltanto
"""

import argparse
import html
import sys

from core import ALERT_STRATEGY_CONFIG, send_telegram_message, BOT_TOKEN, CHAT_ID
from signals_today import all_signals
from strategies_pro import STRATEGY_INFO

ICON = {"mean_reversion_rsi2": "🔁", "trend_10_mesi": "📈", "momentum_12_1": "🚀"}


def fmt_row(r):
    name = f" {html.escape(r['name'])}" if r["name"] != r["ticker"] else ""
    s = f"<b>{html.escape(r['ticker'])}</b>{name}"
    if r["action"] in ("TIENI", "VENDI") and r.get("since") and r.get("open_ret_pct") is not None:
        fmt = "%d/%m" if r["since"].year == r["date"].year else "%d/%m/%y"
        s += f" (dal {r['since'].strftime(fmt)}, {r['open_ret_pct']:+.1f}%)"
    return s


def build_message(sig):
    active = [k for k in STRATEGY_INFO if ALERT_STRATEGY_CONFIG.get(k, False)]
    rows = sig["rows"]
    lines = [f"📊 <b>Market Scanner Pro</b> — chiusura del {sig['last_date']:%d/%m/%Y}",
             "Le operazioni vanno fatte <b>domani all'apertura</b>.", ""]
    n_ops = 0
    for k in active:
        sub = [r for r in rows if r["strategy"] == k]
        buy = [r for r in sub if r["action"] == "COMPRA"]
        sell = [r for r in sub if r["action"] == "VENDI"]
        hold = [r for r in sub if r["action"] == "TIENI"]
        n_ops += len(buy) + len(sell)
        lines.append(f"{ICON.get(k, '•')} <b>{STRATEGY_INFO[k]['name']}</b>")
        for r in buy:
            lines.append(f"🟢 COMPRA {fmt_row(r)} — {html.escape(r['why'])}")
        for r in sell:
            lines.append(f"🔴 VENDI {fmt_row(r)}")
        if not buy and not sell:
            lines.append("Nessuna operazione.")
        if hold:
            label = "Investito su" if k == "trend_10_mesi" else "In portafoglio"
            lines.append(f"⏳ {label}: " + ", ".join(fmt_row(r) for r in hold))
        if k == "momentum_12_1" and sig.get("momentum"):
            lines.append(("✅ " if sig["momentum"]["regime_ok"] else "⛔ ") + html.escape(sig["momentum"]["regime_text"]))
        lines.append("")
    if not active:
        lines.append("Nessuna strategia attiva (vedi ALERT_STRATEGY_CONFIG in core.py).")
    if sig["missing"]:
        lines.append("⚠️ Senza dati oggi: " + ", ".join(sig["missing"]))
    return "\n".join(lines).strip(), n_ops


def main():
    ap = argparse.ArgumentParser(description="Messaggio Telegram serale di Market Scanner Pro")
    ap.add_argument("--dry-run", action="store_true", help="Stampa il messaggio senza inviarlo")
    ap.add_argument("--market", help="Ignorato (compatibilità con i vecchi workflow)")
    args = ap.parse_args()

    sig = all_signals()
    if not sig["rows"] and len(sig["missing"]) == len(sig["universe"]):
        msg = "❗ <b>Market Scanner Pro</b>: nessun dato scaricato oggi, probabile problema con Yahoo Finance."
        n_ops = 0
    else:
        msg, n_ops = build_message(sig)
    print(msg)
    print(f"\nOperazioni per domani: {n_ops}")
    if args.dry_run:
        return
    if not BOT_TOKEN or not CHAT_ID:
        print("ATTENZIONE: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID non impostati (Secrets del repository): messaggio non inviato.")
        return
    send_telegram_message(BOT_TOKEN, CHAT_ID, msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Errore fatale: {e}")
        sys.exit(1)
