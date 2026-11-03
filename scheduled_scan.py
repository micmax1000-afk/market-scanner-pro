"""
scheduled_scan.py — Scanner automatico eseguito da GitHub Actions
(vedi .github/workflows/scan_europe.yml e scan_usa.yml).

Non usa Streamlit: importa solo la logica di calcolo da core.py e
manda gli alert direttamente su Telegram, esattamente come fa lo
Scanner V2 dentro l'app, ma senza bisogno che qualcuno apra il browser.

Uso:
    python scheduled_scan.py --market europe
    python scheduled_scan.py --market usa

Ogni esecuzione scansiona l'universo scelto SOLO in Giornaliero (1d):
tutti i backtest fatti finora (R medio, Profit Factor, drawdown delle
3 strategie testate) sono stati validati esclusivamente sui dati
giornalieri. Il 4 Ore resta disponibile nell'app per uso manuale ed
esplorativo, ma non genera più alert automatici, perché applicare le
stesse regole a un timeframe mai testato (oltre ad essere ricostruito
per approssimazione da dati orari, con meno storico disponibile per
il "riscaldamento" degli indicatori) equivarrebbe a fidarsi di un
segnale senza nessuna prova storica alle spalle — lo stesso motivo
per cui le strategie non profittevoli sono state disattivate dagli
alert.
"""

import argparse
import sys

from core import (
    TICKER_CATALOG,
    run_scanner_v2,
    send_telegram_message,
    BOT_TOKEN,
    CHAT_ID,
)


def build_universe(market):
    if market == "europe":
        tickers = {}
        tickers.update(TICKER_CATALOG["Azioni Italiane"])
        tickers.update(TICKER_CATALOG["Indici Europei"])
        return list(dict.fromkeys(tickers.values())), "Mercati Europei"
    elif market == "usa":
        tickers = dict(TICKER_CATALOG["Indici Americani"])
        return list(dict.fromkeys(tickers.values())), "Mercati Americani"
    else:
        raise ValueError(f"Mercato non riconosciuto: {market}")


def run_market_scan(market):
    tickers, market_label = build_universe(market)

    if not BOT_TOKEN or not CHAT_ID:
        print("ATTENZIONE: TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID non impostati, "
              "gli alert non verranno inviati (controlla i Secrets del repository).")

    print(f"Scansione {market_label} — Giornaliero ({len(tickers)} titoli)...")
    total_hits = 0
    stats = {"analyzed": 0, "alerts_sent": 0, "skipped_no_data": []}
    try:
        df = run_scanner_v2(
            tickers=tickers,
            interval="1d",
            period="2y",  # servono almeno 200 barre (EMA200)
            timeframe_label="Giornaliero",
        )
        total_hits = len(df) if df is not None and not df.empty else 0
        stats = df.attrs.get("stats", stats)
        print(f"  -> {total_hits} titoli hanno superato i filtri dello scanner.")
    except Exception as e:
        print(f"  -> Errore durante la scansione: {e}")

    summary = (
        f"✅ <b>Scanner automatico completato</b>\n"
        f"Mercato: {market_label}\n"
        f"Timeframe: Giornaliero\n"
        f"Titoli nell'universo: {len(tickers)}\n"
        f"Titoli analizzati (dati sufficienti): {stats['analyzed']}\n"
        f"🔔 Alert inviati: {stats['alerts_sent']}\n"
        f"Titoli che passano anche i 7 filtri dello Scanner V2: {total_hits}\n"
    )
    if stats["skipped_no_data"]:
        summary += f"\n⚠️ Saltati per dati insufficienti: {', '.join(stats['skipped_no_data'])}\n"
    if stats["alerts_sent"] == 0 and stats["analyzed"] > 0:
        summary += "\nNessuna strategia attiva ha dato segnale oggi."
    elif stats["analyzed"] == 0:
        summary += "\n❗ Nessun titolo analizzato: probabile problema di download dati."

    if BOT_TOKEN and CHAT_ID:
        send_telegram_message(BOT_TOKEN, CHAT_ID, summary)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scanner automatico Market Scanner Pro")
    parser.add_argument("--market", choices=["europe", "usa"], required=True,
                         help="Quale universo scansionare: europe o usa")
    args = parser.parse_args()

    try:
        run_market_scan(args.market)
    except Exception as e:
        print(f"Errore fatale nello scanner automatico: {e}")
        sys.exit(1)

