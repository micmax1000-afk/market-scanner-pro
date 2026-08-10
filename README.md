# Market Scanner Pro V3 + ML — versione corretta

## Come avviarlo

```bash
pip install -r requirements.txt

# (opzionale) alert Telegram: imposta le variabili d'ambiente
export TELEGRAM_BOT_TOKEN="il-tuo-token"
export TELEGRAM_CHAT_ID="il-tuo-chat-id"

streamlit run app.py
```

Si apre in automatico su `http://localhost:8501`.

## Cosa ho corretto rispetto alla versione originale

1. **Bug critico — lookahead bias nel dataset ML** (`build_ml_dataset`):
   le colonne `TrendlineScore`, `BreakoutScore`, `ScoreV3` venivano
   calcolate una volta sull'intero storico e ripetute identiche su ogni
   riga, quindi il modello "vedeva" implicitamente dati futuri durante
   il training. Ora vengono calcolate punto-per-punto usando solo i
   dati disponibili fino a quel giorno (`compute_pointwise_scores`),
   come già faceva correttamente `backtest_score`. Verificato con test
   automatico che i valori variano nel tempo invece di essere costanti.

2. **Credenziali Telegram**: non più in chiaro nel codice, ma lette da
   variabili d'ambiente (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). Se
   non le imposti, l'app funziona comunque, semplicemente senza alert.

3. **Robustezza `yfinance`**: le versioni recenti possono restituire
   colonne con MultiIndex anche per un singolo ticker, causando errori
   più a valle. Aggiunto `download_data()` che normalizza sempre le
   colonne.

4. **Divisioni per zero**: in RSI, Stocastico e in vari filtri, quando
   il denominatore era 0 (es. nessuna variazione di prezzo/volume)
   potevano generarsi `inf`/`NaN` propagati silenziosamente. Aggiunte
   guardie esplicite.

5. **OBV vettorizzato**: il calcolo con loop `for` riga per riga è
   stato sostituito con un'operazione vettoriale equivalente ma molto
   più veloce, specialmente su scanner con molti titoli.

6. **Disclaimer visibili in app**: aggiunta nota che gli score e le
   probabilità del modello ML sono strumenti informativi, non consigli
   di investimento.

7. Piccoli fix minori: filtro su dati insufficienti (`len(data) < 200`)
   prima di calcolare gli score nello scanner V2, timeout sulla
   richiesta Telegram, `data.copy()` per evitare `SettingWithCopyWarning`.

8. **Bug — lo score pesato non usava mai la scala 0-100**: la somma
   pesata dei punteggi grezzi non superava mai ~17 (verificato: `20*0.25
   + 15*0.10 + 15*0.10 + 20*0.20 + 15*0.20 + 15*0.10 + 10*0.05 = 17`),
   mentre altrove nel codice si confrontava lo score con soglie come
   `score >= 75` (per gli alert forti e per il backtest): quella soglia
   di fatto non scattava quasi mai. Ora ogni componente viene
   normalizzata sul proprio massimo prima di applicare il peso, così lo
   score finale usa l'intera scala 0-100.

## Nuovi indicatori aggiunti

- **Bollinger Bands** — bande di volatilità (media mobile 20 periodi ±
  2 deviazioni standard), visibili sul grafico principale. Punteggio
  bonus se il prezzo chiude sopra la banda superiore (forte momentum)
  o è vicino ad essa.
- **ADX (Average Directional Index)** — misura quanto è *forte* il
  trend in corso (non la direzione). Grafico dedicato nella tab
  "Indicatori", con le linee +DI/-DI. Punteggio bonus se ADX > 25 e
  +DI > -DI (trend rialzista forte e confermato).
- **VWAP** — prezzo medio ponderato per volume, ancorato dall'inizio
  del periodo caricato (essendo barre giornaliere, non è il VWAP
  intraday classico che riparte ogni giorno). Linea tratteggiata sul
  grafico principale. Punteggio bonus se il prezzo è sopra il VWAP.
- **Rottura della trend line** — non è più una semplice regressione su
  tutte le chiusure, ma la trend line "classica" tracciata sugli ultimi
  pivot (i minimi per il supporto rialzista, i massimi per la
  resistenza ribassista), esattamente come si traccia a mano su una
  piattaforma di trading. Un pivot è un minimo/massimo locale (il più
  basso/alto in una finestra di qualche giorno prima e dopo). Sul
  grafico vedi la linea disegnata e i punti di tocco (pivot) segnati
  con dei triangoli:
  - **verde** = supporto rialzista (connette gli ultimi minimi crescenti)
  - **rosso** = resistenza ribassista (connette gli ultimi massimi decrescenti)

  Punteggio massimo se il prezzo rompe *sopra* la resistenza proprio
  oggi (segnale di forza); nessun bonus se rompe *sotto* il supporto
  (segnale di debolezza, comunque evidenziato nella tabella dello
  Scanner e negli alert Telegram con 🔀⬆️ / 🔀⬇️). Se in un dato momento
  non ci sono abbastanza pivot con la pendenza giusta, la linea
  semplicemente non viene disegnata (non tutti i momenti di mercato
  hanno una trend line pulita da tracciare).

Tutti e quattro sono stati integrati anche nello score pesato V3 e tra
le feature del modello ML (Scanner V3).

## Segnale ACQUISTO / VENDITA (rottura trend line + conferme)

Una rottura di trend line da sola genera troppi falsi segnali, quindi
ho aggiunto una regola di conferma:

- **ACQUISTO** → il prezzo rompe *sopra* la trend line ribassista (sui
  massimi decrescenti) **e** almeno 3 tra questi indicatori confermano
  la stessa direzione: RSI > 50, MACD sopra Signal, ADX > 20 con
  +DI > -DI, prezzo sopra il VWAP, Stocastico K > D.
- **VENDITA** → il prezzo rompe *sotto* la trend line rialzista (sui
  minimi crescenti) **e** almeno 3 degli indicatori equivalenti
  confermano la direzione ribassista.
- **NEUTRALE** → nessuna rottura oggi, oppure rottura senza abbastanza
  conferme.

Il segnale compare in modo evidente nella tab "Score" (con il dettaglio
di quali indicatori hanno confermato), come colonna nella tabella dello
Scanner V2, e in cima agli alert Telegram. La soglia di 3 conferme è
modificabile passando `min_confirmations` a `compute_trade_signal()` in
`app.py` se vuoi renderlo più o meno selettivo.

## Limite di questo test

Non ho potuto testare lo script con dati reali di Yahoo Finance perché
l'ambiente in cui ho lavorato non ha accesso a `finance.yahoo.com`. Ho
validato tutta la logica (indicatori, score, filtri, backtest, dataset
ML, training) con dati sintetici generati casualmente: nessun errore,
e il fix del lookahead bias è confermato. Prova comunque tu per primo
lo scanner con un paio di titoli prima di fidarti degli alert su tutta
la lista.

## Nota importante

Questo è uno strumento di analisi/scoring algoritmico, non un bot che
esegue operazioni reali. Gli score (V2 pesato, ML, backtest storici) si
basano su pattern passati e non garantiscono risultati futuri — usali
come un input in più nella tua analisi, non come unica base decisionale.

## Timeframe, indici e menu a tendina

- **Timeframe 4 ore** — yfinance non fornisce l'intervallo "4h"
  direttamente: viene scaricato l'orario (`1h`, limitato dalla stessa
  yfinance a circa gli ultimi 60-730 giorni a seconda del periodo) e
  raggruppato automaticamente in barre da 4 ore. Selezionabile dal menu
  "Timeframe" in alto, si applica al Grafico, agli Indicatori, allo
  Score e allo Scanner V2. Il Backtest e lo Scanner V3 (ML) restano
  sempre su dati giornalieri, perché richiedono più storia di quanta
  ne offra il 4 ore.
- **Indici USA ed Europei** — aggiunti S&P 500, Dow Jones, Nasdaq
  Composite, Russell 2000 (USA) e FTSE MIB, DAX, CAC 40, FTSE 100,
  Euro Stoxx 50 (Europa), selezionabili come le azioni italiane.
- **Menu a tendina** — al posto del campo di testo libero, un
  selectbox categorizzato (Azioni Italiane / Indici Americani / Indici
  Europei) più un'opzione "Personalizzato..." per digitare qualsiasi
  altro ticker (es. azioni USA come `AAPL`, criptovalute come
  `BTC-USD`, ecc. — yfinance supporta molti mercati).
- **Scanner multi-timeframe e multi-universo** — nella tab Scanner puoi
  scegliere quali gruppi di titoli includere (azioni italiane, indici
  USA, indici Europa, anche insieme) e su quali timeframe farlo girare
  (Giornaliero, 4 Ore, o entrambi): lo scanner gira una volta per ogni
  timeframe selezionato e invia un alert Telegram separato per
  ciascuno, con l'etichetta del timeframe inclusa nel messaggio.

## Pubblicarlo online (Streamlit Community Cloud)

GitHub Pages **non funziona per questa app**: pubblica solo siti
statici (HTML/CSS/JS), mentre questa è un'app Python che deve girare
su un server attivo. La soluzione gratuita pensata apposta per le app
Streamlit è **Streamlit Community Cloud**:

1. Crea un repository GitHub (pubblico, il piano gratuito richiede
   repository pubblici) e carica tutti i file di questa cartella
   (`app.py`, `requirements.txt`, `README.md`, `.gitignore`) — stessa
   procedura di upload manuale già usata per l'altro progetto.
2. Vai su [share.streamlit.io](https://share.streamlit.io), accedi con
   il tuo account GitHub.
3. Clicca "New app", seleziona il repository, il branch (`main`) e come
   "Main file path" scrivi `app.py`. Clicca "Deploy".
4. Per gli alert Telegram: nella pagina dell'app su Streamlit Cloud,
   vai su "Settings" → "Secrets" e incolla:
   ```toml
   TELEGRAM_BOT_TOKEN = "il-tuo-token"
   TELEGRAM_CHAT_ID = "il-tuo-chat-id"
   ```
   Il codice li legge automaticamente da lì (non servono variabili
   d'ambiente sul server, funziona sia in locale sia su Streamlit
   Cloud senza modifiche).

Nota: sul piano gratuito l'app "dorme" dopo un periodo di inattività e
si riattiva al primo accesso (qualche secondo di attesa). Inoltre, se
il repository è pubblico, chiunque può vedere il codice sorgente (non
le tue credenziali, che restano nei Secrets criptati di Streamlit
Cloud, mai nel repository).
