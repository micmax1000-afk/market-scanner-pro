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

## Backtest del segnale ACQUISTO/VENDITA (confronto soglie 2/3/4)

Nella tab "Backtest" c'è una sezione dedicata a testare storicamente
se 2, 3 o 4 conferme minime funzionano meglio, su un universo di
titoli a tua scelta e sugli ultimi ~2 anni di dati giornalieri.

Cammina nel tempo giorno per giorno usando solo i dati fino a quel
momento (nessun lookahead), registra ogni volta che scatta un segnale
ACQUISTO o VENDITA con ciascuna soglia, e alla fine misura quante
volte il prezzo si è davvero mosso nella direzione prevista entro
l'orizzonte scelto (5, 10 o 20 giorni).

Da tenere presente leggendo i risultati: soglie più alte danno sempre
un numero di segnali minore o uguale (più selettive), ma una
percentuale di successo alta su pochi segnali (es. 60% su 5 casi) è
poco affidabile — un numero maggiore di segnali con una percentuale
comunque buona è generalmente un'indicazione più solida.

## Due nuove strategie di solo ACQUISTO

Oltre al segnale combinato (rottura trend line + conferme), ci sono
ora due strategie indipendenti, entrambe visibili nella tab "Score"
(con il dettaglio di ogni condizione ✅/❌), come colonne nello Scanner,
e testabili singolarmente nella tab "Backtest".

### 📉 Rottura resistenza discendente con momentum ancora basso

Compra la rottura (non il ritracciamento) quando **tutte** queste
condizioni sono vere insieme:
- Il prezzo rompe (o è appena sopra) la trend line di resistenza
  discendente (entro il 2%)
- Stocastico 10-3-6 ancora basso (K < 40)
- RSI ancora basso (< 50)
- Bande di Bollinger "vicine" tra loro, cioè in squeeze: la larghezza
  attuale delle bande è tra il 30% più stretto degli ultimi 100 giorni

L'idea: una rottura di resistenza mentre gli oscillatori sono ancora
bassi (non ipercomprati) avviene "presto" nel movimento, lasciando più
margine di salita rispetto a comprare una rottura quando RSI/Stocastico
sono già a 80-90. Lo squeeze delle Bollinger è un classico segnale di
bassa volatilità che spesso precede un movimento direzionale forte,
quindi rafforza la view che la rottura non sia "rumore".

### 📈 Pullback alla EMA20 in trend forte (strategia proposta)

L'idea di base: comprare un breakout puro spesso significa comprare
quando il titolo è già "esteso", con più rischio di rientro
immediato. Storicamente tende a funzionare meglio comprare i
ritracciamenti superficiali **dentro** un trend già forte e
confermato, non l'inseguimento della rottura. Condizioni:
- Trend strutturale rialzista: prezzo sopra EMA200, EMA20 sopra EMA50
- Trend abbastanza forte da avere senso seguirlo: ADX > 20 con
  +DI > -DI
- Ritracciamento superficiale: il prezzo è tornato vicino alla EMA20
  (entro l'1.5%), non un crollo profondo
- MACD sopra la Signal line (conferma di ripartenza)
- Volume non anomalo in negativo (almeno il 70% della media a 20
  giorni, per evitare titoli poco liquidi in quel momento)

Entrambe le soglie (percentuali di tolleranza, RSI/Stocastico max,
ecc.) sono modificabili passando parametri diversi alle rispettive
funzioni `compute_strategy_pullback_oversold()` e
`compute_strategy_trend_pullback()` in `app.py`.

## Scanner automatici (Lun-Ven) con GitHub Actions

L'app Streamlit da sola NON può eseguire scansioni a orari fissi in
background: gira solo quando qualcuno apre il link nel browser. Per
avere scanner davvero automatici, il progetto usa **GitHub Actions**
(gratuito) come scheduler, con uno script separato (`scheduled_scan.py`)
che non dipende da Streamlit.

Come funziona:
- **Mercati Europei**: ogni giorno feriale alle ~10:00 (ora italiana),
  scansiona Azioni Italiane + Indici Europei, sia Giornaliero che 4 Ore
- **Mercati Americani**: ogni giorno feriale alle ~18:00 (ora italiana),
  scansiona gli Indici Americani, sia Giornaliero che 4 Ore
- Ogni scansione manda gli alert individuali su Telegram (stessa logica
  dello Scanner V2 nell'app) più un messaggio di riepilogo finale

### Setup (una tantum)

1. **Struttura del codice**: la logica di calcolo ora vive in `core.py`
   (nuovo file), riusato sia da `app.py` (interfaccia) sia da
   `scheduled_scan.py` (scanner automatico). Carica entrambi i file
   più `scheduled_scan.py` e la cartella `.github/workflows/` sul
   repository GitHub (assicurati che il drag&drop includa anche le
   cartelle nascoste come `.github/` — su GitHub.com il modo più
   sicuro è "Add file" → "Upload files" trascinando l'intera cartella
   del progetto, oppure usare `git push` da terminale).

2. **Secrets per GitHub Actions**: gli scanner automatici hanno
   bisogno di leggere `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID`, ma
   NON dai Secrets di Streamlit Cloud (sono due sistemi separati). Vai
   su GitHub → il tuo repository → **Settings** → **Secrets and
   variables** → **Actions** → **New repository secret**, e crea:
   - `TELEGRAM_BOT_TOKEN` con il token del bot
   - `TELEGRAM_CHAT_ID` con il chat id

3. **Verifica che sia attivo**: vai su GitHub → il tuo repository →
   tab **Actions**. Dovresti vedere i due workflow "Scanner Automatico
   - Mercati Europei" e "Scanner Automatico - Mercati Americani". Puoi
   lanciarli manualmente subito per testare, senza aspettare l'orario:
   apri il workflow → **Run workflow** → **Run workflow**.

### Nota sull'ora legale/solare

GitHub Actions usa sempre l'orario UTC e non si adatta da solo al
cambio d'ora italiano. I due workflow sono impostati per essere
precisi in **ora legale** (CEST, UTC+2 — da fine marzo a fine ottobre,
che copre quasi tutta la stagione di mercato attiva):
- Europa: `0 8 * * 1-5` (08:00 UTC = 10:00 italiane in CEST)
- USA: `0 16 * * 1-5` (16:00 UTC = 18:00 italiane in CEST)

Durante l'**ora solare** (CET, UTC+1 — fine ottobre/fine marzo), gli
scanner partiranno un'ora dopo (11:00 e 19:00 italiane) a meno che tu
non aggiorni manualmente i due file in `.github/workflows/`:
- in `scan_europe.yml`, cambia `cron: '0 8 * * 1-5'` in
  `cron: '0 9 * * 1-5'`
- in `scan_usa.yml`, cambia `cron: '0 16 * * 1-5'` in
  `cron: '0 17 * * 1-5'`
e poi tornare ai valori originali a fine marzo.

### Perché Giornaliero + 4 Ore insieme

Ogni esecuzione scansiona automaticamente **entrambi** i timeframe in
sequenza (non serve scegliere): prima il Giornaliero, poi il 4 Ore,
sullo stesso universo di titoli.

## Backtest realistico con Stop Loss / Take Profit

Il backtest "a orizzonte fisso" (5/10/20 giorni) risponde solo alla
domanda "il prezzo è più alto dopo N giorni?", ignorando cosa succede
nel mezzo — non è come si opera davvero, e un "successo %" da solo
non dice se un sistema è profittevole (dipende anche da quanto guadagni
quando va bene vs quanto perdi quando va male).

La nuova sezione "🎯 Backtest realistico con Stop Loss / Take Profit"
(tab Backtest) simula un trade vero:
- **Entrata**: al prezzo di chiusura del giorno del segnale
- **Stop Loss e Take Profit**: espressi come multipli dell'ATR (la
  volatilità del titolo), non valori fissi, così si adattano
  automaticamente a ogni titolo
- **Uscita**: al primo evento tra stop loss toccato, take profit
  toccato, o scadenza del periodo massimo (chiusura al prezzo di quel
  giorno). Se nello stesso giorno vengono toccati sia stop che target,
  per prudenza si assume che lo stop sia stato colpito per primo

Funziona con qualunque delle tre strategie (segnale combinato o le due
strategie di solo acquisto), su qualunque universo di titoli.

**Come leggere il risultato:**
- **R medio**: guadagno/perdita medio per trade, espresso in multipli
  del rischio (1R = distanza tra ingresso e stop loss — es. se rischi
  l'1.5x ATR e guadagni il 3x ATR, quel trade vale +2R). Un R medio
  positivo con **Profit Factor > 1** (somma dei guadagni / somma delle
  perdite) significa che il sistema guadagna più di quanto perde in
  totale — è la metrica che conta davvero, non solo la percentuale di
  trade vincenti.
- Take Profit % + Stop Loss % + Timeout % sommano sempre a 100.

## Regime di mercato (informativo)

Ogni titolo può dare un segnale tecnico valido sul proprio grafico e
comunque muoversi contro l'onda di fondo se il mercato generale è in
un trend ribassista. Per questo, Scanner V2, tab Score e alert
Telegram mostrano ora una colonna/riga **"Regime Mercato"**:
confronta l'indice di riferimento del titolo con la propria media
mobile a 200 giorni.

- Azioni Italiane → confrontate con **FTSE MIB**
- Indici Americani → confrontati con **S&P 500**
- Indici Europei → confrontati con **Euro Stoxx 50**
- Ticker personalizzati non riconosciuti → "n/d" (nessun riferimento)

**È solo informativo: NON blocca né filtra automaticamente nessun
segnale.** Serve a dare un contesto in più prima di decidere: un
segnale di acquisto durante un regime "Ribassista" non è per forza
sbagliato, ma storicamente comprare rialzi in un mercato generale in
discesa è statisticamente meno affidabile.

## Bugfix importante: il filtro/score "Breakout" non funzionava mai

Trovato mentre allentavo le soglie dei filtri: `filter_breakout` e
`breakout_score_advanced` calcolavano la "resistenza" come il massimo
dei 60 giorni **incluso oggi**. Ma il massimo (`High`) di oggi è per
definizione sempre ≥ della chiusura (`Close`) di oggi — quindi il
rapporto breakout non poteva quasi mai essere positivo, qualunque
fosse la soglia impostata. Su un test con 140 finestre storiche
diverse, è risultato positivo **zero volte**.

Questo toccava contemporaneamente:
- Il filtro breakout dello Scanner V2 (mai passato)
- Il 13% del peso dello Score Finale Pesato V3 (componente sempre a 0)
- La condizione "breakout forte" negli alert Telegram (mai scattata)
- La colonna "Breakout Ratio" mostrata nella tabella (sempre negativa)

**Corretto**: ora la resistenza è calcolata sui 60 giorni **prima**
di oggi (escludendo oggi), poi si confronta la chiusura di oggi con
quel livello — esattamente come già faceva correttamente
`backtest_breakouts()` (verificato che dopo il fix i due calcoli
concordano).

## Soglie dei filtri allentate (Scanner V2)

Con tutti e 7 i filtri obbligatori insieme, la probabilità che un
titolo li soddisfi *tutti* contemporaneamente crolla velocemente, in
particolare su un universo piccolo come le Azioni Italiane. Tre soglie
sono state allentate rispetto ai valori originali, e sono ora anche
configurabili nell'app (Scanner V2 → "⚙️ Sensibilità dei filtri"):

| Filtro | Valore originale | Nuovo default |
|---|---|---|
| Volatilità massima 20gg | 4% | 8% |
| Volume minimo richiesto | 1.2x media | 1.0x media |
| Breakout minimo richiesto | 0.5x ATR | 0.2x ATR |

`scheduled_scan.py` (gli scanner automatici Lun-Ven) eredita
automaticamente questi nuovi default senza bisogno di modifiche.

## Score V4 (sperimentale): consolida la ridondanza del trend

Nello Score V3, 4 componenti su 11 misuravano tutte, da angolazioni
diverse, la stessa "forza del trend" (allineamento EMA, pendenza della
regressione a 60gg, rottura della trend line sui pivot, ADX) — insieme
pesavano il **42%** dello score finale. Un titolo in trend pulito
guadagnava punti 4 volte per lo stesso fenomeno.

Lo **Score V4** le consolida in un'unica componente "forza del trend"
(media delle 4 normalizzate), e ridistribuisce il peso liberato su
volume, breakout e Bollinger — dimensioni più indipendenti dal trend
puro. Non sostituisce lo Score V3 (rimasto identico ovunque
nell'app): compare **affiancato**, per confronto:

- **Tab Score**: entrambi i punteggi con la differenza (delta)
- **Scanner V2**: colonna "Score V4" accanto a "Score"

**Come interpretarlo**: se i due punteggi divergono molto per un
titolo (V3 alto, V4 più basso), probabilmente quel titolo deve gran
parte del suo Score V3 alla ripetizione dello stesso segnale di trend
(allineamento EMA + pendenza + rottura pivot + ADX, tutti concordi),
non a conferme davvero indipendenti come volume o breakout.

Verificato con un caso costruito ad hoc: un titolo con trend
perfettamente pulito ma volume piatto e nessun breakout ottiene
Score V3 più alto dello Score V4, confermando che V3 sovrappesa la
ripetizione del segnale di trend.

## Alert Telegram ristretti alle sole strategie testate

L'alert Telegram (`send_alert_v2`) ora scatta **solo** per una di
queste 3 condizioni, tutte testate con backtest (SL/TP, R medio,
Profit Factor, drawdown):

1. Segnale ACQUISTO/VENDITA combinato (rottura trend line + conferme)
2. Strategia "Rottura resistenza + momentum basso"
3. Strategia "Pullback trend EMA20"

Le 8 condizioni "grezze" precedenti (Score ≥75, Breakout ATR, Volume
Spike, pendenza Trendline, incrocio Stocastico, ADX forte, rottura
senza conferme) **non fanno più scattare l'alert da sole** — non erano
mai state testate individualmente con un backtest e generavano troppo
rumore. Restano comunque visibili nel corpo del messaggio come
contesto, quando arriva un alert per una delle 3 strategie vere.

Risultato pratico: meno alert, ma tutti riconducibili a qualcosa di
concretamente misurato.

## Strategia rimossa: Stocastico basso con conferma RSI

Aggiunta e poi rimossa su richiesta: sui primi test (dati sintetici)
dava Profit Factor sotto 1, ed essendo un segnale di ipervenduto
"puro" senza nessun contesto di trend, era coerente con quanto
osservato altrove nel progetto — le strategie che richiedono un trend
di fondo confermato tendono a reggere meglio di quelle puramente
contrarian. Il codice è stato tolto da `core.py` e da tutte le sezioni
dell'app.

## Supertrend come stop loss dinamico (trailing)

Nuova sezione in tab Backtest: **"🔀 Backtest con Stop Loss dinamico
(Supertrend)"**. Stessa logica di ingresso/take profit del backtest
SL/TP normale, ma lo stop loss non è più fisso a un multiplo di ATR
dall'ingresso: segue il Supertrend, un indicatore trend-following
basato sull'ATR che si stringe verso l'alto mano a mano che il trend
si sviluppa (o esce subito se il Supertrend "flippa" da rialzista a
ribassista). L'idea è lasciare correre i guadagni invece di uscire
sempre alla stessa distanza fissa dall'ingresso.

Funziona con qualunque delle tre strategie di solo acquisto, con
parametri configurabili (periodo e moltiplicatore del Supertrend,
take profit, giorni massimi in trade). Usa lo stesso riepilogo (R
medio, Profit Factor, Drawdown massimo, Perdite consecutive) del
backtest SL/TP normale, per un confronto diretto tra i due approcci
sulla stessa strategia/universo.

**Bugfix durante lo sviluppo**: la prima versione del calcolo
ricorsivo delle bande del Supertrend restava bloccata su valori NaN
per sempre una volta incontrato un NaN durante il riscaldamento
dell'ATR (le prime ~10-14 barre) — non si aggiornava mai più,
lasciando la direzione sempre "ribassista". Corretto con
un'inizializzazione esplicita ("bootstrap") delle bande quando il
valore precedente è ancora NaN.

Nota: Supertrend è usato **solo** come meccanismo di stop dinamico
nel backtest, non come segnale di ingresso a sé stante — l'ingresso
resta deciso dalle tre strategie esistenti.

## Turtle Trading (Donchian Channel Breakout)

Sistema storico reale (anni '80, Richard Dennis): compra la rottura
del massimo delle ultime N giornate (default 20). A differenza delle
altre strategie, la resistenza non è una trend line sui pivot ma il
puro canale di Donchian — il massimo assoluto degli N giorni PRIMA di
oggi (stesso principio del bugfix già applicato in precedenza al
breakout score: la resistenza esclude sempre il giorno corrente).

Versione semplificata rispetto all'originale: manca il filtro storico
"salta il trade se l'ultimo breakout era stato vincente" del sistema
Turtle reale.

**Non ancora collegata agli alert Telegram** (come le altre nuove
finché non testate): visibile e testabile in tab Score, colonna nello
Scanner V2, e in tutte e tre le sezioni di backtest (SL/TP normale,
Supertrend, confronto tra strategie).

## Momentum Ranking (fattore cross-sezionale)

Nuova sezione nel tab Scanner: **"🏆 Momentum Ranking"**. A differenza
delle strategie sopra (segnale sì/no sul singolo titolo), questo è un
**fattore**: classifica tutto l'universo scelto per rendimento
storico su un periodo (1/3/6/12 mesi) e mostra i migliori — il
fattore "momentum" documentato in letteratura accademica
(Jegadeesh-Titman) e usato su vasta scala da fondi sistematici, con
la differenza che qui manca il position sizing su volatilità e la
gestione del rischio che i fondi applicano in più: è solo il ranking
di base.

Non è un segnale di acquisto in sé, è uno strumento di selezione: i
titoli in cima hanno avuto il rendimento migliore nel periodo scelto,
scommettendo che il momentum recente tenda a continuare nel breve-medio
termine — funziona in media su portafogli diversificati, non è una
garanzia sul singolo titolo isolato.

## Momentum Ranking via Worker Cloudflare (Twelve Data)

Il Momentum Ranking (tab Scanner) ora supporta due fonti dati, scelte
da un menu a tendina:

1. **yfinance (locale)**: funziona subito, nessun setup, come prima.
2. **Twelve Data via Worker Cloudflare**: usa la metodologia
   "momentum 12-1" (Jegadeesh-Titman — salta l'ultimo mese prima di
   misurare il rendimento, per evitare il reversal a breve termine),
   con caching persistente lato Worker (Streamlit Cloud e GitHub
   Actions sono entrambi stateless: senza il Worker non avremmo modo
   di fare caching persistente tra un'esecuzione e l'altra in Python).

### Setup richiesto (una tantum, se si vuole usare la fonte Twelve Data)

1. Account gratuito su twelvedata.com (piano Basic Free: 800
   chiamate/giorno, 8/minuto — verificato ad agosto 2026)
2. Deploy del Worker Cloudflare fornito dall'utente (`momentum-ranking`
   Worker con KV namespace `MOMENTUM_CACHE` e secret
   `TWELVE_DATA_API_KEY`)
3. Nell'app: incollare l'URL del Worker nel campo dedicato, oppure
   impostarlo una volta per tutte come Secret Streamlit
   `MOMENTUM_WORKER_URL` (stesso meccanismo dei secrets Telegram)

⚠️ **Attenzione al formato dei ticker**: Twelve Data potrebbe usare
simboli diversi da yfinance per gli stessi titoli (es. senza il
suffisso `.MI` per le azioni italiane, o simboli diversi per gli
indici). Se compaiono molti errori "Storico insufficiente", controlla
la corrispondenza dei simboli sulla documentazione di Twelve Data —
non è stato possibile verificarlo direttamente in fase di sviluppo
(serve una API key reale).

Il primo calcolo della giornata può richiedere alcuni minuti (rate
limit del piano gratuito, gestito automaticamente dal Worker a gruppi
con pausa); le chiamate successive nello stesso giorno sono quasi
istantanee grazie alla cache KV a 24h.
