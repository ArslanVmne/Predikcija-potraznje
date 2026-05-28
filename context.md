# Projekat: Predikcija potražnje i optimizacija zaliha

## Tema
"Koliko da naručimo?" — Inteligentni forecasting sistem za maloprodajni lanac.

Dataset: **Favorita / Store Sales – Time Series Forecasting** (Kaggle)

## Rok i isporuka
**Rok: 15. jun 2026.**

Predaje se:
- GitHub repozitorijum (čist kod, `requirements.txt` ili `environment.yml`, detaljan `README.md`)
- Deploy-ovana web aplikacija (Streamlit / Gradio / Dash / Flask + Render / Railway / Hugging Face Spaces ili lokalno)
- Prezentacija (PowerPoint ili PDF, max 10–12 slajdova)

## Bodovanje (ukupno 30 bodova)

| Kategorija | Bodovi |
|---|---|
| Tehnička implementacija i funkcionalnost (kompletan pipeline, čist i reprodukibilan kod) | 10 |
| Kvalitet ML rješenja i inovativnost (modeli, feature engineering, hibridni pristup, evaluacija, baseline poboljšanje) | 8 |
| Web aplikacija / Dashboard i korisničko iskustvo (interaktivnost, vizualizacije, What-if simulacije) | 6 |
| Prezentacija i odbrana (live demo, odgovori na pitanja) | 6 |

## Funkcionalni zahtjevi sistema

### ML pipeline mora:
- Predviđati **dnevnu/nedjeljnu potražnju sa confidence intervalima**
- Računati **optimalnu količinu narudžbe** (EOQ + safety stock) uz lead time, troškove i service level
- **Detektovati anomalije** (neočekivani skokovi potražnje)
- Davati **SHAP objašnjenja** ("Potražnja raste zbog promocije + vikenda + toplog vremena")

### Hibridni model (traženo):
- **Prophet** — trend i sezonalnost
- **XGBoost / LightGBM** — tabularne features
- **LSTM / Transformer** — sezonalnost i dugotrajne zavisnosti

### Web aplikacija mora imati:
- Upload prodajnih podataka + eksternih faktora (praznici, promocije)
- Interaktivni dashboard sa forecast-om po proizvodu/skladištu
- **What-if simulacije** ("šta ako podignemo cijenu ili pokrenemo kampanju")
- Generisanje automatskih narudžbenica

## Dataset — Store Sales Time Series Forecasting

Fajlovi u `data/raw/`:

| Fajl | Opis |
|---|---|
| `train.csv` | Glavna tabela — dnevna prodaja po prodavnici i porodici proizvoda (2013–2017) |
| `test.csv` | Test skup za predikciju (15 dana nakon kraja traina) |
| `stores.csv` | Metadata prodavnica (grad, tip, cluster) |
| `transactions.csv` | Ukupan broj transakcija po prodavnici i danu |
| `oil.csv` | Dnevna cijena nafte (Ekvador — ekonomija osjetljiva na naftu) |
| `holidays_events.csv` | Ekvadorski praznici i eventi sa tipom (local/regional/national) i flagovima (transferred, bridge) |

Ključne kolone `train.csv`:
- `date` — datum
- `store_nbr` — broj prodavnice (1–54)
- `family` — porodica proizvoda (33 kategorije: PRODUCE, BEVERAGES, BREAD/BAKERY, ...)
- `sales` — target (dnevna prodaja, može biti 0)
- `onpromotion` — broj stavki na promociji tog dana

## Prioriteti za bodove

1. **Pipeline mora biti kompletan i reprodukibilan** — najvažnija kategorija (10 bod.)
2. **Hibridni model + SHAP** — direktno traženo, nosi 8 bod.
3. **Web app sa What-if simulacijama** — 6 bod., mora imati live demo
4. Inovacije (anomaly detection, EOQ kalkulator, automatske narudžbenice) — posebno bodovano

## Napomene
- Originalni rad — plagijat = 0 bodova
- Manja poboljšanja i proširenja se posebno boduju
- Live demo na odbrani je obavezan
