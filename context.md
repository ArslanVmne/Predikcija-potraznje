# Projekat: Predikcija potraznje i optimizacija zaliha

## Tema

"Koliko da narucimo?" - Inteligentni forecasting sistem za maloprodajni lanac.

Dataset: Favorita / Store Sales - Time Series Forecasting (Kaggle)

## Rok i isporuka

Rok: 15. jun 2026.

Predaje se:
- GitHub repozitorijum (cist kod, `requirements.txt` ili `environment.yml`, detaljan `README.md`)
- Deploy-ovana web aplikacija (Streamlit / Gradio / Dash / Flask + Render / Railway / Hugging Face Spaces ili lokalno)
- Prezentacija (PowerPoint ili PDF, max 10-12 slajdova)

## Bodovanje (ukupno 30 bodova)

| Kategorija | Bodovi |
|---|---|
| Tehnicka implementacija i funkcionalnost (kompletan pipeline, cist i reprodukibilan kod) | 10 |
| Kvalitet ML rjesenja i inovativnost (modeli, feature engineering, hibridni pristup, evaluacija, baseline poboljsanje) | 8 |
| Web aplikacija / Dashboard i korisnicko iskustvo (interaktivnost, vizualizacije, What-if simulacije) | 6 |
| Prezentacija i odbrana (live demo, odgovori na pitanja) | 6 |

## Funkcionalni zahtjevi sistema

ML pipeline mora:
- predvidjati dnevnu/nedjeljnu potraznju sa confidence intervalima
- racunati optimalnu kolicinu naruzbe (EOQ + safety stock) uz lead time, troskove i service level
- detektovati anomalije (neocekivani skokovi potraznje)
- davati SHAP objasnjenja ("Potraznja raste zbog promocije + vikenda + toplog vremena")

Hibridni model (direktno trazen):
- Prophet za trend i sezonalnost
- XGBoost / LightGBM za tabularne features
- LSTM / Transformer za dugorocne zavisnosti

Web aplikacija mora imati:
- upload prodajnih podataka i eksternih faktora (praznici, promocije)
- interaktivni dashboard sa forecast-om po proizvodu/skladistu
- What-if simulacije ("sta ako podignemo cijenu ili pokrenemo kampanju")
- generisanje automatskih narudžbenica

## Dataset - Store Sales Time Series Forecasting

Fajlovi u `data/raw/`:

| Fajl | Opis |
|---|---|
| `train.csv` | Dnevna prodaja po prodavnici i porodici proizvoda (2013-2017) |
| `test.csv` | Test skup za predikciju (15 dana nakon kraja traina) |
| `stores.csv` | Metadata prodavnica (grad, tip, cluster) |
| `transactions.csv` | Ukupan broj transakcija po prodavnici i danu |
| `oil.csv` | Dnevna cijena nafte (Ekvador ima ekonomiju osjetljivu na naftu) |
| `holidays_events.csv` | Ekvadorski praznici i eventi sa tipom (local/regional/national) i flagovima (transferred, bridge) |

Kljucne kolone `train.csv`:
- `date` - datum
- `store_nbr` - broj prodavnice (1-54)
- `family` - porodica proizvoda (33 kategorije: PRODUCE, BEVERAGES, BREAD/BAKERY, ...)
- `sales` - target (dnevna prodaja, moze biti 0)
- `onpromotion` - broj stavki na promociji tog dana

## Prioriteti za bodove

1. Pipeline mora biti kompletan i reprodukibilan - najvaznija kategorija (10 bod.)
2. Hibridni model + SHAP - direktno trazeno, nosi 8 bod.
3. Web app sa What-if simulacijama - 6 bod., mora imati live demo
4. Inovacije (anomaly detection, EOQ kalkulator, automatske narudžbenice) - posebno bodovano

## Napomene

- Originalni rad - plagijat = 0 bodova
- Manja poboljsanja i prosirenja se posebno boduju
- Live demo na odbrani je obavezan
