# forex-price-action-signal

Automatic **BUY / SELL / NO TRADE** signals for 1-hour forex (and gold) day trading, built only on
classical price action: swings, market structure, support/resistance zones, candlestick patterns and
how price behaves at those levels. **No lagging indicators.** ATR is used only to normalize sizes to
current volatility.

You never supply data. Every command downloads what is missing, caches it locally, and updates
incrementally on the next run.

> **Disclaimer.** This is an educational tool. Its signals are **not financial advice** and come with no
> guarantee of profitability. Backtests are hypothetical. Trading leveraged forex and gold can lose more
> than you expect. You are solely responsible for your trading decisions.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
cp .env.example .env                                    # optional: OANDA / Twelve Data keys
```

## Usage

```bash
python main.py --symbol EURUSD             # download/update data, print the latest signal + explanation
python main.py --all                       # scan every pair in instruments.yaml, table sorted by score
python main.py --symbol GBPUSD --backtest  # 2-year backtest + walk-forward, report saved to ./reports
python main.py --symbol EURUSD --plot      # chart saved to ./reports/charts
python main.py --watch                     # re-evaluate after every 1H candle close, print new signals
```

Options: `--years 2`, `--account-size 10000`, `--risk-percent 1`, `--source dukascopy|yfinance|oanda|twelvedata`
(force one source), `--refresh` (ignore the cache), `--bars 150` (chart width), `--config`, `--instruments`, `-v`.

Example output:

```
EURUSD  candle 2025-03-12 09:00 UTC (closed 10:00)   data: dukascopy
Signal: BUY   score +71   confluence 4 (1H structure, Daily trend, S/R location, candlestick)
Components: structure +0.92 | location +0.66 | candle +0.80 | behavior +0.62 | volume +0.30
Plan: BUY entry 1.08214  SL 1.08031 (beyond bullish engulfing + 0.25 ATR)  TP 1.08640 (next resistance zone 1.08640-1.08700)
      R:R 2.31   risk 18.3 pips   reward 42.3 pips
      position size 0.55 lots (risking 100.00 = 1% of 10,000)
BUY EURUSD: Daily uptrend, 1H HH/HL x4, bullish engulfing at support 1.0810-1.0820 (4 touches, PDL confluence), london session, no high-impact news within 2h, ...
```

(Illustrative format; real values depend on the data.)

---

## Data layer

| Order | Source | Notes |
|---|---|---|
| 1 | **Dukascopy** | Free, no key, long history. First tries the `dukascopy-python` package (1H BID candles). If that fails, it downloads raw daily 1-minute `.bi5` files from `datafeed.dukascopy.com` and resamples them to 1H. |
| 2 | **yfinance** | `EURUSD=X`, `GC=F` for gold (COMEX futures, which trade at a basis to spot). Only ~730 days of 1H history; requested in 120-day chunks. FX volume is always 0 here. |
| 3 | **OANDA v20** | Used only when `OANDA_API_KEY` is set (`OANDA_ENV=practice` or `live`). |
| 4 | **Twelve Data** | Used only when `TWELVEDATA_API_KEY` is set. Requests are spaced for the free-tier limit. |

The data layer:

- **Cache:** `data/cache/{SYMBOL}_1H.parquet` plus a `.meta.json` sidecar (sources used, requested range, validation summary).
- **Incremental updates:** only candles after the last cached timestamp are requested. If `--years` grows, only the missing head is fetched. A gap that falls entirely inside the weekend closure triggers no request.
- **Validation:** drops duplicates, sorts by time, drops candles that break `high >= max(open, close)` or `low <= min(open, close)` and flat weekend placeholders, flags spikes (`spike` column, range > 10x the rolling median), and counts missing open-market hours. **Missing data is never interpolated.**
- **Schema:** UTC `timestamp` (candle open), `open, high, low, close, volume`. All source quirks are handled inside the adapters.
- **Retries and logging:** exponential backoff on network errors, `tqdm` progress bars, and the log names the source actually used.
- **Day boundary:** the forex day ends at the **17:00 New York close** (DST-aware). Daily bars run 17:00 to 17:00 NY, 4H bars start at 17/21/01/05/09/13 NY, and weeks start Sunday 17:00 NY.
- **Economic calendar:** the ForexFactory weekly JSON feed is fetched at most every 6 hours and merged into `data/cache/calendar.parquet`, so history builds up over time. If the feed is down, the program warns and continues.

## How a signal is built

All analysis runs on data up to the **last closed candle**.

1. **Swings** (`analysis/swings.py`): fractal highs and lows (3 candles each side by default). A swing only *exists* once its right-hand candles have closed (`confirmed_idx`). Swings closer than `min_swing_atr x ATR` to the previous swing are treated as noise. Swings whose window spans a weekend or data gap are flagged `across_gap`.
2. **Market structure** (`analysis/structure.py`): HH+HL is an uptrend, LH+LL a downtrend, anything else a range. A close beyond the last swing is a **BOS** when it continues the trend and a **CHoCH** when it breaks against it. Trend strength (0-100) counts consecutive confirming swing labels.
3. **Higher timeframe** (`analysis/htf.py`): 1H is resampled to 4H and Daily using completed bars only. BUY is allowed only in a Daily uptrend and SELL only in a Daily downtrend, unless a recent **4H CHoCH** points the other way.
4. **Support/resistance zones** (`analysis/zones.py`): swings are clustered into zones whose width is set by ATR. Each zone is scored 0-100: 40% touches, 25% recency, 25% reaction strength (move after the touch, in ATR) and 10% tick volume at the touches. Zones then get a bonus when a key level sits inside them (`analysis/levels.py`): PDH/PDL/PDC, PWH/PWL, weekly open, Asian high/low, and round numbers. **Role reversal** is flagged when a zone built from highs now sits below price, or vice versa.
5. **Candlestick patterns** (`analysis/candles.py`): pin bar / hammer / shooting star, engulfing, inside bar, doji, morning/evening star, tweezer top/bottom. Patterns are defined by body/wick ratios and rejected when small relative to ATR. A pattern counts only **at a zone or swing point**. Doji and inside bars take their direction from location.
6. **Behavior** (`analysis/behavior.py`): bounce from a zone, pullback/retest of a recently broken level (continuation), fakeout / stop hunt (wick through, close back), and **failed bounce** (decisive close through the zone).
7. **Forex specifics** (`forex/`): sessions, per-pair entry windows (no entries in the first 15 minutes of a session or the last hour before rollover), Asian range and **London sweeps** of it, the news blackout (30 minutes before to 60 minutes after high-impact events for the pair's currencies, with gold mapped to USD), pip math, and the correlation guard.
8. **Tick volume** (`analysis/volume.py`): signal-candle volume / 20-bar average. Labelled *tick volume (weak confirmation)* and capped at 5% weight.

### Scoring

Each component is a number in [-1, +1]. The score is `100 x sum(weight x component)`:

| Component | Default weight |
|---|---|
| Market structure (1H trend, BOS/CHoCH, Daily alignment) | 30% |
| S/R location (at support vs at resistance, room to move) | 25% |
| Candlestick confirmation | 20% |
| Bounce / breakout behavior (+ London sweeps) | 20% |
| Tick volume | 5% |

**BUY** if score >= +60, **SELL** if score <= -60, and only if every filter passes:

- at least **3 independent factors** agree (1H structure, Daily trend, location, candle, behavior, volume);
- the entry is inside the pair's session window and outside a news blackout;
- the higher-timeframe filter allows the direction;
- a valid trade plan exists with **R:R >= 1.5**.

When a signal is rejected, the output says which filter rejected it.

### Trade plan

- **Entry:** the signal candle's close (market order at the next open).
- **Stop:** beyond the invalidation (pattern extreme or the bounced zone, else the last swing) plus a 0.25 ATR buffer. The stop is at least 0.5 ATR, and the trade is rejected if the stop exceeds 3 ATR.
- **Target:** the near edge of the next opposing zone (key levels count as zones).
- **R:R:** computed after the pair's spread and slippage.
- **Risk:** reported in pips. With `--account-size`, a position size in lots is computed. USD pip value is computed from live cached prices for crosses (e.g. USDJPY for EURJPY).

## Backtester

- **No lookahead:** for every bar *t* the engine receives `df.iloc[:t+1]` and nothing else. Tests prove this by changing all candles after a cut-off and checking that every earlier signal is identical, and by checking that each backtest signal equals a fresh live evaluation at that bar.
- **Fills:** entry at the next candle's open. Buys pay spread + slippage; shorts are closed at the ask. Stops get slippage, and a gap through the stop fills at the open. **If one candle touches both stop and target, the stop is assumed first.** A 72-bar time stop applies. One position per pair at a time.
- **Report** (`reports/{SYMBOL}_backtest_{time}/`): `report.md`, `trades.csv`, `equity.png`, `summary.json`. It covers win rate, profit factor, expectancy (R), max drawdown (R and %), average planned R:R and trade count, overall, per session and per side, plus an equity curve.
- **Walk-forward:** signals are generated once. The decision threshold and minimum R:R are optimized on the first 60% of the data, then validated on the last 40% (alongside the untouched defaults). A warning is printed when out-of-sample expectancy collapses (<= 0, or below 50% of in-sample), which suggests overfitting.
- **Caveat:** the free calendar feed has no history, so news filtering in a backtest only covers the period your local calendar cache has seen. The report says so.

## Tuning parameters

Everything lives in `config.yaml` (strategy) and `instruments.yaml` (pairs, pip size, costs, sessions,
round-number steps, symbols per source).

| Want | Change |
|---|---|
| More / fewer signals | `scoring.buy_threshold` / `sell_threshold`, `scoring.min_confluence` |
| Trade Daily ranges too | `htf.allow_daily_range: true` (the spec default is `false`) |
| Bigger / smaller swings | `swings.fractal_window`, `swings.min_swing_atr` |
| Wider / tighter zones | `zones.cluster_atr`, `zones.min_width_atr`, `zones.max_width_atr` |
| Stricter candles | `candles.min_range_atr`, `pin_wick_body_ratio`, `engulf_min_body_atr` |
| Risk model | `trade.min_rr`, `sl_atr_buffer`, `max_sl_atr`, `backtest.max_hold_bars` |
| Pair costs / sessions | `instruments.yaml` → `spread_pips`, `slippage_pips`, `sessions` |
| Factor weights | `scoring.weights` (volume should stay <= 0.05) |

Use the walk-forward report rather than in-sample results to judge changes. A parameter set that only works in-sample is curve-fitted.

## Mobile web app: analyze a chart photo

`webapp/` is a phone-friendly PWA. Tap **Analyze Chart**, photograph a chart with the rear camera,
and get **BUY / SELL / HOLD** with a confidence level and a short explanation.

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env      # server-side only
python -m webapp.server --port 8000              # then open http://<computer-ip>:8000 on the phone
```

### How the photo is analyzed (and what it costs)

A photo contains no price data, so the trading engine above can't run on it. Instead, the photo
goes to an **image-understanding model: Claude (`claude-opus-5-5`) through the Anthropic API**,
called from the small Python backend (`webapp/server.py`, standard library HTTP server).

- **The API key stays on the server**, in `.env` or an environment variable. The browser only talks to `/api/analyze` and never sees the key.
- **Same rules as the engine:** the model gets the classical price action rules from the engine (structure, S/R zones, candlestick confirmation at a zone, behavior, three-factor confluence, ignore indicators). It must answer in a fixed JSON schema (structured outputs).
- **Hard rules in code** (`webapp/vision.py`), not left to the model:
  - **Unreadable photo:** if the model reports the chart is unreadable (not a chart, blur, glare, cropped, too few candles), the app shows an **error and no signal**.
  - **Weak trade calls:** BUY/SELL below 60% model confidence becomes **HOLD**, and the app says so.
  - **Any failure:** refusal, truncated or malformed output, network or API failure, or a missing API key all show an **error, never a signal**.
  - **No made-up results:** there is no random, hardcoded or fallback signal anywhere in the code.
- **Confidence** is the model's own estimate of how clearly the visible price action supports the call. It is **not** a calibrated probability of profit, and the app says this under the meter.
- **Safety fallback:** if a safety classifier declines a request, Anthropic's server-side fallback (`fallbacks: "default"`) retries it on its recommended model. The app reports which model answered. If you'd rather a decline simply show an error, remove `fallbacks` and the beta flag in `vision.py`.

**Cost:** you pay the Anthropic API per request; there's no subscription.

- **Rates:** Claude Opus 5.5 costs $4 per million input tokens and $20 per million output tokens.
- **Input:** the app shrinks each photo to at most 1568 px on the long side. A typical phone photo is then about 2,500 image tokens; with the instructions, input is about 3,500 tokens (≈ $0.014).
- **Output:** the JSON answer plus the model's reasoning at `high` effort is usually 1,500–4,000 tokens (≈ $0.03–0.08).
- **Total:** expect **roughly $0.04–0.10 per photo**. The exact cost of each analysis comes back in the response, is shown under "What the model saw", and is logged by the server.
- **Cheaper:** `VISION_EFFORT=medium`, or `VISION_MODEL=claude-sonnet-5-5` ($2/$10 per million). Both may read charts less carefully.

### Phone setup and PWA install

- **Camera:** the camera opens through `<input type="file" accept="image/*" capture="environment">`. It works over plain HTTP on your local network.
- **Home-screen install:** installing the PWA and running the service worker need **HTTPS** (browsers only allow `localhost` without it).
  - Put the server behind any HTTPS host or reverse proxy, or a tunnel service.
  - Or set `TLS_CERT` / `TLS_KEY` to a certificate the phone trusts.
  - Then use "Add to Home Screen" (Safari share menu / Chrome menu).
- **Internet-facing server:** set `APP_ACCESS_TOKEN` (the app asks for the code once and remembers it). `RATE_LIMIT_PER_HOUR` (default 30 per IP) stops strangers from spending your API credit.
- **Offline:** the app shell works offline, but analysis needs the network. Analyses are never cached.

## Tests

```bash
python -m pytest -q
```

The suite covers swing detection, structure classification, BOS/CHoCH, every candlestick pattern on hand-made OHLC fixtures, zones and key levels, behavior, sessions, the news blackout, the correlation guard, pip math, the data cache and incremental updates (network mocked), the source fallback chain, validation, NY-close resampling, the Dukascopy `.bi5` parser, backtest fill rules, walk-forward, and the **no-lookahead guarantee**.

## Project layout

```
main.py                 CLI
config.yaml             strategy + data settings
instruments.yaml        pairs, pip math, costs, sessions, source symbols
fxpa/
  config.py             config / instrument loading
  app.py                signal, scan, backtest, plot, watch orchestration
  data/                 sources/ (dukascopy, yfinance, oanda, twelvedata), cache, validate, resample, calendar, manager
  analysis/             atr, swings, structure, htf, levels, zones, candles, behavior, volume
  forex/                sessions, news, correlation, pips
  signal/               engine (scoring, filters, explanation), plan
  backtest/             engine (replay + simulation + metrics), walkforward, report
  viz/plot.py           matplotlib chart
webapp/
  server.py             backend: static PWA + POST /api/analyze (API key stays here)
  vision.py             Claude vision call, JSON schema, no-signal-on-failure rules
  static/               index.html, styles.css, app.js, sw.js, manifest.json, icons/
tests/
```
