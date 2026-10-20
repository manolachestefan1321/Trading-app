"""Chart-photo analysis with Claude's vision model.

The photo is sent to the Anthropic Messages API together with the same classical price action
rules the data-driven engine uses (structure, S/R, candlestick confirmation, confluence). The
model must return a fixed JSON shape; this module then applies hard rules on top:

- unreadable photo (not a chart, blurry, glare, cropped, no price axis) -> AnalysisError, no signal
- BUY/SELL below the confidence floor -> HOLD, and the response says so
- refusal, truncation, malformed output, API failure -> AnalysisError, no signal

Nothing here ever produces a signal without a successful model response.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field

log = logging.getLogger("fxpa.webapp")

MODEL = os.environ.get("VISION_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("VISION_EFFORT", "high")          # low | medium | high | xhigh | max
MIN_CONFIDENCE = int(os.environ.get("MIN_CONFIDENCE", "60"))
MAX_IMAGE_BYTES = 5 * 1024 * 1024                          # API limit per image is 5 MB
ALLOWED_MEDIA = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# USD per million tokens, used only to report the cost of each request in the response/logs.
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
          # possible server-side fallback models
          "claude-opus-5": (5.0, 25.0), "claude-opus-4-8": (5.0, 25.0)}

SYSTEM_PROMPT = """You analyze photos of price charts (usually forex or gold candlestick charts \
photographed from a screen with a phone) using ONLY classical price action.

Method, in order:
1. Readability. Decide whether the photo shows a price chart you can actually read: individual \
candles distinguishable, the most recent candles visible on the right, and enough history \
(roughly 30+ candles) to judge structure. Glare, blur, heavy cropping, a non-chart image, or a \
chart too small to read means chart_readable=false. Do not guess from a partial view.
2. Market structure: higher highs + higher lows = uptrend, lower highs + lower lows = downtrend, \
otherwise range. Note any break of structure or change of character near the right edge.
3. Support/resistance: horizontal zones where price reacted several times; whether the latest \
price is at, between, or breaking through a zone; role reversal of broken levels.
4. Candlestick confirmation on the latest candles: pin bar/hammer/shooting star, engulfing, \
inside bar, doji, morning/evening star, tweezers - only meaningful at a zone or swing point.
5. Behavior: bounce with rejection, pullback to a broken level, fakeout, or decisive break.

Rules:
- Ignore any indicators drawn on the chart (moving averages, RSI, MACD, etc.). Do not base the \
signal on them.
- Say BUY or SELL only when at least three independent factors agree (structure, location, \
candle confirmation, behavior) and there is room to the next opposing zone. Otherwise HOLD.
- confidence (0-100) is how strongly the visible price action supports your call, not a \
probability of profit. Lower it for anything you could not see clearly.
- You cannot see news, spreads, or the time of day unless shown; mention this only if it matters.
- If you can read the symbol or timeframe on the chart, report them; otherwise use "unknown".
- explanation: 2-3 plain sentences a non-expert understands, citing the visible evidence.
- observations: up to 6 short factual notes (e.g. "two lower highs since the 1.0850 peak").
- If chart_readable is false, set signal to HOLD, confidence to 0, and explain what is wrong \
with the photo in unreadable_reason so the user can retake it."""

USER_PROMPT = "Analyze this chart photo and return the JSON result."

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "chart_readable": {"type": "boolean"},
        "unreadable_reason": {"type": "string"},
        "instrument": {"type": "string"},
        "timeframe": {"type": "string"},
        "trend": {"type": "string", "enum": ["uptrend", "downtrend", "range", "unclear"]},
        "observations": {"type": "array", "items": {"type": "string"}},
        "key_levels": {"type": "array", "items": {"type": "string"}},
        "signal": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["chart_readable", "unreadable_reason", "instrument", "timeframe", "trend",
                 "observations", "key_levels", "signal", "confidence", "explanation"],
    "additionalProperties": False,
}


class AnalysisError(Exception):
    """Analysis failed; `message` is safe to show to the user. No signal must be shown."""

    def __init__(self, message: str, status: int = 422, code: str = "analysis_failed"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


@dataclass
class ChartResult:
    signal: str                      # BUY | SELL | HOLD
    confidence: int                  # 0-100, the model's own estimate (not calibrated odds)
    explanation: str
    instrument: str
    timeframe: str
    trend: str
    observations: list[str] = field(default_factory=list)
    key_levels: list[str] = field(default_factory=list)
    model_signal: str = ""           # what the model said before the confidence floor
    note: str = ""
    model: str = MODEL
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _cost(model: str, usage) -> float | None:
    price = PRICES.get(model)
    if not price or usage is None:
        return None
    inp = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
    out = getattr(usage, "output_tokens", 0) or 0
    return round(inp * price[0] / 1e6 + out * price[1] / 1e6, 4)


def interpret(data: dict, min_confidence: int = MIN_CONFIDENCE) -> ChartResult:
    """Apply the hard rules to the model's JSON. Raises AnalysisError when no signal may be shown."""
    if not data.get("chart_readable"):
        reason = (data.get("unreadable_reason") or "").strip() or "the chart could not be read"
        raise AnalysisError(f"Couldn't read this chart: {reason}. Try again with the whole chart in "
                            "frame, less glare, and the phone held steady.", code="unreadable")
    signal = data.get("signal")
    conf = data.get("confidence")
    explanation = (data.get("explanation") or "").strip()
    if signal not in ("BUY", "SELL", "HOLD") or not isinstance(conf, int) or not explanation:
        raise AnalysisError("The analysis came back incomplete, so no signal is shown. Please try again.",
                            status=502, code="bad_output")
    conf = max(0, min(100, conf))
    result = ChartResult(
        signal=signal, confidence=conf, explanation=explanation,
        instrument=(data.get("instrument") or "unknown").strip() or "unknown",
        timeframe=(data.get("timeframe") or "unknown").strip() or "unknown",
        trend=data.get("trend") or "unclear",
        observations=[o for o in data.get("observations", []) if isinstance(o, str)][:6],
        key_levels=[k for k in data.get("key_levels", []) if isinstance(k, str)][:6],
        model_signal=signal,
    )
    if signal in ("BUY", "SELL") and conf < min_confidence:
        result.signal = "HOLD"
        result.note = (f"The chart leaned {signal} at {conf}% confidence, below the {min_confidence}% "
                       "needed to call a trade, so the result is HOLD.")
    return result


def analyze_chart(image: bytes, media_type: str, client=None) -> ChartResult:
    """Send one chart photo to Claude and return a validated result (or raise AnalysisError)."""
    import base64

    import anthropic

    if media_type not in ALLOWED_MEDIA:
        raise AnalysisError(f"Unsupported image type {media_type}. Use a JPEG or PNG photo.", status=415,
                            code="bad_image")
    if not image:
        raise AnalysisError("No image received.", status=400, code="bad_image")
    if len(image) > MAX_IMAGE_BYTES:
        raise AnalysisError("The image is larger than 5 MB. Please retake or resize it.", status=413,
                            code="bad_image")
    if client is None:
        client = anthropic.Anthropic(max_retries=2, timeout=120.0)

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                     # retry a safety-classifier decline on the recommended model
            thinking={"type": "adaptive"},
            output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": RESULT_SCHEMA}},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": base64.standard_b64encode(image).decode("ascii")}},
                {"type": "text", "text": USER_PROMPT},
            ]}],
        )
    except anthropic.AuthenticationError:
        log.error("Anthropic API rejected the key (check ANTHROPIC_API_KEY)")
        raise AnalysisError("The analysis service is not configured correctly.", status=503, code="config")
    except anthropic.BadRequestError as exc:
        log.error("bad request: %s", exc.message)
        raise AnalysisError("The image could not be processed. Try a different photo.", status=422,
                            code="bad_image")
    except anthropic.RateLimitError:
        raise AnalysisError("The analysis service is busy. Wait a minute and try again.", status=429,
                            code="busy")
    except anthropic.APIStatusError as exc:
        log.error("API error %s: %s", exc.status_code, exc.message)
        raise AnalysisError("The analysis service failed. Please try again.", status=502, code="upstream")
    except anthropic.APIConnectionError:
        raise AnalysisError("Couldn't reach the analysis service. Check the connection and retry.",
                            status=503, code="network")

    cost = _cost(getattr(response, "model", MODEL), getattr(response, "usage", None))
    log.info("analysis done: model=%s stop=%s cost=$%s", getattr(response, "model", MODEL),
             response.stop_reason, cost)
    if response.stop_reason == "refusal":
        raise AnalysisError("The model declined to analyze this image, so no signal is shown.", code="refusal")
    if response.stop_reason == "max_tokens":
        raise AnalysisError("The analysis was cut off before finishing, so no signal is shown. Please retry.",
                            status=502, code="truncated")
    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
    if not text:
        raise AnalysisError("The analysis returned no result, so no signal is shown.", status=502,
                            code="bad_output")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise AnalysisError("The analysis returned an unreadable result, so no signal is shown.", status=502,
                            code="bad_output")
    result = interpret(data)
    result.model = getattr(response, "model", MODEL)
    result.cost_usd = cost
    return result
