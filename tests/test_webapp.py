"""Chart-photo web app: result rules, failure handling, and the HTTP API (Anthropic client mocked)."""

from __future__ import annotations

import base64
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from webapp import server as srv
from webapp.vision import AnalysisError, ChartResult, analyze_chart, interpret

GOOD = {
    "chart_readable": True, "unreadable_reason": "", "instrument": "EURUSD", "timeframe": "1H",
    "trend": "uptrend", "observations": ["higher highs and higher lows", "bullish engulfing at 1.0810"],
    "key_levels": ["1.0810 support", "1.0880 resistance"], "signal": "BUY", "confidence": 78,
    "explanation": "Price is in an uptrend and just printed a bullish engulfing candle at support.",
}
JPEG = b"\xff\xd8\xff\xe0" + b"0" * 200


class FakeClient:
    """Stands in for anthropic.Anthropic(); records the request and returns a canned response."""

    def __init__(self, data=None, stop_reason="end_turn", text=None, exc=None):
        self.calls = []
        self.exc = exc
        payload = text if text is not None else json.dumps(data or GOOD)
        self.response = SimpleNamespace(
            model="claude-opus-5-5", stop_reason=stop_reason,
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=payload)],
            usage=SimpleNamespace(input_tokens=3000, output_tokens=2000, cache_creation_input_tokens=0),
        )
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


# ---------------------------------------------------------------- interpretation rules
def test_readable_chart_returns_signal():
    r = interpret(GOOD)
    assert r.signal == "BUY" and r.confidence == 78 and r.note == ""


def test_unreadable_chart_raises_instead_of_signal():
    with pytest.raises(AnalysisError) as e:
        interpret({**GOOD, "chart_readable": False, "unreadable_reason": "strong glare over the candles"})
    assert e.value.code == "unreadable" and "glare" in e.value.message


def test_low_confidence_trade_becomes_hold_with_note():
    r = interpret({**GOOD, "confidence": 40}, min_confidence=60)
    assert r.signal == "HOLD" and r.model_signal == "BUY" and "40%" in r.note


@pytest.mark.parametrize("bad", [{"signal": "MAYBE"}, {"confidence": "high"}, {"explanation": ""}])
def test_incomplete_output_raises(bad):
    with pytest.raises(AnalysisError):
        interpret({**GOOD, **bad})


# ---------------------------------------------------------------- API call handling
def test_request_shape_keeps_key_server_side_and_uses_structured_output():
    client = FakeClient()
    r = analyze_chart(JPEG, "image/jpeg", client=client)
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    img = call["messages"][0]["content"][0]
    assert img["type"] == "image" and base64.b64decode(img["source"]["data"]) == JPEG
    assert r.cost_usd == pytest.approx(3000 * 4 / 1e6 + 2000 * 20 / 1e6)


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_or_truncation_is_an_error(stop):
    with pytest.raises(AnalysisError):
        analyze_chart(JPEG, "image/jpeg", client=FakeClient(stop_reason=stop))


def test_malformed_json_is_an_error():
    with pytest.raises(AnalysisError) as e:
        analyze_chart(JPEG, "image/jpeg", client=FakeClient(text="BUY!!"))
    assert e.value.code == "bad_output"


def test_api_connection_failure_is_an_error():
    import anthropic
    import httpx2
    exc = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    with pytest.raises(AnalysisError) as e:
        analyze_chart(JPEG, "image/jpeg", client=FakeClient(exc=exc))
    assert e.value.code == "network"


def test_rejects_bad_media_and_empty_image():
    with pytest.raises(AnalysisError):
        analyze_chart(JPEG, "application/pdf", client=FakeClient())
    with pytest.raises(AnalysisError):
        analyze_chart(b"", "image/jpeg", client=FakeClient())


# ---------------------------------------------------------------- HTTP server
@pytest.fixture
def live_server(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(srv.Handler, "access_token", "")
    monkeypatch.setattr(srv.Handler, "limiter", srv.RateLimiter(100))
    httpd = srv.make_server("127.0.0.1", 0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(base, body, headers=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(base + "/api/analyze", data=data, method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _img():
    return {"image": base64.b64encode(JPEG).decode(), "media_type": "image/jpeg"}


def test_api_success(live_server, monkeypatch):
    monkeypatch.setattr(srv.Handler, "analyzer", staticmethod(lambda img, mt: interpret(GOOD)))
    status, body = _post(live_server, _img())
    assert status == 200 and body["ok"] and body["result"]["signal"] == "BUY"


def test_api_failure_never_contains_signal(live_server, monkeypatch):
    def boom(img, mt):
        raise AnalysisError("Couldn't read this chart: blurry.", code="unreadable")
    monkeypatch.setattr(srv.Handler, "analyzer", staticmethod(boom))
    status, body = _post(live_server, _img())
    assert status == 422 and body["ok"] is False and "result" not in body


def test_api_unexpected_exception_is_generic_error(live_server, monkeypatch):
    monkeypatch.setattr(srv.Handler, "analyzer", staticmethod(lambda img, mt: 1 / 0))
    status, body = _post(live_server, _img())
    assert status == 500 and body["ok"] is False and "result" not in body


def test_api_without_key_returns_503(live_server, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    status, body = _post(live_server, _img())
    assert status == 503 and body["code"] == "config"


def test_api_bad_payload(live_server):
    assert _post(live_server, {"image": "not base64!!"})[0] == 400


def test_access_token_and_rate_limit(live_server, monkeypatch):
    monkeypatch.setattr(srv.Handler, "analyzer", staticmethod(lambda img, mt: interpret(GOOD)))
    monkeypatch.setattr(srv.Handler, "access_token", "s3cret")
    assert _post(live_server, _img())[0] == 401
    assert _post(live_server, _img(), {"X-Access-Token": "s3cret"})[0] == 200
    monkeypatch.setattr(srv.Handler, "limiter", srv.RateLimiter(1))
    assert _post(live_server, _img(), {"X-Access-Token": "s3cret"})[0] == 200
    assert _post(live_server, _img(), {"X-Access-Token": "s3cret"})[0] == 429


def test_static_files_and_traversal(live_server):
    for path, ctype in (("/", "text/html"), ("/manifest.json", "json"), ("/sw.js", "javascript"),
                        ("/icons/icon-192.png", "image/png")):
        with urllib.request.urlopen(live_server + path) as r:
            assert r.status == 200 and ctype in r.headers["Content-Type"]
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(live_server + "/../server.py")
    assert e.value.code == 404


def test_result_dataclass_serializes():
    d = ChartResult("HOLD", 0, "x", "unknown", "unknown", "unclear").to_dict()
    assert d["signal"] == "HOLD" and "cost_usd" in d
