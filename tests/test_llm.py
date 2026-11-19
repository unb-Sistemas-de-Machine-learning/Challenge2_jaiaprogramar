import json

import anthropic
import httpx2
import pytest
from anthropic import DefaultHttpxClient

from zerodesperdicio import llm as llm_mod
from zerodesperdicio.llm import ClaudeLLM, LLMError, get_llm


def make_client(handler):
    return anthropic.Anthropic(api_key="sk-test", max_retries=0,
                               http_client=DefaultHttpxClient(transport=httpx2.MockTransport(handler)))


def ok_response(content, stop_reason="end_turn"):
    return httpx2.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5}})


def capture():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return ok_response([{"type": "thinking", "thinking": "", "signature": "s"},
                            {"type": "text", "text": "Faça a omelete [R1]."}])
    return seen, handler


def test_default_request_shape():
    seen, handler = capture()
    out = ClaudeLLM(client=make_client(handler)).complete("SISTEMA", "PERGUNTA")
    assert out == "Faça a omelete [R1]."
    b = seen["body"]
    assert b["model"] == "claude-opus-5-5"
    assert b["max_tokens"] == 16000
    assert b["fallbacks"] == "default"
    assert b["output_config"] == {"effort": "low"}
    assert b["system"] == "SISTEMA"
    assert b["messages"][0]["content"] == "PERGUNTA"
    assert "server-side-fallback-2026-07-01" in seen["beta"]


def test_model_without_fallback():
    seen, handler = capture()
    ClaudeLLM(client=make_client(handler), model="claude-haiku-5-5").complete("s", "u")
    assert "fallbacks" not in seen["body"]
    assert "server-side-fallback" not in seen["beta"]


def test_refusal_raises():
    client = make_client(lambda r: ok_response([], stop_reason="refusal"))
    with pytest.raises(LLMError):
        ClaudeLLM(client=client).complete("s", "u")


def test_no_text_raises():
    client = make_client(lambda r: ok_response([{"type": "thinking", "thinking": "", "signature": "s"}]))
    with pytest.raises(LLMError):
        ClaudeLLM(client=client).complete("s", "u")


@pytest.mark.parametrize("code,etype,word", [
    (401, "authentication_error", "chave"),
    (429, "rate_limit_error", "limite"),
])
def test_http_errors(code, etype, word):
    client = make_client(lambda r: httpx2.Response(code, json={"type": "error", "error": {"type": etype, "message": "x"}}))
    with pytest.raises(LLMError) as e:
        ClaudeLLM(client=client).complete("s", "u")
    assert word in str(e.value).lower()


def test_connection_error():
    def boom(request):
        raise httpx2.ConnectError("sem rede")
    with pytest.raises(LLMError) as e:
        ClaudeLLM(client=make_client(boom)).complete("s", "u")
    assert "rede" in str(e.value).lower()


def test_get_llm_without_credentials(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    assert get_llm() is None


def test_get_llm_with_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ZD_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    assert isinstance(get_llm(), ClaudeLLM)
