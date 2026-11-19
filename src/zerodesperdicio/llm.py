"""Cliente do LLM (Claude, SDK oficial `anthropic`). Opcional: sem credencial, o app usa regras.

O LLM só redige a explicação. Ranking, validade e restrições são código determinístico.
"""
from __future__ import annotations

import os
from typing import Protocol

import anthropic
from dotenv import load_dotenv

from . import config

DEFAULT_MODEL = "claude-opus-5-5"
# modelos que aceitam o fallback do lado do servidor (beta server-side-fallback-2026-07-01)
FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})


class LLMError(Exception):
    """Falha do LLM com mensagem em português pronta para a tela."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class ClaudeLLM:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        self.client = client or anthropic.Anthropic(timeout=90.0, max_retries=2)
        self.model = model or os.environ.get("ZD_LLM_MODEL", DEFAULT_MODEL)
        self.name = self.model

    def complete(self, system: str, user: str) -> str:
        kwargs = dict(model=self.model, max_tokens=16000, system=system,
                      messages=[{"role": "user", "content": user}],
                      output_config={"effort": "low"})
        try:
            if self.model in FALLBACK_MODELS:
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("A chave da API foi recusada (401). Confira ANTHROPIC_API_KEY no arquivo .env.") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError("A chave da API não tem permissão para este modelo (403).") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Modelo não encontrado: {self.model} (404).") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Atingimos o limite de uso da API (429). Tente de novo em instantes.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"A API respondeu com erro {e.status_code}.") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Sem conexão com a API do LLM (verifique a rede).") from e
        if resp.stop_reason == "refusal":
            raise LLMError("O modelo recusou o pedido.")
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise LLMError("O modelo não devolveu texto.")
        return text


def get_llm() -> LLMClient | None:
    """Devolve o cliente se houver credencial (variável de ambiente, .env ou perfil `ant`)."""
    load_dotenv(config.root() / ".env", override=False)
    try:
        client = anthropic.Anthropic(timeout=90.0, max_retries=2)
    except Exception:
        return None
    if client.api_key or getattr(client, "auth_token", None) or getattr(client, "credentials", None):
        return ClaudeLLM(client=client)
    return None
