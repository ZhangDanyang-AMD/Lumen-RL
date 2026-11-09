"""OpenAI-compatible Chat Completions policy (DSV4.1 served via ATOM/vLLM).

Model calls have no external effect, so transport failures are retried; the
environment side never retries.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Mapping, Sequence

import httpx

from lumenrl.agentic.core.protocols import ToolCall
from lumenrl.agentic.core.trajectory import PolicyTurn
from lumenrl.agentic.rollout.policy import AgentPolicy, PolicyError
from lumenrl.agentic.rollout.token_alignment import extract_choice_tokens, rollout_token_status

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class OpenAIChatPolicy(AgentPolicy):
    """Calls ``POST {base_url}/chat/completions`` with native tool calling."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None,
        sampling: Mapping[str, Any],
        timeout_s: float,
        max_retries: int,
        verify_tls: bool,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        """Args:
            base_url: Endpoint root ending in ``/v1``.
            model: Model name sent in the request.
            api_key_env: Environment variable holding the bearer token; ``None`` for unauthenticated local servers.
            sampling: Request fields such as ``max_tokens``/``temperature``.
            timeout_s: Per-request timeout.
            max_retries: Retries on connection errors and retryable HTTP status.
            verify_tls: Verify the server certificate.
            extra_body: Additional request fields (e.g. ``return_token_ids``).
        """
        headers: dict[str, str] = {}
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise PolicyError(f"environment variable {api_key_env} is not set")
            headers["Authorization"] = f"Bearer {api_key}"
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._sampling = dict(sampling)
        self._extra_body = dict(extra_body or {})
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            verify=verify_tls,
            headers=headers,
            trust_env=False,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "model": self._model,
            "url": self._url,
            "sampling": self._sampling,
            "extra_body": self._extra_body,
        }

    async def act(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        turn_index: int,
    ) -> PolicyTurn:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "tools": list(tools),
            "tool_choice": "auto",
            **self._sampling,
            **self._extra_body,
        }
        body.setdefault("logprobs", True)
        body.setdefault("top_logprobs", 0)
        body.setdefault("return_token_ids", True)
        start = time.monotonic()
        payload = await self._post(body)
        latency = time.monotonic() - start
        return self._parse(payload, turn_index, latency)

    async def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self._url, json=body)
            except httpx.TransportError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in RETRYABLE_STATUS:
                    break
            if attempt < self._max_retries:
                delay = 2.0 ** attempt
                logger.warning("policy request failed (%s); retry %d in %.0fs", last_error, attempt + 1, delay)
                await asyncio.sleep(delay)
        raise PolicyError(last_error)

    def _parse(self, payload: Mapping[str, Any], turn_index: int, latency: float) -> PolicyTurn:
        choices = payload.get("choices") or []
        if not choices:
            raise PolicyError(f"response has no choices: {str(payload)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = [
            ToolCall.from_raw(
                call.get("id") or f"call-{turn_index}-{index}",
                (call.get("function") or {}).get("name", ""),
                (call.get("function") or {}).get("arguments") or "",
            )
            for index, call in enumerate(message.get("tool_calls") or [])
        ]
        usage = dict(payload.get("usage") or {})
        tokens = extract_choice_tokens(choice, payload)
        status = rollout_token_status(
            tokens["prompt_token_ids"],
            tokens["response_token_ids"],
            tokens["response_logprobs"],
            tool_call_count=len(calls),
            usage=usage,
            token_ids_cover=tokens["token_ids_cover"],
        )
        if status != "aligned":
            logger.info("turn %d logprobs %s (tool_calls=%d)", turn_index, status, len(calls))
        return PolicyTurn(
            turn_index=turn_index,
            model=payload.get("model", self._model),
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content"),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            prompt_token_ids=tokens["prompt_token_ids"],
            response_token_ids=tokens["response_token_ids"],
            response_logprobs=tokens["response_logprobs"],
            logprob_status=status,
            token_ids_cover=tokens["token_ids_cover"] if isinstance(tokens["token_ids_cover"], str) else None,
            latency_s=latency,
        )

    async def close(self) -> None:
        await self._client.aclose()
