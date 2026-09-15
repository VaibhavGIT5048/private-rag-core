"""Chat providers — Azure OpenAI Service by default, plain OpenAI for BYO-key.

Both speak the OpenAI Chat Completions wire format (`chat.completions.create`),
so RAGService calls either one identically. The only difference is which
`openai.OpenAI` client/base_url/model gets constructed underneath.
"""

from __future__ import annotations

import os

from openai import BadRequestError, OpenAI


class ContentFilterError(Exception):
    """Azure OpenAI's Responsible AI filter blocked the request or response.

    Raised instead of letting BadRequestError propagate as an unhandled 500 —
    a content-filter trip is an expected, user-facing outcome (422), not a
    server fault. Distinct from the general BadRequestError re-raise in
    _create() below: that one is for a param the model rejects, which is a
    bug in our own request; this one is Azure refusing to answer at all.
    """


class ChatProvider:
    chat_model: str
    # Reasoning-family models (gpt-5-mini included) burn part of their
    # completion-token budget on invisible internal reasoning before writing
    # any visible output — confirmed anywhere from 0 to 448 tokens on the same
    # trivial prompt across repeated calls. For a grounded-QA task where the
    # real reasoning already happened during retrieval, 'minimal' reliably
    # measured 0 reasoning tokens with no quality loss, vs. hundreds by
    # default — real cost/latency, not just tidiness. None for providers that
    # don't support it at all (plain OpenAI models reject the param).
    reasoning_effort: str | None = None

    def _build_client(self) -> OpenAI:
        raise NotImplementedError

    @property
    def client(self) -> OpenAI:
        if getattr(self, "_client", None) is None:
            self._client = self._build_client()
        return self._client

    def _create(self, **kwargs) -> str:
        """chat.completions.create with unsupported-param self-healing.

        Reasoning models reject params like `temperature` or
        `reasoning_effort` outright (400, structured `param` field naming the
        culprit) — rather than hardcoding which models support what, strip
        whatever the API itself just said it doesn't like and retry. Bounded
        by kwargs' size, so this can't loop forever.
        """
        for _ in range(len(kwargs)):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content or ""
            except BadRequestError as exc:
                # The SDK sets .body to the *inner* error object already
                # (openai._client.OpenAI._make_status_error unwraps
                # response_json["error"] before constructing the exception),
                # so this reads the same top-level keys the raw API error
                # carries — {message, type, param, code, ...} — not a
                # {"error": {...}} wrapper.
                body = getattr(exc, "body", None) or {}
                code = body.get("code") if isinstance(body, dict) else None
                if code == "content_filter":
                    # Azure's Responsible AI layer sits in front of the model
                    # and can refuse a request outright — most relevantly here,
                    # its jailbreak classifier does not distinguish text that
                    # DESCRIBES injection/jailbreak patterns (our own system
                    # prompt, defending against them) from text that IS one.
                    # Dense defensive vocabulary — "ignore", "override",
                    # "jailbreak payload" — reads the same to the classifier
                    # either way.
                    raise ContentFilterError(
                        "This question could not be answered: Azure's safety "
                        "filter flagged the request. Try rephrasing the "
                        "question."
                    ) from exc
                bad_param = body.get("param") if isinstance(body, dict) else None
                if bad_param and bad_param in kwargs:
                    kwargs.pop(bad_param)
                    continue
                raise
        return ""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0) -> str:
        return self.complete_messages(
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0,
    ) -> str:
        """Complete an explicitly structured chat turn.

        Keeping system guidance in a system message is important for both
        instruction hierarchy and Azure's safety classifier: a long policy
        document pasted into a user message can look like the very attack it
        is trying to defend against.
        """
        # max_completion_tokens, not max_tokens — gpt-5-mini rejects
        # max_tokens outright with a 400, and OpenAI's own API has deprecated
        # max_tokens in favor of this name across the board, so it's the
        # right call for both providers, not just Azure.
        kwargs = {
            "model": self.chat_model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "messages": messages,
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return self._create(**kwargs)

    def is_healthy(self) -> bool:
        # A minimal real completion, not client.models.retrieve() — Azure's
        # v1 endpoint doesn't expose model-listing the way OpenAI's does, and
        # the thing we actually depend on is chat.completions working at all.
        # max_completion_tokens=20, not 1 — reasoning tokens count against
        # this budget too, and 1 is guaranteed to starve any visible output.
        kwargs = {
            "model": self.chat_model,
            "max_completion_tokens": 20,
            "messages": [{"role": "user", "content": "ping"}],
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            self._create(**kwargs)
            return True  # no exception raised — reachable and functioning
        except Exception:
            return False


class AzureOpenAIChatProvider(ChatProvider):
    """Default — billed to the Azure subscription, never a personal OpenAI account."""

    def __init__(self, api_key: str, endpoint: str, model: str, reasoning_effort: str | None = None):
        self._api_key = api_key
        self._base_url = _as_v1_base_url(endpoint)
        self.chat_model = model
        self.reasoning_effort = reasoning_effort
        self._client: OpenAI | None = None

    def _build_client(self) -> OpenAI:
        return OpenAI(api_key=self._api_key, base_url=self._base_url)


class OpenAIChatProvider(ChatProvider):
    """Plain OpenAI — used for a user-supplied BYO key, or as a local-dev
    fallback when no Azure credentials are configured at all.
    """

    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini"):
        self._api_key = api_key
        self.chat_model = model
        self._client: OpenAI | None = None

    def _build_client(self) -> OpenAI:
        return OpenAI(api_key=self._api_key)


def _as_v1_base_url(endpoint: str) -> str:
    """Normalizes an Azure AI Foundry endpoint to its `.../openai/v1` base.

    The Foundry portal's copy-paste endpoint often includes a trailing
    `/responses` (the Responses API path) — chat.completions.create() needs
    the bare `/openai/v1` base instead, per Azure's OpenAI-SDK-compatible v1
    surface (confirmed: "An Azure OpenAI resource provides only the
    /openai/v1 endpoint, and API keys work on /openai/v1 by passing the key
    as api_key" — https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/chatgpt).
    """
    base = endpoint.rstrip("/")
    if base.endswith("/responses"):
        base = base.rsplit("/responses", 1)[0]
    return base


def build_default_chat_provider() -> ChatProvider:
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")
    # Env-overridable so a future model that benefits from deeper reasoning
    # doesn't need a code change; empty string disables the param entirely.
    reasoning_effort = os.getenv("AZURE_OPENAI_REASONING_EFFORT", "minimal") or None
    if azure_key and azure_endpoint:
        return AzureOpenAIChatProvider(azure_key, azure_endpoint, azure_model, reasoning_effort)
    # No Azure creds configured — fall back to plain OpenAI (may still have
    # no key at all; that's fine, nothing touches the client until a request
    # actually needs it, matching the lazy-construction pattern throughout).
    return OpenAIChatProvider(os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
