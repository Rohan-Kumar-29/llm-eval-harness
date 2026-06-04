import os
import time
from dataclasses import dataclass, field
from typing import Optional

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


def _is_retryable(exc: BaseException) -> bool:
    """Only retry transient failures (rate limits, timeouts, upstream 5xx).

    Permanent errors — bad API key, malformed request, unknown model — are not
    retried, so we fail fast instead of burning four attempts and free-tier quota
    on something that can never succeed.
    """
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "429", "timeout", "overloaded", "503", "502"))


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _call_litellm(model_id: str, messages: list, temperature: float) -> litellm.ModelResponse:
    return await litellm.acompletion(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )


async def generate(
    model_id: str,
    system_prompt: str,
    user_input: str,
    temperature: float = 0.0,
) -> GenerationResult:
    """Call any LLM via litellm and return a GenerationResult.

    On hard failure returns a result with error set and text="" so the
    runner can continue without crashing the whole benchmark run.
    """
    _check_env_keys(model_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    start = time.perf_counter()
    try:
        response = await _call_litellm(model_id, messages, temperature)
        latency_ms = (time.perf_counter() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage or {}
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        return GenerationResult(
            text=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return GenerationResult(
            text="",
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(exc),
        )


def _check_env_keys(model_id: str) -> None:
    """Fail clearly if a required API key is missing."""
    checks = {
        "gemini/": ("GEMINI_API_KEY", "https://aistudio.google.com"),
        "groq/": ("GROQ_API_KEY", "https://console.groq.com"),
        "openrouter/": ("OPENROUTER_API_KEY", "https://openrouter.ai"),
    }
    for prefix, (env_var, url) in checks.items():
        if model_id.startswith(prefix):
            if not os.environ.get(env_var):
                raise EnvironmentError(
                    f"Missing {env_var} in .env — get a free key at {url}"
                )
