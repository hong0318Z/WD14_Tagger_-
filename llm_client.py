import time

from openai import OpenAI

DEFAULT_BASE_URL = "http://192.168.0.116:8000/v1"
AVAILABLE_MODELS = [
    "mlx-community--gemma-4-26b-a4b-it-8bit",
    "Qwen3.6-35B-A3B-4bit",
    "gpt-oss-20b-MXFP4-Q8",
]
DEFAULT_MODEL = AVAILABLE_MODELS[0]
MAX_TOKENS = 32768
MAX_HISTORY_MESSAGES = 20
REQUEST_TIMEOUT = 300


def trim_history(history: list) -> list:
    if not history:
        return []
    return history[-MAX_HISTORY_MESSAGES:]


def _client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(
        api_key=api_key or "local",
        base_url=base_url or DEFAULT_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


def chat(api_key: str, messages: list, temperature: float = 0.7,
         model: str = None, base_url: str = None, response_format=None) -> str:
    model = model or DEFAULT_MODEL
    client = _client(api_key, base_url)
    started = time.time()
    print(f"[llm] request: model={model} messages={len(messages)} chars={sum(len(m['content']) for m in messages)}")
    kwargs = dict(model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=temperature)
    if response_format:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - started
    content = resp.choices[0].message.content
    print(f"[llm] done: elapsed={elapsed:.1f}s usage={resp.usage}")
    return content


def chat_stream(api_key: str, messages: list, temperature: float = 0.7,
                model: str = None, base_url: str = None, response_format=None):
    """Yields (accumulated_text, finish_reason). finish_reason is None for mid-stream yields."""
    model = model or DEFAULT_MODEL
    client = _client(api_key, base_url)
    started = time.time()
    print(f"[llm] stream: model={model} messages={len(messages)} chars={sum(len(m['content']) for m in messages)}")
    kwargs = dict(model=model, messages=messages, max_tokens=MAX_TOKENS, temperature=temperature, stream=True)
    if response_format:
        kwargs["response_format"] = response_format

    full = ""
    finish_reason = None
    with client.chat.completions.create(**kwargs) as stream:
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta.content or ""
            if delta:
                full += delta
                yield full, None
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    elapsed = time.time() - started
    print(f"[llm] stream done: elapsed={elapsed:.1f}s chars={len(full)} finish_reason={finish_reason}")
    yield full, finish_reason
