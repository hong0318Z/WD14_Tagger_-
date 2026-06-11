import json
import time

import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"
AVAILABLE_MODELS = ["deepseek-v4-pro", "deepseek-v4-flash"]
MAX_CONTEXT_TOKENS = 160000
MAX_TOKENS = 50000
MAX_HISTORY_MESSAGES = 20  # cap on accumulated user/assistant messages (excluding system)
REQUEST_TIMEOUT = 180  # seconds


def trim_history(history: list) -> list:
    """Keep only the most recent messages so accumulated context stays bounded."""
    if not history:
        return []
    return history[-MAX_HISTORY_MESSAGES:]


def chat(api_key: str, messages: list, temperature: float = 0.7, response_format=None, model: str = None):
    if not api_key:
        raise ValueError("DeepSeek API 키가 필요합니다.")

    model = model or MODEL
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    started = time.time()
    print(f"[deepseek] request: model={model} messages={len(messages)} "
          f"chars={sum(len(m['content']) for m in messages)}")

    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - started
        raise RuntimeError(f"DeepSeek API 요청이 {elapsed:.0f}초 후 타임아웃되었습니다 (limit={REQUEST_TIMEOUT}s).") from e
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - started
        raise RuntimeError(f"DeepSeek API 요청 실패 ({elapsed:.0f}초 경과): {e}") from e

    elapsed = time.time() - started
    print(f"[deepseek] response: status={resp.status_code} elapsed={elapsed:.1f}s")

    if resp.status_code != 200:
        raise RuntimeError(
            f"DeepSeek API 오류 (status={resp.status_code}, {elapsed:.1f}s): {resp.text[:1000]}"
        )

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"[deepseek] usage: {usage}")
    return content


def chat_stream(api_key: str, messages: list, temperature: float = 0.7, response_format=None, model: str = None):
    """Yields the accumulated response text as it streams in from the API."""
    if not api_key:
        raise ValueError("DeepSeek API 키가 필요합니다.")

    model = model or MODEL
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "stream": True,
    }
    if response_format:
        payload["response_format"] = response_format

    started = time.time()
    print(f"[deepseek] stream request: model={model} messages={len(messages)} "
          f"chars={sum(len(m['content']) for m in messages)}")

    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
    except requests.exceptions.Timeout as e:
        elapsed = time.time() - started
        raise RuntimeError(f"DeepSeek API 요청이 {elapsed:.0f}초 후 타임아웃되었습니다 (limit={REQUEST_TIMEOUT}s).") from e
    except requests.exceptions.RequestException as e:
        elapsed = time.time() - started
        raise RuntimeError(f"DeepSeek API 요청 실패 ({elapsed:.0f}초 경과): {e}") from e

    if resp.status_code != 200:
        elapsed = time.time() - started
        raise RuntimeError(
            f"DeepSeek API 오류 (status={resp.status_code}, {elapsed:.1f}s): {resp.text[:1000]}"
        )

    full = ""
    finish_reason = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str.strip() == "[DONE]":
            break
        chunk = json.loads(data_str)
        choice = chunk["choices"][0]
        delta = choice.get("delta", {}).get("content", "")
        if delta:
            full += delta
            yield full, None
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    elapsed = time.time() - started
    print(f"[deepseek] stream done: elapsed={elapsed:.1f}s chars={len(full)} finish_reason={finish_reason}")
    yield full, finish_reason
