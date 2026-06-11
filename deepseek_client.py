import json
import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-pro"
MAX_CONTEXT_TOKENS = 160000
MAX_TOKENS = 16000
MAX_HISTORY_MESSAGES = 20  # cap on accumulated user/assistant messages (excluding system)


def trim_history(history: list) -> list:
    """Keep only the most recent messages so accumulated context stays bounded."""
    if not history:
        return []
    return history[-MAX_HISTORY_MESSAGES:]


def chat(api_key: str, messages: list, temperature: float = 0.7, response_format=None):
    if not api_key:
        raise ValueError("DeepSeek API 키가 필요합니다.")

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
