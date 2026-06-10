import json

from PIL import Image


def read_image_metadata(image_path: str):
    """Returns (raw_metadata_text, extracted_prompt) for an image file.

    Looks at PNG text chunks commonly used by NAI / Stable Diffusion
    (e.g. "parameters", "Comment", "Software", "Description").
    A file path is required (not a re-encoded PIL image) because
    re-encoding via libraries like Gradio's Image component strips
    these text chunks.
    """
    image = Image.open(image_path)
    info = image.info or {}
    if not info:
        return "메타데이터가 없습니다 (이미지에 PNG info 청크가 없음).", ""

    lines = []
    prompt = ""

    for key, value in info.items():
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                value = str(value)
        lines.append(f"[{key}]\n{value}")

        if key.lower() == "comment" and not prompt:
            try:
                data = json.loads(value)
                prompt = data.get("prompt", "")
            except (json.JSONDecodeError, AttributeError):
                pass

        if key.lower() == "parameters" and not prompt:
            prompt = str(value).split("\nNegative prompt:")[0].strip()

    return "\n\n".join(lines), prompt
