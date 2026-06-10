import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
import onnxruntime as ort

MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
MODEL_FILE = "model.onnx"
LABEL_FILE = "selected_tags.csv"

_session = None
_tags_df = None
_input_size = None


def _load():
    global _session, _tags_df, _input_size
    if _session is not None:
        return
    model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    label_path = hf_hub_download(MODEL_REPO, LABEL_FILE)

    _session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    _tags_df = pd.read_csv(label_path)
    _input_size = _session.get_inputs()[0].shape[1]


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    arr = np.asarray(image)
    arr = arr[:, :, ::-1]  # RGB -> BGR

    size = max(arr.shape[:2])
    pad_y = size - arr.shape[0]
    pad_x = size - arr.shape[1]
    top, left = pad_y // 2, pad_x // 2
    arr = np.pad(
        arr,
        ((top, pad_y - top), (left, pad_x - left), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    img = Image.fromarray(arr).resize((_input_size, _input_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return arr[np.newaxis, :, :, :]


def predict(image: Image.Image, general_threshold: float = 0.35, character_threshold: float = 0.85):
    """Returns (general_tags, character_tags, rating) for the given image."""
    _load()

    input_name = _session.get_inputs()[0].name
    output_name = _session.get_outputs()[0].name
    probs = _session.run([output_name], {input_name: _preprocess(image)})[0][0]

    df = _tags_df.copy()
    df["prob"] = probs

    ratings = df[df["category"] == 9].sort_values("prob", ascending=False)
    rating = ratings.iloc[0]["name"] if len(ratings) else None

    general = df[(df["category"] == 0) & (df["prob"] >= general_threshold)]
    general = general.sort_values("prob", ascending=False)

    character = df[(df["category"] == 4) & (df["prob"] >= character_threshold)]
    character = character.sort_values("prob", ascending=False)

    general_tags = [(row["name"], float(row["prob"])) for _, row in general.iterrows()]
    character_tags = [(row["name"], float(row["prob"])) for _, row in character.iterrows()]

    return general_tags, character_tags, rating
