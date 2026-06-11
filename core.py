import json
import os
import shutil
import time

import deepseek_client
import local_config
from exif_reader import read_image_metadata
from tag_db import TagDB
from wd14_tagger import predict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAVED_TAG_DB_PATH = os.path.join(DATA_DIR, "tag_db.csv")

EXAMPLE_PRESET = {
    "id": "<timestamp_ms>",
    "name": "<series name in Korean>",
    "scenes": [
        {
            "id": "<timestamp_ms>",
            "name": "<short scene code, e.g. m_s_1>",
            "scenePrompt": "<grouped danbooru tags in English, see ASSET_GROUPING_RULES>",
            "queueCount": 0,
            "images": [],
            "createdAt": "<timestamp_ms>",
            "width": 1216,
            "height": 832,
        }
    ],
    "createdAt": "<timestamp_ms>",
}

ASSET_GROUPING_RULES = """[IMAGE ASSET SYSTEM - PROMPT GROUPING RULES]

NAMING RULE for scene/file codes: [char]_[category]_[number]
NUMBER MEANING: 1-3=resistance/daily, 4-6=acceptance, 7-9=indulgence, 10+=full corruption
CATEGORIES: loc=location sex=intercourse orl=oral fpl=foreplay emo=expression com=daily grp=group (expand as needed)
CHARS: defined per project by the user (e.g. a=Alice, b=Bob)

PROMPT GROUPING (MODE 2 style):
Every generated prompt MUST be a single line where tags are grouped thematically inside curly braces { }, \
groups separated by ", ". Use this group order:
{quality}, {background}, {composition / camera angle}, [for each character present, two adjacent groups:] \
{that character's appearance traits}, {that character's pose / action / composition role}, {clothing}, {expression / emotional state / effects}

Rules:
- Each character gets its OWN pair of groups: one group for fixed appearance traits (body type, hair, skin, \
distinguishing features, "1girl"/"1boy"/"faceless male" etc.), and a separate adjacent group for that \
character's pose/action/role in the composition. Keeping these separate makes later edits easy \
(e.g. swap only the pose group without touching appearance).
- If male and female characters are both present, output the male group(s) first, then the female group(s), \
matching the example order: {male appearance}, {male pose/action}, {female appearance}, {female pose/action}.
- Quality tags first, then background, then composition/camera angle.
- Expression / emotional state / effect tags (blush, sweat, tears, trembling, etc.) always go in the LAST group.
- All tags inside groups must be in English, danbooru-style, comma separated within each group.
EXAMPLE: {masterpiece, best quality, highres}, {dark background}, {full body shot, from side}, \
{1boy, dark-skinned male, bald, faceless}, {standing, gripping her hips}, \
{1girl, long hair, black pubic hair}, {lying on back, legs spread}, {nude}, {blushing, trembling, biting lip, shame}"""


def load_tag_db(file_obj):
    db = TagDB()
    count = db.load(file_obj)
    if count == 0:
        return db, "태그 DB가 로드되지 않았습니다. (선택 사항)"

    # persist a copy so it survives restarts
    if hasattr(file_obj, "name"):
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copyfile(file_obj.name, SAVED_TAG_DB_PATH)

    return db, f"태그 DB 로드 완료: {count}개 태그"


def load_saved_state():
    """Called on app startup: restore the DeepSeek API key, tag DB, and standing notes from disk."""
    cfg = local_config.load_config()
    api_key = cfg.get("deepseek_api_key", "")
    notes = cfg.get("standing_notes", "")

    db = TagDB()
    status = "태그 DB가 로드되지 않았습니다. (선택 사항)"
    if os.path.exists(SAVED_TAG_DB_PATH):
        count = db.load(SAVED_TAG_DB_PATH)
        if count:
            status = f"태그 DB 로드 완료: {count}개 태그 (저장된 파일에서 복원)"

    return api_key, db, status, notes


def save_api_key(key):
    local_config.save_config(deepseek_api_key=key or "")
    return key


def save_notes(notes):
    local_config.save_config(standing_notes=notes or "")
    return notes


# ---------------------------------------------------------------------------
# Tab 1: image -> tags
# ---------------------------------------------------------------------------

def tag_image(image, general_threshold, character_threshold, db: TagDB, filter_existing):
    if image is None:
        return "이미지를 업로드해주세요.", ""

    general, character, rating = predict(image, general_threshold, character_threshold)

    if filter_existing and db is not None and len(db) > 0:
        general = db.filter_existing(general)
        character = db.filter_existing(character)

    all_tags = [n for n, _ in character] + [n for n, _ in general]
    tag_string = ", ".join(t.replace("_", " ") for t in all_tags)

    detail_lines = []
    if rating:
        detail_lines.append(f"Rating: {rating}")
    detail_lines.append("Character tags:")
    detail_lines += [f"  {n} ({p:.2f})" for n, p in character]
    detail_lines.append("General tags:")
    detail_lines += [f"  {n} ({p:.2f})" for n, p in general]

    return tag_string, "\n".join(detail_lines)


def analyze_image_metadata(image_path):
    if image_path is None:
        return "이미지를 업로드해주세요.", ""
    raw, prompt = read_image_metadata(image_path)
    return raw, prompt


# ---------------------------------------------------------------------------
# Tab 2: natural language -> tag combination (2-step DeepSeek calls)
# ---------------------------------------------------------------------------

CONCEPT_SYSTEM_PROMPT = """You are an assistant that breaks down an image generation request \
into a list of concrete visual concepts that should be represented as danbooru-style tags.
Read the user's request (it may be written in Korean) and output ONLY a JSON object of the form:
{"concepts": ["concept1", "concept2", ...]}

Cover ALL of the following aspects whenever relevant to the request, each as its own concept:
- composition / camera angle / shot framing (e.g. "full body shot", "from above", "close-up", "dutch angle")
- background / setting
- number and type of characters present (e.g. "1girl", "1boy", "faceless male")
- character physical features (body type, hair, skin, distinguishing features)
- clothing / state of undress
- pose / action / position
- interaction between characters if any
- facial expression and emotional state
- lighting / atmosphere / effects (sweat, blush, tears, etc.)

Produce at least 10-15 concepts in total so the final prompt can be rich and well composed. \
Each concept should be a short English phrase describing ONE visual element. \
Do not include any explanation, only the JSON."""

FINAL_SYSTEM_PROMPT = """You are an assistant that builds final danbooru tag prompts for an image generation model (NovelAI style).
You will receive the user's original request (possibly in Korean), a list of candidate tags \
retrieved from a tag database (each with a Korean description), and a requested number of variants.

""" + ASSET_GROUPING_RULES + """

Additional rules:
- Prefer tags from the candidate list when they fit, since those are confirmed to exist in the tag database.
- If the candidate list is missing tags needed for composition, camera angle, quality, or background, \
you MAY add common, well-known danbooru/NovelAI tags for those even if they are not in the candidate list.
- Each variant should be a single, detailed, well-composed prompt (aim for 20-35 tags total across all groups).
- Use underscores or spaces as found in the candidates for tags taken from the database; \
for added tags, use standard danbooru tag formatting (lowercase, underscores between words).
- If multiple variants are requested, make them meaningfully different (different composition/pose/angle) \
while staying consistent with the user's request.
- If the conversation history contains earlier prompts, treat the new request as a refinement/follow-up \
of that conversation (the user may be asking for a small change relative to the previous result).

Respond ONLY with a JSON object of the form:
{"variants": [{"tags": "{group1}, {group2}, ...", "explanation": "<설명을 한국어로 작성>"}, ...]}
The "tags" field must be in English, formatted per the grouping rules above (curly braces around each group). \
The "explanation" field must be written in Korean, briefly explaining the composition and why these tags were chosen."""


def generate_tag_combo(api_key, user_request, db: TagDB, variant_count: int,
                        standing_notes: str = "", history: list = None, accumulate: bool = False):
    history = history or []
    if not api_key:
        return "", "DeepSeek API 키를 입력해주세요.", "", history
    if not user_request or not user_request.strip():
        return "", "요청 내용을 입력해주세요.", "", history
    if db is None or len(db) == 0:
        return "", "먼저 태그 DB(CSV)를 업로드해주세요.", "", history

    variant_count = max(1, min(int(variant_count or 1), 5))

    try:
        return _generate_tag_combo_inner(api_key, user_request, db, variant_count, standing_notes, history, accumulate)
    except (RuntimeError, ValueError) as e:
        return "", "", f"오류 발생: {e}", history


def _generate_tag_combo_inner(api_key, user_request, db, variant_count, standing_notes, history, accumulate):
    # Step 1: ask DeepSeek for required concepts
    step1_user_content = user_request
    if standing_notes and standing_notes.strip():
        step1_user_content += (
            f"\n\nSTANDING INSTRUCTIONS / CORRECTIONS (always follow these, "
            f"they fix things the AI previously got wrong):\n{standing_notes.strip()}"
        )
    step1_messages = [
        {"role": "system", "content": CONCEPT_SYSTEM_PROMPT},
        {"role": "user", "content": step1_user_content},
    ]
    raw_concepts = deepseek_client.chat(
        api_key, step1_messages, temperature=0.5,
        response_format={"type": "json_object"},
    )
    try:
        concepts = json.loads(raw_concepts).get("concepts", [])
    except json.JSONDecodeError:
        concepts = [user_request]

    # Step 2: python looks up matching tags in the local tag DB
    candidates = db.candidates_for_terms(concepts, per_term_limit=8)
    candidate_lines = [
        f"{c['name']} : {c['description']}" for c in candidates
    ]
    candidates_text = "\n".join(candidate_lines) if candidate_lines else "(no candidates found)"

    # Step 3: ask DeepSeek to compose the final tag prompt(s)
    step2_user_content = (
        f"User request:\n{user_request}\n\n"
        f"Number of variants requested: {variant_count}\n\n"
        f"Candidate tags from the database:\n{candidates_text}"
    )
    if standing_notes and standing_notes.strip():
        step2_user_content += (
            f"\n\nSTANDING INSTRUCTIONS / CORRECTIONS (always follow these, "
            f"they fix things the AI previously got wrong):\n{standing_notes.strip()}"
        )
    step2_messages = [{"role": "system", "content": FINAL_SYSTEM_PROMPT}]
    if accumulate:
        step2_messages += deepseek_client.trim_history(history)
    step2_messages.append({"role": "user", "content": step2_user_content})

    raw_final = deepseek_client.chat(
        api_key, step2_messages, temperature=0.8,
        response_format={"type": "json_object"},
    )
    try:
        final = json.loads(raw_final)
        variants = final.get("variants", [])
    except json.JSONDecodeError:
        variants = [{"tags": raw_final, "explanation": ""}]

    if not variants:
        variants = [{"tags": "", "explanation": "결과를 생성하지 못했습니다."}]

    tags_blocks = []
    explanation_blocks = []
    for i, v in enumerate(variants, start=1):
        prefix = f"--- Variant {i} ---\n" if len(variants) > 1 else ""
        tags_blocks.append(prefix + v.get("tags", ""))
        explanation_blocks.append(prefix + v.get("explanation", ""))

    new_history = history
    if accumulate:
        new_history = deepseek_client.trim_history(
            history + [
                {"role": "user", "content": step2_user_content},
                {"role": "assistant", "content": raw_final},
            ]
        )

    debug_info = "검색된 후보 태그:\n" + candidates_text
    return "\n\n".join(tags_blocks), "\n\n".join(explanation_blocks), debug_info, new_history


# ---------------------------------------------------------------------------
# Tab 3: series description -> NAIS preset JSON (multi-scene)
# ---------------------------------------------------------------------------

MULTI_SCENE_SYSTEM_PROMPT = """You are an assistant that writes prompt presets for the NAIS image generation tool.
Given a description of a series of scenes (in Korean), output a single JSON object with EXACTLY this structure:

{
  "id": "<13-digit timestamp string>",
  "name": "<series name, in Korean>",
  "scenes": [
    {
      "id": "<13-digit timestamp string, unique per scene>",
      "name": "<short scene code following the NAMING RULE below, e.g. m_s_1>",
      "scenePrompt": "<grouped danbooru tags in English, see PROMPT GROUPING RULES below>",
      "queueCount": 0,
      "images": [],
      "createdAt": <same number as id, as an integer>,
      "width": 1216,
      "height": 832
    }
  ],
  "createdAt": <same number as the top-level id, as an integer>
}

""" + ASSET_GROUPING_RULES + """

Additional rules:
- "scenePrompt" must follow the PROMPT GROUPING RULES above (curly-brace groups, English danbooru tags).
- You will be given a list of candidate tags retrieved from a tag database. Prefer these tags when they fit, \
since they are confirmed to exist in the database. You may still add common, well-known danbooru/NovelAI \
tags (quality, composition, etc.) that are not in the candidate list.
- Scene "name" codes must follow the NAMING RULE ([char]_[category]_[number]) using the CHARS/CATEGORIES \
the user provides (or sensible defaults if none given), and the number should reflect the \
NUMBER MEANING (1-3/4-6/7-9/10+) for that scene's intensity.
- Generate as many scenes as make sense for the user's description (each meaningful step/pose should be its own scene).
- "id" and "createdAt" values must be plausible 13-digit millisecond timestamps, each scene with a distinct id.
- If the conversation history contains an earlier series JSON, treat the new request as a revision/follow-up \
of that series (the user may be asking to add, change, or extend scenes).

OUTPUT FORMAT: respond with ONLY the JSON object described above, valid JSON, ready to be pasted directly \
into the NAI preset tool. Do not add any commentary, explanation, or extra text before or after the JSON."""


SCENE_DESCRIPTION_SYSTEM_PROMPT = """You are an assistant that writes short Korean descriptions for scenes \
in a NAIS image-generation preset JSON.
You will be given the series description (Korean) and the generated preset JSON (containing scene names \
and scenePrompt tags).
For EACH scene in the JSON, output one line in this exact format:
[scene name]: [한국어로 이 씬이 어떤 장면인지 서술]

Output ONLY these lines, one per scene, in the same order as the scenes appear in the JSON. \
No extra commentary, no headers."""


def generate_multi_scene(api_key, description, char_def, db: TagDB,
                          standing_notes: str = "", history: list = None, accumulate: bool = False):
    history = history or []
    if not api_key:
        yield "", "", "DeepSeek API 키를 입력해주세요.", "", history
        return
    if not description or not description.strip():
        yield "", "", "시리즈 설명을 입력해주세요.", "", history
        return

    try:
        for item in _generate_multi_scene_inner(api_key, description, char_def, db, standing_notes, history, accumulate):
            yield item
    except (RuntimeError, ValueError) as e:
        yield "", "", f"오류 발생: {e}", "", history


def _generate_multi_scene_inner(api_key, description, char_def, db, standing_notes, history, accumulate):
    candidates_text = "(태그 DB가 업로드되지 않았습니다)"
    if db is not None and len(db) > 0:
        # Step 1: ask DeepSeek for the visual concepts needed across all scenes
        concept_messages = [
            {"role": "system", "content": CONCEPT_SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ]
        raw_concepts = deepseek_client.chat(
            api_key, concept_messages, temperature=0.5,
            response_format={"type": "json_object"},
        )
        try:
            concepts = json.loads(raw_concepts).get("concepts", [])
        except json.JSONDecodeError:
            concepts = [description]

        # Step 2: python looks up matching tags in the local tag DB
        candidates = db.candidates_for_terms(concepts, per_term_limit=8)
        candidate_lines = [f"{c['name']} : {c['description']}" for c in candidates]
        candidates_text = "\n".join(candidate_lines) if candidate_lines else "(no candidates found)"

    base_ts = int(time.time() * 1000)
    user_content = (
        f"Reference structure (field names and types only, not real content):\n"
        f"{json.dumps(EXAMPLE_PRESET, ensure_ascii=False, indent=2)}\n\n"
        f"A timestamp around {base_ts} can be used as a base for generating IDs.\n\n"
    )
    if char_def and char_def.strip():
        user_content += f"CHARS / CATEGORIES for naming: {char_def.strip()}\n\n"
    if standing_notes and standing_notes.strip():
        user_content += (
            f"STANDING INSTRUCTIONS / CORRECTIONS (always follow these, "
            f"they fix things the AI previously got wrong):\n{standing_notes.strip()}\n\n"
        )
    user_content += (
        f"Candidate tags from the database (prefer these when they fit, "
        f"since they are confirmed to exist):\n{candidates_text}\n\n"
    )
    user_content += f"Series description:\n{description}"

    messages = [{"role": "system", "content": MULTI_SCENE_SYSTEM_PROMPT}]
    if accumulate:
        messages += deepseek_client.trim_history(history)
    messages.append({"role": "user", "content": user_content})

    raw = ""
    for raw in deepseek_client.chat_stream(
        api_key, messages, temperature=0.8,
    ):
        debug_info = "검색된 후보 태그:\n" + candidates_text + "\n\n--- AI 원본 응답 (JSON, 생성 중) ---\n" + raw
        yield raw, "", "JSON 생성 중...", debug_info, history

    json_part = raw

    new_history = history
    if accumulate:
        new_history = deepseek_client.trim_history(
            history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": raw},
            ]
        )

    json_part = json_part.strip()
    if json_part.startswith("```"):
        json_part = json_part.split("\n", 1)[1] if "\n" in json_part else json_part
        if json_part.endswith("```"):
            json_part = json_part[:-3]
        json_part = json_part.strip()

    try:
        parsed = json.loads(json_part)
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        debug_info = "검색된 후보 태그:\n" + candidates_text + "\n\n--- AI 원본 응답 (JSON) ---\n" + raw
        yield json_part, "", "JSON 파싱에 실패했습니다. 원본 응답을 표시합니다.", debug_info, new_history
        return

    yield pretty, "", "씬별 설명 생성 중...", "검색된 후보 태그:\n" + candidates_text + "\n\n--- AI 원본 응답 (JSON) ---\n" + raw, new_history

    # Step 4: ask DeepSeek for per-scene Korean descriptions, based on the generated JSON
    desc_messages = [
        {"role": "system", "content": SCENE_DESCRIPTION_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Series description:\n{description}\n\n"
            f"Generated preset JSON:\n{pretty}"
        )},
    ]
    description_part = ""
    for description_part in deepseek_client.chat_stream(api_key, desc_messages, temperature=0.5):
        debug_info = (
            "검색된 후보 태그:\n" + candidates_text
            + "\n\n--- AI 원본 응답 (JSON) ---\n" + raw
            + "\n\n--- AI 원본 응답 (설명, 생성 중) ---\n" + description_part
        )
        yield pretty, description_part, "씬별 설명 생성 중...", debug_info, new_history

    debug_info = (
        "검색된 후보 태그:\n" + candidates_text
        + "\n\n--- AI 원본 응답 (JSON) ---\n" + raw
        + "\n\n--- AI 원본 응답 (설명) ---\n" + description_part
    )
    yield pretty, description_part, "생성 완료", debug_info, new_history


# ---------------------------------------------------------------------------
# Tab 4: Image Asset System (filename / prompt / asset guide)
# ---------------------------------------------------------------------------

ASSET_SYSTEM_PROMPT = """[IMAGE ASSET SYSTEM]

NAMING RULE: [char]_[category]_[number]
NUMBER MEANING: 1-3=resistance/daily 4-6=acceptance 7-9=indulgence 10+=full corruption

CHARS: defined per project by the user (e.g. a=Alice b=Bob)
CATEGORIES: loc=location sex=intercourse orl=oral fpl=foreplay emo=expression com=daily grp=group (expand as needed)

---

MODE 1 - FILENAME DEFINITION
TRIGGER: "define filename" / "create series"
OUTPUT FORMAT:
SERIES: [series_name]
[char]_[cat]_[number]: [keywords only, comma separated]
RULES: no sentences, keywords only, follow number meaning

---

MODE 2 - PROMPT GENERATION
TRIGGER: "generate prompt" / "NAI prompt"
OUTPUT FORMAT (single line, groups in {}, comma separated tags inside):
{quality}, {background}, {male if present}, {female + physical features}, {clothing}, {act/position}, {expression/emotional state}
RULES: English tags only, each thematic group wrapped in {}, written as ONE single line, character-specific fixed traits always included in their {}, emotions concentrated in last {}
EXAMPLE: {masterpiece, best quality, highres}, {dark background}, {1dark-skinned male, bald, faceless}, {1girl, black pubic hair, sweat}, {full nelson, standing sex}, {blushing, trembling, biting lip, shame}

---

MODE 3 - ASSET GUIDE WRITE
TRIGGER: "write asset guide" / "create guide"
OUTPUT FORMAT:
[SERIES NAME]
FORMAT: [char]_[cat]_[number]
1-3:[summary] 4-6:[summary] 7-9:[summary] 10+:[summary]
ENTRIES: _1:[keywords] _2:[keywords] _3:[keywords] ...
RULES: no USAGE GUIDELINES section, no sentences, keywords and numbers only

---

GENERAL RULES:
- Output tags / keywords / prompts in English.
- Any free-text explanation outside the required output format must be written in Korean.
- Strictly follow the OUTPUT FORMAT of the requested mode, no extra sections."""

MODE_TRIGGERS = {
    "1. 파일명 정의 (define filename)": "define filename",
    "2. NAI 프롬프트 생성 (generate prompt)": "generate prompt",
    "3. 에셋 가이드 작성 (write asset guide)": "write asset guide",
}


def generate_asset_output(api_key, mode_label, char_def, user_input, history: list = None, accumulate: bool = False):
    history = history or []
    if not api_key:
        return "DeepSeek API 키를 입력해주세요.", history
    if not user_input or not user_input.strip():
        return "내용을 입력해주세요.", history

    try:
        return _generate_asset_output_inner(api_key, mode_label, char_def, user_input, history, accumulate)
    except (RuntimeError, ValueError) as e:
        return f"오류 발생: {e}", history


def _generate_asset_output_inner(api_key, mode_label, char_def, user_input, history, accumulate):
    trigger = MODE_TRIGGERS[mode_label]
    user_content = trigger
    if char_def and char_def.strip():
        user_content += f"\n\nCHARS: {char_def.strip()}"
    user_content += f"\n\n{user_input.strip()}"

    messages = [{"role": "system", "content": ASSET_SYSTEM_PROMPT}]
    if accumulate:
        messages += deepseek_client.trim_history(history)
    messages.append({"role": "user", "content": user_content})

    result = deepseek_client.chat(api_key, messages, temperature=0.7)

    new_history = history
    if accumulate:
        new_history = deepseek_client.trim_history(
            history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": result},
            ]
        )

    return result, new_history
