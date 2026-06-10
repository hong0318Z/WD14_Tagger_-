import json
import time

import gradio as gr
from PIL import Image

import deepseek_client
from exif_reader import read_image_metadata
from tag_db import TagDB
from wd14_tagger import predict

EXAMPLE_PRESET = {
    "id": "<timestamp_ms>",
    "name": "<series name in Korean>",
    "scenes": [
        {
            "id": "<timestamp_ms>",
            "name": "<short scene code>",
            "scenePrompt": "<comma separated danbooru tags in English>",
            "queueCount": 0,
            "images": [],
            "createdAt": "<timestamp_ms>",
            "width": 1216,
            "height": 832,
        }
    ],
    "createdAt": "<timestamp_ms>",
}


def load_tag_db(file_obj):
    db = TagDB()
    count = db.load(file_obj)
    if count == 0:
        return db, "태그 DB가 로드되지 않았습니다. (선택 사항)"
    return db, f"태그 DB 로드 완료: {count}개 태그"


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


# ---------------------------------------------------------------------------
# Tab 2: natural language -> tag combination (2-step DeepSeek calls)
# ---------------------------------------------------------------------------

CONCEPT_SYSTEM_PROMPT = """You are an assistant that breaks down an image generation request \
into a list of concrete visual concepts that should be represented as danbooru-style tags.
Read the user's request (it may be written in Korean) and output ONLY a JSON object of the form:
{"concepts": ["concept1", "concept2", ...]}
Each concept should be a short English phrase describing one visual element \
(pose, clothing, expression, action, setting, etc). Do not include any explanation, only the JSON."""

FINAL_SYSTEM_PROMPT = """You are an assistant that builds a final danbooru tag prompt for an image generation model.
You will receive the user's original request (possibly in Korean) and a list of candidate tags \
retrieved from a tag database, each with a Korean description.
Pick and order the most relevant tags to satisfy the user's request, and combine them into a single \
comma-separated list of English danbooru tags (use underscores or spaces as found in the candidates, \
keep them as valid danbooru tag names).
Respond ONLY with a JSON object of the form:
{"tags": "tag1, tag2, tag3, ...", "explanation": "<설명을 한국어로 작성>"}
The "tags" field must be in English. The "explanation" field must be written in Korean, \
briefly explaining why these tags were chosen."""


def generate_tag_combo(api_key, user_request, db: TagDB):
    if not api_key:
        return "", "DeepSeek API 키를 입력해주세요.", ""
    if not user_request or not user_request.strip():
        return "", "요청 내용을 입력해주세요.", ""
    if db is None or len(db) == 0:
        return "", "먼저 태그 DB(CSV)를 업로드해주세요.", ""

    # Step 1: ask DeepSeek for required concepts
    step1_messages = [
        {"role": "system", "content": CONCEPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
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

    # Step 3: ask DeepSeek to compose the final tag prompt
    step2_user_content = (
        f"User request:\n{user_request}\n\n"
        f"Candidate tags from the database:\n{candidates_text}"
    )
    step2_messages = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": step2_user_content},
    ]
    raw_final = deepseek_client.chat(
        api_key, step2_messages, temperature=0.7,
        response_format={"type": "json_object"},
    )
    try:
        final = json.loads(raw_final)
        tags = final.get("tags", "")
        explanation = final.get("explanation", "")
    except json.JSONDecodeError:
        tags = raw_final
        explanation = ""

    debug_info = "검색된 후보 태그:\n" + candidates_text
    return tags, explanation, debug_info


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
      "name": "<short scene code, e.g. m_s_1>",
      "scenePrompt": "<comma-separated danbooru tags in English describing this scene>",
      "queueCount": 0,
      "images": [],
      "createdAt": <same number as id, as an integer>,
      "width": 1216,
      "height": 832
    }
  ],
  "createdAt": <same number as the top-level id, as an integer>
}

Rules:
- "scenePrompt" must be written in English using danbooru-style tags, comma separated.
- You may use weighting syntax like "tag::weight::" or "{tag}" or "<group/option>" if it helps express the scene, \
following the style of typical NAI prompt presets.
- Generate as many scenes as make sense for the user's description (each meaningful step/pose should be its own scene).
- "id" and "createdAt" values must be plausible 13-digit millisecond timestamps, each scene with a distinct id.
- Output ONLY the JSON object, no extra commentary."""


def generate_multi_scene(api_key, description):
    if not api_key:
        return "", "DeepSeek API 키를 입력해주세요."
    if not description or not description.strip():
        return "", "시리즈 설명을 입력해주세요."

    base_ts = int(time.time() * 1000)
    user_content = (
        f"Reference structure (field names and types only, not real content):\n"
        f"{json.dumps(EXAMPLE_PRESET, ensure_ascii=False, indent=2)}\n\n"
        f"A timestamp around {base_ts} can be used as a base for generating IDs.\n\n"
        f"Series description:\n{description}"
    )
    messages = [
        {"role": "system", "content": MULTI_SCENE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = deepseek_client.chat(
        api_key, messages, temperature=0.8,
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(raw)
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
        return pretty, "생성 완료"
    except json.JSONDecodeError:
        return raw, "JSON 파싱에 실패했습니다. 원본 응답을 표시합니다."


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


def generate_asset_output(api_key, mode_label, char_def, user_input):
    if not api_key:
        return "DeepSeek API 키를 입력해주세요."
    if not user_input or not user_input.strip():
        return "내용을 입력해주세요."

    trigger = MODE_TRIGGERS[mode_label]
    user_content = trigger
    if char_def and char_def.strip():
        user_content += f"\n\nCHARS: {char_def.strip()}"
    user_content += f"\n\n{user_input.strip()}"

    messages = [
        {"role": "system", "content": ASSET_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return deepseek_client.chat(api_key, messages, temperature=0.7)


def analyze_image_metadata(image: Image.Image):
    if image is None:
        return "이미지를 업로드해주세요.", ""
    raw, prompt = read_image_metadata(image)
    return raw, prompt


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="WD14 Tagger Toolkit") as demo:
    gr.Markdown("# WD14 Tagger Toolkit")

    db_state = gr.State(TagDB())

    with gr.Accordion("설정", open=True):
        deepseek_key = gr.BrowserState("", storage_key="deepseek_api_key")
        api_key_box = gr.Textbox(
            label="DeepSeek API Key (브라우저에 저장됨)",
            type="password",
            placeholder="sk-...",
        )
        tag_db_file = gr.File(label="단부루 태그 CSV 업로드 (선택)", file_types=[".csv"])
        tag_db_status = gr.Markdown("태그 DB가 로드되지 않았습니다. (선택 사항)")

        demo.load(lambda k: k, inputs=deepseek_key, outputs=api_key_box)
        api_key_box.change(lambda k: k, inputs=api_key_box, outputs=deepseek_key)
        tag_db_file.change(load_tag_db, inputs=tag_db_file, outputs=[db_state, tag_db_status])

    with gr.Tab("1. 이미지 태그 분석"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="이미지 업로드")
                general_threshold = gr.Slider(0, 1, value=0.35, label="General tag threshold")
                character_threshold = gr.Slider(0, 1, value=0.85, label="Character tag threshold")
                filter_existing = gr.Checkbox(
                    label="업로드한 태그 DB에 존재하는 태그만 표시", value=False
                )
                tag_btn = gr.Button("태그 분석", variant="primary")
            with gr.Column():
                tag_output = gr.Textbox(label="태그 (복사해서 사용)", lines=4)
                tag_detail = gr.Textbox(label="상세 결과 (확률 포함)", lines=15)

        tag_btn.click(
            tag_image,
            inputs=[image_input, general_threshold, character_threshold, db_state, filter_existing],
            outputs=[tag_output, tag_detail],
        )

    with gr.Tab("2. 태그 조합 생성"):
        gr.Markdown("자연어로 원하는 이미지를 설명하면, 태그 DB에서 관련 태그를 찾아 AI가 조합해줍니다.")
        combo_request = gr.Textbox(label="요청 내용 (한국어 가능)", lines=4)
        combo_btn = gr.Button("태그 조합 생성", variant="primary")
        combo_tags = gr.Textbox(label="결과 태그 (영어, 복사해서 사용)", lines=3)
        combo_explanation = gr.Textbox(label="설명 (한국어)", lines=4)
        combo_debug = gr.Textbox(label="검색된 후보 태그 (디버그)", lines=10)

        combo_btn.click(
            generate_tag_combo,
            inputs=[deepseek_key, combo_request, db_state],
            outputs=[combo_tags, combo_explanation, combo_debug],
        )

    with gr.Tab("3. 다중 씬(시리즈) 생성"):
        gr.Markdown("시리즈에 대한 설명을 입력하면, NAIS 프리셋 JSON 형식으로 여러 씬의 프롬프트를 생성합니다.")
        series_description = gr.Textbox(label="시리즈 설명 (한국어)", lines=6)
        series_btn = gr.Button("시리즈 JSON 생성", variant="primary")
        series_status = gr.Markdown("")
        series_output = gr.Code(label="결과 JSON", language="json", lines=25)

        series_btn.click(
            generate_multi_scene,
            inputs=[deepseek_key, series_description],
            outputs=[series_output, series_status],
        )

    with gr.Tab("4. 에셋 시스템 / EXIF 분석"):
        gr.Markdown("### 이미지 메타데이터(EXIF/PNG info) 분석")
        with gr.Row():
            with gr.Column():
                exif_image = gr.Image(type="pil", label="이미지 업로드")
                exif_btn = gr.Button("메타데이터 분석")
            with gr.Column():
                exif_prompt = gr.Textbox(label="추출된 프롬프트 (있는 경우)", lines=4)
                exif_raw = gr.Textbox(label="원본 메타데이터", lines=12)

        exif_btn.click(
            analyze_image_metadata,
            inputs=[exif_image],
            outputs=[exif_raw, exif_prompt],
        )

        gr.Markdown("---\n### 이미지 에셋 시스템 (파일명 정의 / NAI 프롬프트 생성 / 에셋 가이드)")
        asset_mode = gr.Radio(
            choices=list(MODE_TRIGGERS.keys()),
            value=list(MODE_TRIGGERS.keys())[1],
            label="모드 선택",
        )
        asset_chars = gr.Textbox(
            label="캐릭터 정의 (선택, 예: a=Alice, b=Bob)", lines=1
        )
        asset_input = gr.Textbox(label="요청 내용 (한국어 가능)", lines=6)
        asset_btn = gr.Button("생성", variant="primary")
        asset_output = gr.Textbox(label="결과", lines=12)

        asset_btn.click(
            generate_asset_output,
            inputs=[deepseek_key, asset_mode, asset_chars, asset_input],
            outputs=[asset_output],
        )


if __name__ == "__main__":
    demo.launch()
