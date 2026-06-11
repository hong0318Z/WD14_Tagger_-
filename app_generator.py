import gradio as gr

import core
import deepseek_client
import presets as preset_store
from tag_db import TagDB

with gr.Blocks(title="DeepSeek Prompt Generator") as demo:
    gr.Markdown("# DeepSeek Prompt Generator (태그 조합 / 다중 씬 / 에셋 시스템)")
    gr.Markdown(
        f"모델: `{deepseek_client.MODEL}` · "
        f"최대 컨텍스트: {deepseek_client.MAX_CONTEXT_TOKENS:,} 토큰 · "
        f"최대 응답: {deepseek_client.MAX_TOKENS:,} 토큰"
    )

    db_state = gr.State(TagDB())
    history_state = gr.State([])

    with gr.Accordion("설정", open=True):
        deepseek_key = gr.Textbox(
            label="DeepSeek API Key (서버에 저장됨, data/local_config.json)",
            type="password",
            placeholder="sk-...",
        )
        tag_db_file = gr.File(label="단부루 태그 CSV 업로드 (선택, 서버에 저장되어 재시작 후에도 유지됨)", file_types=[".csv"])
        tag_db_status = gr.Markdown("태그 DB가 로드되지 않았습니다. (선택 사항)")
        standing_notes = gr.Textbox(
            label="고정 지시사항 / 메모 (모든 생성에 항상 함께 전달됨, 서버에 저장되어 유지됨)",
            placeholder="예: faceless male은 항상 얼굴 태그를 넣지 말 것. 배경은 항상 실내로.",
            lines=4,
        )
        accumulate_context = gr.Checkbox(
            label="컨텍스트 누적 (이전 대화 기억) — 끄면 매번 새 응답으로 생성",
            value=False,
        )
        history_status = gr.Markdown("")
        clear_history_btn = gr.Button("대화 기록 초기화")

        demo.load(
            core.load_saved_state,
            inputs=None,
            outputs=[deepseek_key, db_state, tag_db_status, standing_notes],
        )
        deepseek_key.change(core.save_api_key, inputs=deepseek_key, outputs=None)
        tag_db_file.change(core.load_tag_db, inputs=tag_db_file, outputs=[db_state, tag_db_status])
        standing_notes.change(core.save_notes, inputs=standing_notes, outputs=None)

        def _clear_history():
            return [], "대화 기록이 초기화되었습니다."

        clear_history_btn.click(_clear_history, inputs=None, outputs=[history_state, history_status])

        def _toggle_note(checked):
            if checked:
                return "컨텍스트 누적 ON: 이전 요청/응답을 기억하고 이어서 생성합니다."
            return "컨텍스트 누적 OFF: 매번 새로운 대화로 생성합니다."

        accumulate_context.change(_toggle_note, inputs=accumulate_context, outputs=history_status)

    with gr.Tab("1. 태그 조합 생성"):
        gr.Markdown("자연어로 원하는 이미지를 설명하면, 태그 DB에서 관련 태그를 찾아 AI가 조합해줍니다.")
        combo_request = gr.Textbox(label="요청 내용 (한국어 가능)", lines=4)
        combo_variant_count = gr.Slider(1, 5, value=1, step=1, label="생성 개수 (variants)")
        combo_btn = gr.Button("태그 조합 생성", variant="primary")
        combo_tags = gr.Textbox(label="결과 태그 (영어, {그룹} 단위로 구분됨, 복사해서 사용)", lines=6)
        combo_explanation = gr.Textbox(label="설명 (한국어)", lines=4)
        combo_debug = gr.Textbox(label="검색된 후보 태그 (디버그)", lines=10)

        combo_btn.click(
            core.generate_tag_combo,
            inputs=[deepseek_key, combo_request, db_state, combo_variant_count, standing_notes,
                    history_state, accumulate_context],
            outputs=[combo_tags, combo_explanation, combo_debug, history_state],
        )

        gr.Markdown("---\n#### 태그 프리셋 저장/불러오기")
        with gr.Row():
            preset_name = gr.Textbox(label="프리셋 이름", scale=2)
            preset_save_btn = gr.Button("현재 결과 태그를 프리셋으로 저장", scale=1)
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(preset_store.load_presets().keys()),
                label="저장된 프리셋",
                scale=2,
            )
            preset_load_btn = gr.Button("불러오기", scale=1)
            preset_delete_btn = gr.Button("삭제", scale=1)
        preset_status = gr.Markdown("")

        def _save_preset(name, tags):
            if not name or not name.strip():
                return gr.update(), "프리셋 이름을 입력해주세요."
            if not tags or not tags.strip():
                return gr.update(), "저장할 태그가 없습니다."
            new_presets = preset_store.save_preset(name, tags)
            return gr.update(choices=list(new_presets.keys()), value=name.strip()), f"'{name.strip()}' 프리셋 저장 완료"

        def _load_preset(name):
            if not name:
                return "", "프리셋을 선택해주세요."
            presets = preset_store.load_presets()
            return presets.get(name, ""), f"'{name}' 프리셋 불러옴"

        def _delete_preset(name):
            if not name:
                return gr.update(), "삭제할 프리셋을 선택해주세요."
            new_presets = preset_store.delete_preset(name)
            return gr.update(choices=list(new_presets.keys()), value=None), f"'{name}' 프리셋 삭제됨"

        preset_save_btn.click(
            _save_preset, inputs=[preset_name, combo_tags], outputs=[preset_dropdown, preset_status]
        )
        preset_load_btn.click(
            _load_preset, inputs=[preset_dropdown], outputs=[combo_tags, preset_status]
        )
        preset_delete_btn.click(
            _delete_preset, inputs=[preset_dropdown], outputs=[preset_dropdown, preset_status]
        )

    with gr.Tab("2. 다중 씬(시리즈) 생성"):
        gr.Markdown("시리즈에 대한 설명을 입력하면, NAIS 프리셋 JSON 형식으로 여러 씬의 프롬프트를 생성합니다.")
        series_chars = gr.Textbox(
            label="캐릭터/카테고리 정의 (선택, 예: a=Alice, b=Bob, 카테고리는 기본값 사용)", lines=1
        )
        series_description = gr.Textbox(label="시리즈 설명 (한국어)", lines=6)
        series_btn = gr.Button("시리즈 JSON 생성", variant="primary")
        series_status = gr.Markdown("")
        series_output = gr.Code(label="결과 JSON (NAI 프리셋에 그대로 붙여넣기)", language="json", lines=25)
        series_descriptions = gr.Textbox(label="씬별 설명 (한국어, 별도 메모용)", lines=10)
        series_debug = gr.Textbox(label="디버그 (검색된 후보 태그 / AI 원본 응답)", lines=15)

        series_btn.click(
            core.generate_multi_scene,
            inputs=[deepseek_key, series_description, series_chars, db_state, standing_notes,
                    history_state, accumulate_context],
            outputs=[series_output, series_descriptions, series_status, series_debug, history_state],
        )

    with gr.Tab("3. 이미지 에셋 시스템"):
        gr.Markdown("파일명 정의 / NAI 프롬프트 생성 / 에셋 가이드 작성")
        asset_mode = gr.Radio(
            choices=list(core.MODE_TRIGGERS.keys()),
            value=list(core.MODE_TRIGGERS.keys())[1],
            label="모드 선택",
        )
        asset_chars = gr.Textbox(
            label="캐릭터 정의 (선택, 예: a=Alice, b=Bob)", lines=1
        )
        asset_input = gr.Textbox(label="요청 내용 (한국어 가능)", lines=6)
        asset_btn = gr.Button("생성", variant="primary")
        asset_output = gr.Textbox(label="결과", lines=12)

        asset_btn.click(
            core.generate_asset_output,
            inputs=[deepseek_key, asset_mode, asset_chars, asset_input, history_state, accumulate_context],
            outputs=[asset_output, history_state],
        )


if __name__ == "__main__":
    demo.launch(server_port=7861)
