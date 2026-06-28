import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

import core
import embedding_client
import llm_client
import presets as preset_store
from tag_db import TagDB

with gr.Blocks(title="Prompt Generator") as demo:
    gr.Markdown("# Prompt Generator")

    db_state = gr.State(TagDB())
    history_state = gr.State([])

    with gr.Accordion("⚙️ 설정", open=True):
        # Provider
        provider_radio = gr.Radio(
            choices=list(llm_client.PROVIDERS.keys()),
            value=list(llm_client.PROVIDERS.keys())[0],
            label="프로바이더",
        )
        with gr.Row():
            api_key = gr.Textbox(
                label="API Key (로컬 서버는 빈값 가능, 저장됨)",
                type="password",
                placeholder="sk-... 또는 비워두기",
                scale=2,
            )
            base_url = gr.Textbox(
                label="서버 URL",
                value=llm_client.DEFAULT_BASE_URL,
                scale=3,
            )
            model_select = gr.Dropdown(
                label="모델",
                choices=llm_client.AVAILABLE_MODELS,
                value=llm_client.DEFAULT_MODEL,
                allow_custom_value=True,
                scale=2,
            )

        def _on_provider_change(provider):
            info = llm_client.PROVIDERS[provider]
            models = info["models"]
            saved_key = core.get_api_key_for_provider(provider)
            core.save_provider(provider)
            return info["base_url"], gr.update(choices=models, value=models[0]), saved_key

        provider_radio.change(
            _on_provider_change,
            inputs=provider_radio,
            outputs=[base_url, model_select, api_key],
        )

        with gr.Row():
            tag_db_file = gr.File(
                label="단부루 태그 CSV (선택, 저장됨)", file_types=[".csv"], scale=3
            )
            tag_db_status = gr.Markdown("태그 DB 없음", scale=2)

        gr.Markdown("##### 임베딩 (태그 DB 검색용, 항상 로컬 시스템으로 호출됨)")
        with gr.Row():
            embedding_base_url = gr.Textbox(
                label="임베딩 서버 URL (저장됨)",
                value=embedding_client.DEFAULT_EMBEDDING_BASE_URL,
                scale=3,
            )
            embedding_api_key = gr.Textbox(
                label="임베딩 API Key (선택, 저장됨)",
                type="password",
                placeholder="로컬 서버는 빈값 가능",
                scale=2,
            )
            embedding_model_select = gr.Dropdown(
                label="임베딩 모델 (저장됨)",
                choices=embedding_client.DEFAULT_EMBEDDING_MODELS,
                value=embedding_client.DEFAULT_EMBEDDING_MODEL,
                allow_custom_value=True,
                scale=2,
            )
            embedding_models_refresh_btn = gr.Button("모델 목록 조회", scale=1)
        embedding_models_status = gr.Markdown("")

        standing_notes = gr.Textbox(
            label="고정 지시사항 (모든 생성에 포함, 저장됨)",
            placeholder="예: faceless male은 얼굴 태그 제외. 배경 항상 실내.",
            lines=3,
        )
        extra_system_prompt = gr.Textbox(
            label="추가 시스템 프롬프트 (모든 AI 호출의 시스템 메시지 끝에 추가됨, 저장됨)",
            placeholder="예: 항상 태그를 50개 이상 출력할 것. 배경 태그는 반드시 포함.",
            lines=3,
        )
        with gr.Row():
            accumulate_context = gr.Checkbox(
                label="컨텍스트 누적 (이전 대화 기억)", value=False
            )
            clear_history_btn = gr.Button("대화 기록 초기화", size="sm")
            history_status = gr.Markdown("")

        demo.load(
            core.load_saved_state,
            inputs=None,
            outputs=[api_key, db_state, tag_db_status, standing_notes, base_url, extra_system_prompt, provider_radio,
                     embedding_base_url, embedding_model_select, embedding_api_key],
        )
        api_key.change(core.save_api_key_for_provider, inputs=[api_key, provider_radio])
        base_url.change(core.save_base_url, inputs=base_url)
        embedding_base_url.change(core.save_embedding_base_url, inputs=embedding_base_url)
        embedding_model_select.change(core.save_embedding_model, inputs=embedding_model_select)
        embedding_api_key.change(core.save_embedding_api_key, inputs=embedding_api_key)
        embedding_models_refresh_btn.click(
            core.list_embedding_models,
            inputs=[embedding_api_key, embedding_base_url],
            outputs=[embedding_model_select, embedding_models_status],
        )
        tag_db_file.change(core.load_tag_db, inputs=tag_db_file, outputs=[db_state, tag_db_status])
        standing_notes.change(core.save_notes, inputs=standing_notes)
        extra_system_prompt.change(core.save_extra_system_prompt, inputs=extra_system_prompt)

        def _clear_history():
            return [], "대화 기록 초기화됨"

        clear_history_btn.click(_clear_history, outputs=[history_state, history_status])
        accumulate_context.change(
            lambda v: "컨텍스트 누적 ON" if v else "컨텍스트 누적 OFF",
            inputs=accumulate_context, outputs=history_status,
        )

    with gr.Tab("1. 태그 조합 생성"):
        gr.Markdown("자연어로 원하는 이미지를 설명하면 AI가 태그를 조합합니다.")
        combo_request = gr.Textbox(label="요청 내용 (한국어 가능)", lines=4)
        combo_variant_count = gr.Slider(1, 5, value=1, step=1, label="생성 개수 (variants)")
        combo_btn = gr.Button("태그 조합 생성", variant="primary")
        combo_tags = gr.Textbox(label="결과 태그 (복사해서 사용)", lines=6)
        combo_explanation = gr.Textbox(label="설명 (한국어)", lines=4)
        combo_debug = gr.Textbox(label="검색된 후보 태그 (디버그)", lines=10)

        combo_btn.click(
            core.generate_tag_combo,
            inputs=[api_key, combo_request, db_state, combo_variant_count, standing_notes,
                    history_state, accumulate_context, model_select, base_url, extra_system_prompt,
                    embedding_base_url, embedding_model_select, embedding_api_key],
            outputs=[combo_tags, combo_explanation, combo_debug, history_state],
        )

        gr.Markdown("---\n#### 태그 프리셋 저장/불러오기")
        with gr.Row():
            preset_name = gr.Textbox(label="프리셋 이름", scale=2)
            preset_save_btn = gr.Button("현재 결과 태그 저장", scale=1)
        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=list(preset_store.load_presets().keys()),
                label="저장된 프리셋", scale=2,
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
            return gr.update(choices=list(new_presets.keys()), value=name.strip()), f"'{name.strip()}' 저장 완료"

        def _load_preset(name):
            if not name:
                return "", "프리셋을 선택해주세요."
            presets = preset_store.load_presets()
            return presets.get(name, ""), f"'{name}' 불러옴"

        def _delete_preset(name):
            if not name:
                return gr.update(), "삭제할 프리셋을 선택해주세요."
            new_presets = preset_store.delete_preset(name)
            return gr.update(choices=list(new_presets.keys()), value=None), f"'{name}' 삭제됨"

        preset_save_btn.click(_save_preset, inputs=[preset_name, combo_tags], outputs=[preset_dropdown, preset_status])
        preset_load_btn.click(_load_preset, inputs=[preset_dropdown], outputs=[combo_tags, preset_status])
        preset_delete_btn.click(_delete_preset, inputs=[preset_dropdown], outputs=[preset_dropdown, preset_status])

    with gr.Tab("2. 다중 씬(시리즈) 생성"):
        gr.Markdown("시리즈 설명 → NAIS 프리셋 JSON + 씬별 설명 (스트리밍)")
        series_chars = gr.Textbox(label="캐릭터/카테고리 정의 (예: a=Alice, b=Bob)", lines=1)
        series_description = gr.Textbox(label="시리즈 설명 (한국어)", lines=6)
        series_btn = gr.Button("시리즈 JSON 생성", variant="primary")
        series_status = gr.Markdown("")
        series_output = gr.Code(label="결과 JSON (NAI 프리셋에 붙여넣기)", language="json", lines=25)
        series_download = gr.DownloadButton(label="JSON 파일 다운로드", visible=False)
        series_descriptions = gr.Textbox(label="씬별 설명 (한국어)", lines=10)
        series_debug = gr.Textbox(label="디버그 (후보 태그 / AI 원본 응답)", lines=15)

        series_btn.click(
            core.generate_multi_scene,
            inputs=[api_key, series_description, series_chars, db_state, standing_notes,
                    history_state, accumulate_context, model_select, base_url, extra_system_prompt,
                    embedding_base_url, embedding_model_select, embedding_api_key],
            outputs=[series_output, series_descriptions, series_status, series_debug, history_state],
        ).then(
            core.prepare_json_download,
            inputs=[series_output],
            outputs=[series_download],
        )

    with gr.Tab("3. 이미지 에셋 시스템"):
        asset_mode = gr.Radio(
            choices=list(core.MODE_TRIGGERS.keys()),
            value=list(core.MODE_TRIGGERS.keys())[1],
            label="모드 선택",
        )
        asset_chars = gr.Textbox(label="캐릭터 정의 (예: a=Alice, b=Bob)", lines=1)
        asset_input = gr.Textbox(label="요청 내용 (한국어 가능)", lines=6)
        asset_btn = gr.Button("생성", variant="primary")
        asset_output = gr.Textbox(label="결과", lines=12)

        asset_btn.click(
            core.generate_asset_output,
            inputs=[api_key, asset_mode, asset_chars, asset_input, history_state, accumulate_context,
                    model_select, base_url, extra_system_prompt],
            outputs=[asset_output, history_state],
        )

    with gr.Tab("4. AI 대화 / 편집"):
        gr.Markdown(
            "프롬프트나 JSON을 베이스로 붙여넣고 AI와 자유롭게 대화·수정하세요.\n"
            "베이스 없이도 일반 질의/논의가 가능합니다."
        )
        with gr.Row():
            with gr.Column(scale=1):
                base_content = gr.Textbox(
                    label="베이스 콘텐츠 (기존 태그 프롬프트 또는 JSON, 선택사항)",
                    lines=18,
                    placeholder="여기에 기존 프롬프트나 NAIS JSON을 붙여넣으면 AI가 참고합니다.",
                )
                clear_base_btn = gr.Button("베이스 초기화", size="sm")
            with gr.Column(scale=2):
                chat_display = gr.Chatbot(label="대화", height=450)
                chat_input = gr.Textbox(
                    label="메시지 (Enter로 전송)",
                    placeholder="예: 이 JSON에 씬 3개 추가해줘 / 이 프롬프트를 더 자세하게 만들어줘",
                    lines=2,
                )
                with gr.Row():
                    chat_send_btn = gr.Button("전송", variant="primary", scale=3)
                    chat_clear_btn = gr.Button("대화 초기화", scale=1)

        chat_response = gr.Textbox(label="최근 AI 응답 (복사용)", lines=8)

        def _chat_send(api_key_val, base_url_val, model_val, user_msg, base_cont,
                       notes_val, extra_prompt_val, chatbot_history, history_val, accumulate_val):
            if not user_msg or not user_msg.strip():
                yield chatbot_history, "", history_val
                return

            new_display = list(chatbot_history) + [{"role": "user", "content": user_msg}]
            yield new_display, "", history_val

            response = ""
            new_history = history_val
            for response, new_history in core.chat_with_context(
                api_key_val, base_url_val, model_val, user_msg, base_cont,
                notes_val, history_val, accumulate_val, extra_prompt_val,
            ):
                display = list(new_display) + [{"role": "assistant", "content": response}]
                yield display, response, new_history

        chat_send_btn.click(
            _chat_send,
            inputs=[api_key, base_url, model_select, chat_input, base_content,
                    standing_notes, extra_system_prompt, chat_display, history_state, accumulate_context],
            outputs=[chat_display, chat_response, history_state],
        ).then(lambda: "", outputs=chat_input)

        chat_input.submit(
            _chat_send,
            inputs=[api_key, base_url, model_select, chat_input, base_content,
                    standing_notes, extra_system_prompt, chat_display, history_state, accumulate_context],
            outputs=[chat_display, chat_response, history_state],
        ).then(lambda: "", outputs=chat_input)

        chat_clear_btn.click(lambda: ([], ""), outputs=[chat_display, chat_response])
        clear_base_btn.click(lambda: "", outputs=base_content)

    with gr.Tab("5. JSON 통합"):
        gr.Markdown("NAIS 프리셋 JSON 여러 개를 업로드하면 씬들을 하나의 프리셋으로 합쳐줍니다. (AI 호출 없음)")
        merge_files = gr.File(label="통합할 JSON 파일들", file_count="multiple", file_types=[".json"])
        merge_name = gr.Textbox(label="통합 결과 이름 (선택, 비우면 자동 생성)")
        merge_btn = gr.Button("통합", variant="primary")
        merge_status = gr.Markdown("")
        merge_output = gr.Code(label="통합된 JSON", language="json", lines=25)
        merge_download = gr.DownloadButton(label="JSON 파일 다운로드", visible=False)

        merge_btn.click(
            core.merge_json_presets,
            inputs=[merge_files, merge_name],
            outputs=[merge_output, merge_status],
        ).then(
            core.prepare_json_download,
            inputs=[merge_output],
            outputs=[merge_download],
        )


if __name__ == "__main__":
    demo.launch(server_port=7861)
