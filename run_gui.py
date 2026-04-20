"""GUI 실행 스크립트"""

from gui.gradio_ui import create_ui

if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",  # 0.0.0.0 → 127.0.0.1
        server_port=7870,
        share=False,
        inbrowser=True,
    )