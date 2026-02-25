"""
Multi-user Ollama streaming chat interface with Gradio
- Dynamic add/remove users
- Per-user model selection & temperature
- Streaming responses
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import Generator

import yaml

# ────────────────────────────────────────────────
# Load configuration
# ────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIG_DIR / "configs.yml"

try:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f) or {}
except Exception as e:
    print(f"Failed to load config {CONFIG_PATH}: {e}")
    CONFIG = {}

def cfg(path: str, default=None):
    keys = path.split(".")
    val = CONFIG
    for k in keys:
        val = val.get(k, {})
    return val if val != {} else default


# ────────────────────────────────────────────────
# Config values with sane defaults
# ────────────────────────────────────────────────
OLLAMA_API_BASE     = cfg("ollama.api_base",    "http://localhost:11434")
AVAILABLE_MODELS    = cfg("models.available",   ["llama3.2:3b"])
DEFAULT_MODEL       = cfg("models.default",     AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "llama3.2:3b")
MODEL_DISPLAY_NAMES = cfg("models.display_names", {})

DEFAULT_TEMPERATURE = cfg("generation.default_temperature", 0.75)
TEMP_MIN            = cfg("generation.temperature.min",     0.0)
TEMP_MAX            = cfg("generation.temperature.max",     2.0)
TEMP_STEP           = cfg("generation.temperature.step",    0.05)

DEFAULT_USER_PREFIX = cfg("ui.default_user_name_prefix",    "User")
INITIAL_USERS_COUNT = cfg("ui.initial_users_count",         1)
STREAM_DELAY        = cfg("ui.streaming_animation_delay",   0.012)
MAX_PROMPT_LINES    = cfg("ui.max_prompt_lines",            4)
MAX_OUTPUT_LINES    = cfg("ui.max_output_lines",            12)


def get_model_label(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


# Simple counter for user numbering
user_counter = 0


# ────────────────────────────────────────────────
# Ollama streaming generator
# ────────────────────────────────────────────────
def stream_from_ollama(
    prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Generator[str, None, None]:
    url = f"{OLLAMA_API_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "stream": True,
        "options": {"temperature": max(0.0, min(2.0, temperature))},
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        return
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as e:
        yield f"\n\n**Ollama connection error**\n{str(e)}"
    except Exception as e:
        yield f"\n\n**Unexpected error**\n{str(e)}"


# ────────────────────────────────────────────────
# Create one user session block
# ────────────────────────────────────────────────
def create_user_ui():
    global user_counter
    user_counter += 1

    with gr.Column(elem_classes="user-session") as session_block:
        with gr.Row(equal_height=True):
            name_box = gr.Textbox(
                value=f"{DEFAULT_USER_PREFIX} {user_counter}",
                label="Name",
                max_lines=1,
                scale=3,
            )
            model_dropdown = gr.Dropdown(
                choices=[(get_model_label(m), m) for m in AVAILABLE_MODELS],
                value=DEFAULT_MODEL,
                label="Model",
                interactive=True,
                scale=4,
            )
            temp_slider = gr.Slider(
                minimum=TEMP_MIN,
                maximum=TEMP_MAX,
                step=TEMP_STEP,
                value=DEFAULT_TEMPERATURE,
                label="Temperature",
                scale=3,
            )
            remove_button = gr.Button(
                "× Remove",
                variant="stop",
                size="sm",
                min_width=80,
                scale=1,
            )

        prompt_input = gr.Textbox(
            label="Prompt",
            lines=MAX_PROMPT_LINES,
            placeholder="Type your message… (Shift+Enter to send)",
            show_label=True,
        )

        response_output = gr.Textbox(
            label="Response",
            lines=MAX_OUTPUT_LINES,
            interactive=False,
            show_copy_button=True,
            autoscroll=True,
        )

    # ─── Events ───
    def remove_session():
        session_block.clear()

    remove_button.click(
        fn=remove_session,
        inputs=None,
        outputs=None,
        queue=False,
    )

    def generate_response(prompt, model, temperature):
        if not prompt or not prompt.strip():
            yield "Please type a message."
            return

        yield "▌ Thinking..."

        full_response = ""
        for chunk in stream_from_ollama(prompt, model, temperature):
            full_response += chunk
            yield full_response + "▌"
            time.sleep(STREAM_DELAY)

        yield full_response

    prompt_input.submit(
        fn=generate_response,
        inputs=[prompt_input, model_dropdown, temp_slider],
        outputs=response_output,
        queue=True,
    )

    return session_block


# ────────────────────────────────────────────────
# Gradio Interface
# ────────────────────────────────────────────────
CSS = """
.user-session {
    margin: 1.4rem 0;
    padding: 1.3rem;
    border: 1px solid #555;
    border-radius: 10px;
    background: #111;
}
.add-button {
    margin: 1rem 0 1.5rem;
}
"""

with gr.Blocks(title="Multi-User Ollama Chat") as demo:   # ← theme removed here

    gr.Markdown(
        "# Multi-User Ollama Streaming Chat\n\n"
        f"**Available models:** {', '.join(get_model_label(m) for m in AVAILABLE_MODELS)}"
    )

    add_new = gr.Button(
        "➕ Add New User",
        variant="primary",
        elem_classes="add-button",
        size="lg",
    )

    users_container = gr.Column()

    # Initialize with configured number of users
    with users_container:
        for _ in range(INITIAL_USERS_COUNT):
            create_user_ui()

    add_new.click(
        fn=create_user_ui,
        inputs=None,
        outputs=users_container,
        queue=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    demo.queue(max_size=20).launch(
        # server_name="0.0.0.0",
        # server_port=7860,
        share=args.share,               # ← can be set via python app.py --share
        # theme=gr.themes.Default(),
        # css=CSS,
    )