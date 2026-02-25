"""
Multi-user Ollama streaming chat interface with Gradio 5.x
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
# Load configuration (unchanged)
# ────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_PATH = CONFIG_DIR / "config.yml"
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

OLLAMA_API_BASE     = cfg("ollama.api_base",    "http://localhost:11434")
AVAILABLE_MODELS    = cfg("models.available",   ["llama3.2:3b", "phi3:mini", "gemma2:2b"])
DEFAULT_MODEL       = cfg("models.default",     AVAILABLE_MODELS[0] if AVAILABLE_MODELS else "llama3.2:3b")
MODEL_DISPLAY_NAMES = cfg("models.display_names", {})
DEFAULT_TEMPERATURE = cfg("generation.default_temperature", 0.75)
TEMP_MIN            = cfg("generation.temperature.min",     0.0)
TEMP_MAX            = cfg("generation.temperature.max",     2.0)
TEMP_STEP           = cfg("generation.temperature.step",    0.05)
DEFAULT_USER_PREFIX = cfg("ui.default_user_name_prefix",    "User")
INITIAL_USERS_COUNT = cfg("ui.initial_users_count",         2)
STREAM_DELAY        = cfg("ui.streaming_animation_delay",   0.015)
MAX_PROMPT_LINES    = cfg("ui.max_prompt_lines",            4)
MAX_OUTPUT_LINES    = cfg("ui.max_output_lines",            14)

def get_model_label(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)

# ────────────────────────────────────────────────
# Ollama streaming (unchanged)
# ────────────────────────────────────────────────
def stream_from_ollama(prompt: str, model: str, temperature: float = DEFAULT_TEMPERATURE) -> Generator[str, None, None]:
    url = f"{OLLAMA_API_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "stream": True,
        "options": {"temperature": max(0.0, min(2.0, temperature))},
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line: continue
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data: yield data["response"]
                    if data.get("done"): return
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as e:
        yield f"\n\n**Ollama connection error**\n{str(e)}"
    except Exception as e:
        yield f"\n\n**Unexpected error**\n{str(e)}"

# ────────────────────────────────────────────────
# Single user UI creator (unchanged)
# ────────────────────────────────────────────────
def create_single_user_ui(idx: int, user_data: dict):
    name = user_data.get("name", f"{DEFAULT_USER_PREFIX} {idx+1}")
    model = user_data.get("model", DEFAULT_MODEL)
    temp = user_data.get("temp", DEFAULT_TEMPERATURE)

    with gr.Column(elem_classes="user-session") as col:
        with gr.Row(equal_height=True):
            name_box = gr.Textbox(value=name, label="Name", max_lines=1, scale=3)
            model_dropdown = gr.Dropdown(
                choices=[(get_model_label(m), m) for m in AVAILABLE_MODELS],
                value=model, label="Model", interactive=True, scale=4,
            )
            temp_slider = gr.Slider(
                minimum=TEMP_MIN, maximum=TEMP_MAX, step=TEMP_STEP,
                value=temp, label="Temperature", scale=3,
            )
            remove_btn = gr.Button("× Remove", variant="stop", size="sm", min_width=80, scale=1)

        prompt_input = gr.Textbox(
            label="Prompt", lines=MAX_PROMPT_LINES,
            placeholder="Type your message… (Shift+Enter to send)", show_label=True,
        )
        response_output = gr.Textbox(
            label="Response", lines=MAX_OUTPUT_LINES, interactive=False,
            show_copy_button=True, autoscroll=True, max_lines=MAX_OUTPUT_LINES + 2,
        )

    return {
        "col": col, "remove_btn": remove_btn, "name": name_box,
        "model": model_dropdown, "temp": temp_slider,
        "prompt": prompt_input, "response": response_output,
    }

# ────────────────────────────────────────────────
# CSS (unchanged)
# ────────────────────────────────────────────────
CSS = """
.user-session {
    margin: 1.5rem 0;
    padding: 1.4rem;
    border: 1px solid #555;
    border-radius: 12px;
    background: #0f0f0f;
}
.add-button { margin: 1.2rem 0 1.8rem; }
"""

# ────────────────────────────────────────────────
# Main app
# ────────────────────────────────────────────────
with gr.Blocks(title="Multi-User Ollama Chat", css=CSS, theme=gr.themes.Soft()) as demo:
    users_state = gr.State([], render=False)   # ← Fix: inside Blocks + render=False

    gr.Markdown(
        "# Multi-User Ollama Streaming Chat\n\n"
        f"**Available models:** {', '.join(get_model_label(m) for m in AVAILABLE_MODELS)}"
    )

    with gr.Row():
        add_btn = gr.Button(
            "➕ Add New User", variant="primary", size="lg",
            elem_classes="add-button", scale=1,
        )

    users_container = gr.Column()

    with users_container:
        @gr.render(inputs=users_state)
        def render_users(users):
            if not users:
                gr.Markdown("**No users yet** — click **Add New User** to begin.")
                return

            for i, user in enumerate(users):
                comps = create_single_user_ui(i, user)

                def make_remove_handler(remove_idx=i):
                    def remove():
                        current = users_state.value.copy()
                        if 0 <= remove_idx < len(current):
                            current.pop(remove_idx)
                        return current
                    return remove

                comps["remove_btn"].click(
                    make_remove_handler(), outputs=users_state, queue=False
                )

                def make_generate_handler(gen_idx=i):
                    def generate(prompt, model, temp):
                        if not prompt or not prompt.strip():
                            yield "Please enter a message."
                            return
                        yield "▌ Thinking..."
                        full_response = ""
                        for chunk in stream_from_ollama(prompt, model, float(temp)):
                            full_response += chunk
                            yield full_response + "▌"
                            time.sleep(STREAM_DELAY)
                        yield full_response
                    return generate

                comps["prompt"].submit(
                    make_generate_handler(),
                    inputs=[comps["prompt"], comps["model"], comps["temp"]],
                    outputs=comps["response"],
                    queue=True,
                )

                def make_state_updater(up_idx=i):
                    def update(name, model, temp):
                        lst = users_state.value.copy()
                        if 0 <= up_idx < len(lst):
                            lst[up_idx].update({"name": name, "model": model, "temp": temp})
                        return lst
                    return update

                for comp in [comps["name"], comps["model"], comps["temp"]]:
                    comp.change(
                        make_state_updater(),
                        inputs=[comps["name"], comps["model"], comps["temp"]],
                        outputs=users_state,
                        queue=False,
                    )

    def add_new_user():
        current = users_state.value.copy()
        new_user = {
            "name": f"{DEFAULT_USER_PREFIX} {len(current) + 1}",
            "model": DEFAULT_MODEL,
            "temp": DEFAULT_TEMPERATURE,
        }
        current.append(new_user)
        return current

    add_btn.click(add_new_user, outputs=users_state, queue=False)

    demo.load(
        lambda: [
            {"name": f"{DEFAULT_USER_PREFIX} {i+1}", "model": DEFAULT_MODEL, "temp": DEFAULT_TEMPERATURE}
            for i in range(INITIAL_USERS_COUNT)
        ],
        outputs=users_state,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-user Ollama chat")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()
    demo.queue(max_size=20).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=7860,
    )