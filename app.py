"""
Multi-user Ollama streaming chat with Gradio ≥5.x
Features:
- Dynamic add/remove users
- Per-user model + temperature
- Streaming responses with typewriter effect
- Better state/component management
"""

import gradio as gr
import requests
import json
import time
from pathlib import Path
from typing import Generator, Dict, Any
import yaml

# ─── Configuration ──────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILE = CONFIG_DIR / "config.yml"   # ← more conventional name

DEFAULT_CONFIG = {
    "ollama": {
        "api_base": "http://localhost:11434"
    },
    "models": {
        "available": ["llama3.2:3b", "phi3:mini", "gemma2:2b"],
        "default": "llama3.2:3b",
        "display_names": {}
    },
    "generation": {
        "default_temperature": 0.75,
        "temperature": {"min": 0.0, "max": 2.0, "step": 0.05}
    },
    "ui": {
        "default_user_name_prefix": "User",
        "initial_users_count": 2,
        "streaming_animation_delay": 0.012,
        "max_prompt_lines": 4,
        "max_output_lines": 14
    }
}

def load_config() -> dict:
    if not CONFIG_FILE.is_file():
        return DEFAULT_CONFIG
    try:
        with CONFIG_FILE.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    except Exception as e:
        print(f"Config load error: {e}")
        return DEFAULT_CONFIG


CONFIG = load_config()

# Convenience getters
def cfg(*keys, default=None):
    val = CONFIG
    for k in keys:
        val = val.get(k, {})
    return val if val != {} else default


OLLAMA_API_BASE     = cfg("ollama", "api_base", default="http://localhost:11434")
AVAILABLE_MODELS    = cfg("models", "available", default=["llama3.2:3b"])
DEFAULT_MODEL       = cfg("models", "default", default=AVAILABLE_MODELS[0])
MODEL_DISPLAY_NAMES = cfg("models", "display_names", default={})
DEFAULT_TEMPERATURE = cfg("generation", "default_temperature", default=0.75)
TEMP_RANGE          = cfg("generation", "temperature", default={})
TEMP_MIN            = TEMP_RANGE.get("min", 0.0)
TEMP_MAX            = TEMP_RANGE.get("max", 2.0)
TEMP_STEP           = TEMP_RANGE.get("step", 0.05)
USER_PREFIX         = cfg("ui", "default_user_name_prefix", default="User")
INITIAL_USERS       = cfg("ui", "initial_users_count", default=2)
STREAM_DELAY        = cfg("ui", "streaming_animation_delay", default=0.015)
MAX_PROMPT_LINES    = cfg("ui", "max_prompt_lines", default=4)
MAX_OUTPUT_LINES    = cfg("ui", "max_output_lines", default=14)


def model_label(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


# ─── Ollama Streaming ───────────────────────────────────────────────────────────

def stream_ollama(
    prompt: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE
) -> Generator[str, None, None]:
    url = f"{OLLAMA_API_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "stream": True,
        "options": {"temperature": float(temperature)},
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=200) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        return
                except json.JSONDecodeError:
                    continue
    except requests.RequestException as e:
        yield f"\n\n**Ollama connection failed**\n{str(e)}"
    except Exception as e:
        yield f"\n\n**Unexpected error**\n{str(e)}"


# ─── UI Building Blocks ─────────────────────────────────────────────────────────

def create_user_ui(idx: int, user_data: dict) -> Dict[str, Any]:
    name = user_data.get("name", f"{USER_PREFIX} {idx+1}")
    model = user_data.get("model", DEFAULT_MODEL)
    temp = user_data.get("temp", DEFAULT_TEMPERATURE)

    with gr.Column(elem_classes="user-session") as column:
        with gr.Row(equal_height=True, variant="panel"):
            name_box = gr.Textbox(
                value=name,
                label="Name",
                max_lines=1,
                scale=3
            )
            model_dd = gr.Dropdown(
                choices=[(model_label(m), m) for m in AVAILABLE_MODELS],
                value=model,
                label="Model",
                scale=4
            )
            temp_slider = gr.Slider(
                minimum=TEMP_MIN,
                maximum=TEMP_MAX,
                step=TEMP_STEP,
                value=temp,
                label="Temp",
                scale=3
            )
            remove_btn = gr.Button(
                "× Remove",
                variant="stop",
                size="sm",
                min_width=80,
                scale=1
            )

        prompt_tb = gr.Textbox(
            label="Prompt",
            lines=MAX_PROMPT_LINES,
            placeholder="Type message… (Shift+Enter to send)",
            autofocus=idx == 0,
        )

        response_tb = gr.Textbox(
            label="Response",
            lines=MAX_OUTPUT_LINES,
            interactive=False,
            show_copy_button=True,
            autoscroll=True,
            max_lines=MAX_OUTPUT_LINES + 4,
        )

    return {
        "column": column,
        "remove_btn": remove_btn,
        "name": name_box,
        "model": model_dd,
        "temp": temp_slider,
        "prompt": prompt_tb,
        "response": response_tb,
    }


# ─── Styling ────────────────────────────────────────────────────────────────────

CSS = """
.user-session {
    margin: 1.4rem 0;
    padding: 1.3rem 1.5rem;
    border: 1px solid #444;
    border-radius: 10px;
    background: #0d0d0d;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
"""

# ─── Main Interface ─────────────────────────────────────────────────────────────

with gr.Blocks(title="Multi-User Ollama Chat", css=CSS, theme=gr.themes.Soft()) as app:

    gr.Markdown(
        f"""
        # Multi-User Ollama Chat
        
        **Models**: {", ".join(model_label(m) for m in AVAILABLE_MODELS)}
        """
    )

    users_state = gr.State([])          # list of dicts: [{"name":…, "model":…, "temp":…}]
    add_btn = gr.Button("➕ Add User", variant="primary", size="lg")

    users_container = gr.Column()

    @gr.render(inputs=users_state)
    def render_all_users(users_list):
        # No .clear() needed — Gradio already cleared the container

        if not users_list:
            with users_container:
                gr.Markdown("**No users yet** — add one to start chatting.")
            return

        with users_container:
            for i, user in enumerate(users_list):
                comps = create_user_ui(i, user)

                # ── Remove user ─────────────────────────────────────
                def make_remove_fn(index=i):
                    def remove_current():
                        lst = users_state.value[:]
                        if 0 <= index < len(lst):
                            lst.pop(index)
                        return lst
                    return remove_current

                comps["remove_btn"].click(
                    make_remove_fn(),
                    outputs=users_state,
                    queue=False
                )

                # ── Generate response ───────────────────────────────
                def make_generate_fn(index=i):
                    def generate(prompt, model, temp):
                        if not prompt or not prompt.strip():
                            yield "Please type something…"
                            return

                        yield "▌ Thinking..."

                        full = ""
                        for token in stream_ollama(prompt, model, float(temp)):
                            full += token
                            yield full + "▌"
                            time.sleep(STREAM_DELAY)
                        yield full

                    return generate

                comps["prompt"].submit(
                    make_generate_fn(),
                    inputs=[comps["prompt"], comps["model"], comps["temp"]],
                    outputs=comps["response"],
                    queue=True
                )

                # ── Save settings live ──────────────────────────────
                def make_update_fn(index=i):
                    def update_settings(name, model, temp):
                        lst = users_state.value[:]
                        if 0 <= index < len(lst):
                            lst[index].update({"name": name, "model": model, "temp": temp})
                        return lst
                    return update_settings

                for field in [comps["name"], comps["model"], comps["temp"]]:
                    field.change(
                        make_update_fn(),
                        inputs=[comps["name"], comps["model"], comps["temp"]],
                        outputs=users_state,
                        queue=False
                    )

    # ── Add user ────────────────────────────────────────────────────────────────
    def add_user(current):
        lst = current[:] if current is not None else []
        lst.append({
            "name": f"{USER_PREFIX} {len(lst)+1}",
            "model": DEFAULT_MODEL,
            "temp": DEFAULT_TEMPERATURE,
        })
        return lst

    add_btn.click(add_user, inputs=users_state, outputs=users_state, queue=False)

    # ── Initial users ───────────────────────────────────────────────────────────
    app.load(
        lambda: [
            {"name": f"{USER_PREFIX} {i+1}", "model": DEFAULT_MODEL, "temp": DEFAULT_TEMPERATURE}
            for i in range(INITIAL_USERS)
        ],
        outputs=users_state,
        queue=False
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app.queue(max_size=20).launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=7860,
        # inbrowser=True,           # optional
    )