import gradio as gr
import requests
import json
import time
from typing import Dict, List

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
DEFAULT_MODEL = "llama3.2"
DEFAULT_TEMP = 0.7

# We'll store user sessions here
users: Dict[str, Dict] = {}           # id → {"name": str, "output": str, "component": gr.Blocks or None}
next_user_id = 1


def ollama_stream(prompt: str, model: str = DEFAULT_MODEL, temp: float = DEFAULT_TEMP):
    """Generator that yields tokens from Ollama"""
    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temp}
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=90) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            yield None   # signal end
                            break
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n\nError: {str(e)}"


def add_new_user():
    global next_user_id

    user_id = f"user_{next_user_id}"
    next_user_id += 1

    default_name = f"User {next_user_id-1}"

    with gr.Column() as user_block:
        with gr.Row(equal_height=True):
            name = gr.Textbox(value=default_name, label="Name", scale=2, max_lines=1)
            remove_btn = gr.Button("× Remove", variant="stop", scale=0, size="sm")

        prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Type your message here…")
        output = gr.Textbox(label="Output", lines=8, interactive=False)

        # Store reference
        users[user_id] = {
            "name": name,
            "prompt": prompt,
            "output": output,
            "block": user_block,
            "remove_btn": remove_btn
        }

        # Bind remove
        remove_btn.click(
            fn=remove_user,
            inputs=[gr.State(user_id)],
            outputs=[],
            api_name=False
        )

        # Bind generate on prompt submit (Shift+Enter or button)
        prompt.submit(
            fn=generate_for_user,
            inputs=[gr.State(user_id), prompt],
            outputs=output,
            api_name=False
        )

    return user_block


def remove_user(user_id: str):
    if user_id in users:
        users[user_id]["block"].clear()
        del users[user_id]


def generate_for_user(user_id: str, prompt: str):
    if not prompt.strip():
        return "Please enter a prompt."

    if user_id not in users:
        return "User session no longer exists."

    output_comp = users[user_id]["output"]
    yield "▌ Thinking..."

    full_response = ""
    for token in ollama_stream(prompt):
        if token is None:
            break
        full_response += token
        yield full_response + "▌"
        time.sleep(0.01)  # smoother feeling

    yield full_response


# ────────────────────────────────────────────────
#  MAIN INTERFACE
# ────────────────────────────────────────────────

css = """
.user-row { margin-bottom: 24px; border-bottom: 1px solid #444; padding-bottom: 16px; }
.gradio-container { max-width: 980px !important; }
"""

with gr.Blocks(title="Multi-User Ollama Chat", css=css) as demo:

    gr.Markdown("### Multi-user Ollama streamer")

    add_btn = gr.Button("ADD NEW USER", variant="primary", elem_classes="add-new-user")

    user_container = gr.Column()

    # Initial users (optional)
    with user_container:
        add_new_user()      # start with 1 user
        add_new_user()      # or start with 2, 3...

    # When ADD button is clicked → append new user block
    add_btn.click(
        fn=add_new_user,
        inputs=[],
        outputs=user_container,
        queue=False
    )

demo.queue().launch()