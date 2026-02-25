"""
Multi-user Ollama streaming chat with Streamlit
- Dynamic add/remove users
- Per-user model & temperature selection
- Streaming responses
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path
import yaml
from typing import Generator

# ─── Configuration ──────────────────────────────────────────────────────────────

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_FILE = CONFIG_DIR / "configs.yml"

DEFAULT_CONFIG = {
    "ollama": {"api_base": "http://localhost:11434"},
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

def load_config():
    if not CONFIG_FILE.is_file():
        return DEFAULT_CONFIG
    try:
        with CONFIG_FILE.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    except Exception as e:
        st.warning(f"Config load error: {e}")
        return DEFAULT_CONFIG

CONFIG = load_config()

OLLAMA_API_BASE     = CONFIG["ollama"]["api_base"]
AVAILABLE_MODELS    = CONFIG["models"]["available"]
DEFAULT_MODEL       = CONFIG["models"]["default"]
MODEL_DISPLAY_NAMES = CONFIG["models"].get("display_names", {})
DEFAULT_TEMP        = CONFIG["generation"]["default_temperature"]
TEMP_MIN            = CONFIG["generation"]["temperature"]["min"]
TEMP_MAX            = CONFIG["generation"]["temperature"]["max"]
TEMP_STEP           = CONFIG["generation"]["temperature"]["step"]
USER_PREFIX         = CONFIG["ui"]["default_user_name_prefix"]
STREAM_DELAY        = CONFIG["ui"]["streaming_animation_delay"]

def model_label(model_id: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)

# ─── Ollama Streaming ───────────────────────────────────────────────────────────

def stream_ollama(prompt: str, model: str, temperature: float) -> Generator[str, None, None]:
    url = f"{OLLAMA_API_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "stream": True,
        "options": {"temperature": float(temperature)},
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
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
        yield f"\n\n**Error**\n{str(e)}"

# ─── Streamlit UI ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multi-User Ollama Chat", layout="wide")

st.title("Multi-User Ollama Streaming Chat")
st.caption(f"Available models: {', '.join(model_label(m) for m in AVAILABLE_MODELS)}")

# Initialize session state
if "users" not in st.session_state:
    st.session_state.users = [
        {
            "name": f"{USER_PREFIX} {i+1}",
            "model": DEFAULT_MODEL,
            "temp": DEFAULT_TEMP,
            "response": ""
        }
        for i in range(CONFIG["ui"]["initial_users_count"])
    ]

if "next_user_id" not in st.session_state:
    st.session_state.next_user_id = len(st.session_state.users) + 1

# ── Add new user button ─────────────────────────────────────────────────────────
if st.button("➕ Add New User", type="primary", use_container_width=True):
    st.session_state.users.append({
        "name": f"{USER_PREFIX} {st.session_state.next_user_id}",
        "model": DEFAULT_MODEL,
        "temp": DEFAULT_TEMP,
        "response": ""
    })
    st.session_state.next_user_id += 1
    st.rerun()

# ── Render all users ────────────────────────────────────────────────────────────
for idx, user in enumerate(st.session_state.users):
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([3, 4, 3, 1])

        with col1:
            user["name"] = st.text_input(
                "Name",
                value=user["name"],
                key=f"name_{idx}",
                label_visibility="collapsed"
            )

        with col2:
            user["model"] = st.selectbox(
                "Model",
                options=AVAILABLE_MODELS,
                format_func=model_label,
                index=AVAILABLE_MODELS.index(user["model"]),
                key=f"model_{idx}",
                label_visibility="collapsed"
            )

        with col3:
            user["temp"] = st.slider(
                "Temperature",
                min_value=TEMP_MIN,
                max_value=TEMP_MAX,
                value=user["temp"],
                step=TEMP_STEP,
                format="%.2f",
                key=f"temp_{idx}",
                label_visibility="collapsed"
            )

        with col4:
            if st.button("×", key=f"remove_{idx}", type="tertiary", help="Remove user"):
                st.session_state.users.pop(idx)
                st.rerun()

        # Prompt
        prompt = st.text_area(
            "Prompt",
            placeholder="Type your message… (Shift+Enter or click Send)",
            height=110,
            key=f"prompt_{idx}",
            label_visibility="visible"
        )

        # Response area
        response_container = st.empty()

        # Send button + logic
        if st.button("Send", key=f"send_{idx}", use_container_width=True, type="primary"):
            if not prompt.strip():
                response_container.markdown("**Please enter a message.**")
            else:
                with response_container.container():
                    placeholder = st.empty()
                    placeholder.markdown("**Thinking...**")

                    full_response = ""
                    try:
                        for chunk in stream_ollama(prompt, user["model"], user["temp"]):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                            time.sleep(STREAM_DELAY)
                        placeholder.markdown(full_response)
                        user["response"] = full_response  # optional: keep last response
                    except Exception as e:
                        placeholder.error(f"Error: {str(e)}")

        # Show previous response if exists (nice to have)
        if user.get("response") and not prompt:
            st.markdown("**Last response:**")
            st.markdown(user["response"])

st.markdown("---")
st.caption("Streamlit + Ollama multi-user chat • refresh removes unsent prompts")