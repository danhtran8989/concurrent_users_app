import streamlit as st
import requests
import json
import time
from pathlib import Path
import yaml
from typing import Iterator

# ─── Configuration (same as yours) ──────────────────────────────────────────────
# ... keep your CONFIG loading, constants, model_label, etc. unchanged ...

# ─── Ollama Streaming (small change: return Iterator[str]) ──────────────────────
def stream_ollama(prompt: str, model: str, temperature: float) -> Iterator[str]:
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


# ─── Fragment ────────────────────────────────────────────────────────────────────
@st.fragment
def user_chat_ui(idx: int, user: dict):
    """Everything that should re-run independently for this user"""

    col1, col2, col3, col4 = st.columns([3, 4, 3, 1])

    with col1:
        user["name"] = st.text_input(
            "Name", value=user["name"], key=f"name_{idx}", label_visibility="collapsed"
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

    prompt = st.text_area(
        "Prompt",
        placeholder="Type your message… (Shift+Enter or click Send)",
        height=110,
        key=f"prompt_{idx}",
        label_visibility="visible"
    )

    if st.button("Send", key=f"send_{idx}", use_container_width=True, type="primary"):
        if not prompt.strip():
            st.error("Please enter a message.")
        else:
            # The magic line ── streams without blocking whole page
            full_response = st.write_stream(
                stream_ollama(prompt, user["model"], user["temp"])
            )
            # Optional: save last answer
            user["response"] = full_response

    # Optional: show last response when prompt is empty
    if user.get("response") and not prompt.strip():
        st.markdown("**Last response:**")
        st.markdown(user["response"])


# ─── Main page ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-User Ollama Chat", layout="wide")
st.title("Multi-User Ollama Streaming Chat")
st.caption(f"Available models: {', '.join(model_label(m) for m in AVAILABLE_MODELS)}")

# Init session state (same as yours)
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

if st.button("➕ Add New User", type="primary", use_container_width=True):
    st.session_state.users.append({
        "name": f"{USER_PREFIX} {st.session_state.next_user_id}",
        "model": DEFAULT_MODEL,
        "temp": DEFAULT_TEMP,
        "response": ""
    })
    st.session_state.next_user_id += 1
    st.rerun()

# ── Render users with fragment ──────────────────────────────────────────────────
for idx, user in enumerate(st.session_state.users):
    with st.container(border=True):
        user_chat_ui(idx, user)

st.markdown("---")
st.caption("Streamlit + Ollama multi-user chat • each user streams independently")