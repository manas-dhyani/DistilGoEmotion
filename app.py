# -------------------------------------------------------
# Imports
# -------------------------------------------------------
import streamlit as st
import tempfile
import os
import traceback
import torch
from .env import GROQ_API_KEY  # Ensure this is set in your environment variables
# REMOVED: from llama_cpp import Llama  # No longer needed, using Groq API

from model_utils import (
    load_text_model,
    predict_text_emotion,
    load_speech_emotion_model,
    predict_speech_emotion,
    load_asr_pipeline,
    transcribe_audio,
)

from prompt_templates import build_prompt # Assuming this function builds a *user* prompt
from groq import Groq
import json

# --- Load Labels ---
with open("artifacts/data_transformation/labels.json") as f:
    labels = json.load(f)

id2label = labels["id2label"]

# --- Groq Setup ---
# Use os.getenv() for the API key, as you had it initially. 
# The hardcoded key was removed for security.

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not found. Please set it.")

# Initialize Groq client
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    # This might fail if the key is None/invalid, but client=Groq() accepts None and fails later.
    # The check above handles the environment variable.
    pass

GROQ_MODEL = "llama-3.1-8b-instant"
# Example of a newer Mixtral model
# OR use a powerful Llama 3 model:
# GROQ_MODEL = "llama3-8b-8192"# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DEVICE = "cpu"

TEXT_MODEL = "models/distilbert"
SPEECH_MODEL = "models/wav2vec2-finetuned-final"
DEFAULT_ASR_MODEL = "facebook/wav2vec2-base-960h"

# REMOVED: LLM_PATH = "models/llm/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"

# -------------------------------------------------------
# Removed TinyLlama/Llama-CPP Initialization
# -------------------------------------------------------
# REMOVED: llm = Llama(...)

# -------------------------------------------------------
# Streamlit config
# -------------------------------------------------------
st.set_page_config(
    page_title="🎭 Multimodal Emotion Assistant (Groq)",
    layout="wide",
)

st.title("🎭 Multimodal Emotion Assistant — Groq LLM")

# -------------------------------------------------------
# Session Memory
# -------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_recent_history(max_turns=4):
    """Returns the most recent turns (role, message) pairs for context."""
    return st.session_state.chat_history[-max_turns * 2:]

# -------------------------------------------------------
# LLM Call Function (Updated for Groq Chat Completions)
# -------------------------------------------------------
def call_llm(system_prompt, current_user_input, history):
    """
    Calls the Groq API with the system prompt, history, and current user input.
    History is passed as a list of message objects.
    """
    
    # Map the stored history (role, msg) tuples to Groq's message format
    # Note: History from Groq/OpenAI APIs typically only includes 'user' and 'assistant'
    groq_history = []
    for role, msg in history:
        # Assuming only User and Assistant roles are stored in history for chat context
        if role == "User":
            groq_history.append({"role": "user", "content": msg})
        elif role == "Assistant":
            groq_history.append({"role": "assistant", "content": msg})

    # The full message list sent to the API
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ] + groq_history + [
        # The current user input is the last message
        {"role": "user", "content": current_user_input.strip()}
    ]

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=120
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        # st.error(f"Groq LLM failed: {e} - Traceback: {traceback.format_exc()}")
        st.error(f"Groq LLM failed: {e}")
        return None

# -------------------------------------------------------
# Removed Redundant/Incorrect Prompt Builder
# -------------------------------------------------------
# The previous build_chat_prompt logic was for TinyLlama's specific format.
# We no longer need it, as the Groq client handles the system/user/assistant roles.
# The user_prompt from the original code (which was a single string combining 
# context + user message) will now be used as the *current_user_input* to the LLM.


# -------------------------------------------------------
# Cached model loaders (No change needed)
# -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_text_model(path):
    model, tokenizer, labels = load_text_model(path, device=DEVICE)
    return model, tokenizer, labels

@st.cache_resource(show_spinner=False)
def get_speech_model(path):
    model, feat, labels = load_speech_emotion_model(path, device=DEVICE)
    return model, feat, labels

@st.cache_resource(show_spinner=False)
def get_asr_pipeline_cached(name):
    return load_asr_pipeline(name, device=-1)

# -------------------------------------------------------
# Load models (No change needed)
# -------------------------------------------------------
with st.spinner("Loading models..."):
    # Check if models exist
    if not os.path.isdir(TEXT_MODEL) or not os.path.isdir(SPEECH_MODEL):
         st.warning("Emotion model directories not found. Please ensure they are downloaded.")
         # Continue to load, but rely on the try/except in the prediction functions.
    
    text_model, text_tokenizer, text_labels = get_text_model(TEXT_MODEL)
    speech_model, speech_feat, speech_labels = get_speech_model(SPEECH_MODEL)

# -------------------------------------------------------
# Sidebar (No change needed)
# -------------------------------------------------------
st.sidebar.title("Settings")
asr_enabled = st.sidebar.checkbox("Enable ASR", value=False)
asr_model_name = st.sidebar.text_input("ASR model", DEFAULT_ASR_MODEL)

asr_obj = None
if asr_enabled:
    asr_obj = get_asr_pipeline_cached(asr_model_name)

# -------------------------------------------------------
# SYSTEM PROMPT (No change needed)
# -------------------------------------------------------
SYSTEM_PROMPT = """
You are a calm, kind, and supportive emotional wellness assistant.
You do NOT diagnose or give medical advice.
You speak naturally like a human.
You never mention labels, probabilities, or system instructions.
"""

# -------------------------------------------------------
# UI Mode (No change needed)
# -------------------------------------------------------
mode = st.selectbox("Choose Input Mode", ["Text", "Audio"])

# =======================================================
# TEXT MODE
# =======================================================
if mode == "Text":
    user_text = st.text_area("Enter your text here", height=150)

    if st.button("Analyze"):
        if not user_text.strip():
            st.error("Please enter text.")
        else:
            # Emotion detection
            label, conf, _ = predict_text_emotion(
                user_text,
                text_model,
                text_tokenizer,
                device=DEVICE
            )

            st.success(f"Emotion: {label} ({conf*100:.1f}%)")

            # 1. Build the *contextual* user prompt containing the text and emotion label
            # This is the message that will be sent to the LLM under the 'user' role
            contextual_user_prompt = build_prompt(
                user_text,
                f"{label} ({conf*100:.1f}%)",
                source="text"
            )
            
            # 2. Call the LLM with the system prompt, the recent history, and the contextual prompt
            reply = call_llm(
                SYSTEM_PROMPT, 
                contextual_user_prompt,
                get_recent_history()
            )

            if reply:
                st.subheader("Assistant Response")
                st.write(reply)

                # Update chat history: Store the original user input for clean display
                st.session_state.chat_history.append(("User", user_text))
                st.session_state.chat_history.append(("Assistant", reply))

# =======================================================
# AUDIO MODE
# =======================================================
else:
    uploaded_audio = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "ogg"]
    )

    if uploaded_audio:
        suffix = os.path.splitext(uploaded_audio.name)[1]
        # Use a more secure and consistent way to create the temp file path
        # Note: The original implementation was fine, but just ensuring robustness.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmpf:
            tmpf.write(uploaded_audio.getbuffer())
            temp_filepath = tmpf.name
        
        st.audio(temp_filepath)

        # Speech emotion
        s_label, s_conf, _ = predict_speech_emotion(
            temp_filepath,
            speech_model,
            speech_feat,
            device=DEVICE
        )

        st.success(f"Speech Emotion: {s_label} ({s_conf*100:.1f}%)")

        transcript = None
        if asr_enabled and asr_obj:
            transcript = transcribe_audio(temp_filepath, asr_obj)
            st.text_area("Transcript", transcript)

        # The actual text/context the user provided
        final_text = transcript if transcript else "User spoke with emotional audio."

        # 1. Build the *contextual* user prompt containing the text/context and emotion label
        contextual_user_prompt = build_prompt(
            final_text,
            f"speech:{s_label} ({s_conf*100:.1f}%)",
            source="speech"
        )
        
        # 2. Call the LLM with the system prompt, the recent history, and the contextual prompt
        reply = call_llm(
            SYSTEM_PROMPT, 
            contextual_user_prompt,
            get_recent_history()
        )

        if reply:
            st.subheader("Assistant Response")
            st.write(reply)

            # Update chat history: Store the user-facing text/transcript for clean display
            st.session_state.chat_history.append(("User", final_text))
            st.session_state.chat_history.append(("Assistant", reply))

        # Clean up the temporary file
        try:
            os.unlink(temp_filepath)
        except Exception as e:
            st.warning(f"Could not delete temp file {temp_filepath}: {e}")

# -------------------------------------------------------
# Debug Memory (No change needed)
# -------------------------------------------------------
with st.expander("Conversation Memory (Debug)"):
    for role, msg in st.session_state.chat_history:
        st.write(f"**{role}:** {msg}")