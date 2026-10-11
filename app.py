import os
import json
import tempfile
import streamlit as st
import numpy as np
import tensorflow as tf

from generate import generate_sequence
from export import tokens_to_stream, save_midi

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Bach AI Music Generator",
    page_icon="🎵",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_bach_lstm.keras")
VOCAB_PATH = os.path.join(BASE_DIR,  "vocab.json")
EXISTING_MIDI = os.path.join(
    BASE_DIR, "outputs", "generated_bach.mid"
)

# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
.stApp {
    background-color: #10131c;
    color: #ffffff;
}
h1 {
    color: #d9b56d;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #b9c0d0;
    font-size: 18px;
}
.stButton > button {
    background-color: #d9b56d;
    color: #10131c;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab = json.load(file)

    token_to_idx = vocab["token_to_idx"]

    idx_to_token = {
        str(index): token
        for token, index in token_to_idx.items()
    }

    return token_to_idx, idx_to_token

# ---------------- HEADER ----------------

st.title("🎵 Bach AI Music Generator")

st.markdown(
    '<p class="subtitle">'
    'Create original Bach-inspired musical sequences '
    'using Deep Learning and LSTM neural networks.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Generation Settings")

temperature = st.sidebar.slider(
    "Creativity",
    min_value=0.2,
    max_value=1.5,
    value=0.8,
    step=0.1,
    help="Lower values produce more predictable sequences."
)

length = st.sidebar.slider(
    "Number of musical notes/tokens",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

st.sidebar.info(
    "The trained LSTM model predicts musical tokens "
    "one step at a time."
)

# ---------------- MODEL STATUS ----------------

model_exists = os.path.exists(MODEL_PATH)
vocab_exists = os.path.exists(VOCAB_PATH)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Architecture", "LSTM")

with col2:
    st.metric("Music Style", "Bach-inspired")

with col3:
    st.metric(
        "Model Status",
        "Ready" if model_exists and vocab_exists else "Missing files"
    )

st.divider()

# ---------------- MUSIC GENERATION ----------------

st.subheader("🎼 Generate New Music")

st.write(
    "Generate a new musical sequence and download it as a MIDI file."
)

if not model_exists:
    st.error(
        "Trained model not found. Expected location: "
        "models/best_bach_lstm.keras"
    )

elif not vocab_exists:
    st.error(
        "Vocabulary file not found. Expected location: "
        "outputs/vocab.json"
    )

else:
    if st.button("🎹 Generate Music", use_container_width=True):

        try:
            with st.spinner(
                "Loading your trained model and composing music..."
            ):
                model = load_model()
                token_to_idx, idx_to_token = load_vocab()

                # Select a valid seed from the model vocabulary.
                # The model expects a sequence of 100 tokens.
                sorted_tokens = sorted(
                    token_to_idx.items(),
                    key=lambda item: item[1]
                )

                seed_sequence = [
                    token for token, index in sorted_tokens[:100]
                ]

                if not seed_sequence:
                    raise ValueError("The vocabulary is empty.")

                generated_tokens = generate_sequence(
                    model=model,
                    seed_sequence=seed_sequence,
                    length=length,
                    token_to_idx=token_to_idx,
                    idx_to_token=idx_to_token,
                    temperature=temperature
                )

                music_stream = tokens_to_stream(generated_tokens)

                with tempfile.NamedTemporaryFile(
                    suffix=".mid",
                    delete=False
                ) as temp_file:
                    midi_path = temp_file.name

                save_midi(music_stream, midi_path)

                st.session_state["generated_midi"] = midi_path

            st.success("🎉 Your music has been generated!")

            st.download_button(
                label="⬇️ Download Generated MIDI",
                data=open(midi_path, "rb").read(),
                file_name="ai_generated_bach.mid",
                mime="audio/midi",
                use_container_width=True
            )

            st.info(
                "MIDI files contain musical instructions. "
                "Use a MIDI player or music software to listen to them."
            )

        except Exception as error:
            st.error(f"Music generation failed: {error}")

            st.warning(
                "Check the model, vocabulary, dependencies, "
                "and hosting logs for details."
            )

# ---------------- EXISTING MIDI DOWNLOAD ----------------

st.divider()

st.subheader("🎼 Sample Generated Music")

if os.path.exists(EXISTING_MIDI):

    with open(EXISTING_MIDI, "rb") as midi_file:
        sample_data = midi_file.read()

    st.download_button(
        label="⬇️ Download Existing Bach MIDI",
        data=sample_data,
        file_name="generated_bach.mid",
        mime="audio/midi",
        use_container_width=True
    )

else:
    st.caption("No previously generated MIDI file was found.")

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center>Built with Python, TensorFlow, "
    "Streamlit and music21</center>",
    unsafe_allow_html=True
)
