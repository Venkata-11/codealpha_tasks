import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
import html
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Handwriting Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent

MNIST_PATH = BASE_DIR / "models" / "mnist_cnn.keras"
EMNIST_PATH = BASE_DIR / "models" / "emnist_cnn.keras"
CRNN_PATH = BASE_DIR / "models" / "iam_crnn_improved.keras"
VOCAB_PATH = BASE_DIR / "data" / "iam" / "character_vocabulary.json"


# =========================================================
# CUSTOM DESIGN  (Paper & Ink theme)
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Caveat:wght@500;600&family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --paper: #F3EDE2;
    --card: #FBF8F1;
    --line: #E0D5C2;
    --ink: #1B2A4A;
    --muted: #4A5266;
    --vermilion: #B8321B;
    --blush: #E9B7AB;
    --blush-soft: #F6E3DD;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: var(--paper);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

/* HEADER */

.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 15px 20px;
    margin-bottom: 34px;
    background: var(--card);
    border: 1.5px solid var(--line);
    border-radius: 999px;
    box-shadow: 0 10px 30px -18px rgba(27,42,74,.35);
}

.app-logo {
    width: 46px;
    height: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: var(--ink);
    color: var(--paper);
    font-size: 22px;
}

.app-name {
    font-family: 'Instrument Serif', serif;
    font-size: 26px;
    font-weight: 400;
    color: var(--ink);
    line-height: 1;
}

.app-description {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 11px;
    letter-spacing: .08em;
    margin-top: 4px;
}

/* MAIN TITLE */

.page-title {
    font-family: 'Instrument Serif', serif;
    font-size: clamp(44px, 6vw, 72px);
    font-weight: 400;
    line-height: .98;
    letter-spacing: -1px;
    color: var(--ink);
    margin-bottom: 10px;
}

.page-description {
    font-size: 16px;
    line-height: 1.6;
    color: var(--muted);
    margin-bottom: 25px;
}

/* MODE SELECTOR */

.mode-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: .14em;
    color: var(--vermilion);
    margin-bottom: 8px;
}

/* SECTION HEADINGS */

.section-heading {
    font-family: 'Instrument Serif', serif;
    font-size: 34px;
    font-weight: 400;
    color: var(--ink);
    margin-bottom: 5px;
}

.section-description {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 18px;
}

/* INPUT AND OUTPUT PANELS */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 28px !important;
    overflow: hidden;
    box-shadow: 0 25px 50px -25px rgba(27,42,74,.35);
}

.input-panel [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card) !important;
    border: 1.5px solid var(--line) !important;
}

.output-panel [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--blush-soft) !important;
    border: 1.5px solid var(--blush) !important;
}

/* PANEL CONTENT */

.panel-heading {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--vermilion);
    margin-bottom: 12px;
}

.panel-caption {
    color: var(--muted);
    font-size: 12px;
}

/* UPLOADER */

[data-testid="stFileUploader"] {
    background: var(--paper);
    border: 1.5px dashed var(--ink);
    border-radius: 18px;
    padding: 12px;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--vermilion);
}

/* RESULT */

.result-box {
    background-color: var(--ink);
    color: var(--paper);
    border-radius: 22px;
    padding: 27px;
    min-height: 190px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 20px 40px -16px rgba(27,42,74,.6);
    animation: appear .35s ease-out;
}

.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .14em;
    color: var(--blush);
    text-transform: uppercase;
    margin-bottom: 14px;
}

.result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(22px, 2.6vw, 32px);
    font-weight: 500;
    color: var(--paper);
    overflow-wrap: anywhere;
    line-height: 1.5;
}

.character-value {
    font-family: 'Caveat', cursive;
    font-size: 110px;
    font-weight: 600;
    color: var(--paper);
    text-align: center;
    line-height: 1.1;
}

/* EMPTY RESULT */

.empty-result {
    min-height: 190px;
    background: var(--card);
    border: 1.5px dashed var(--blush);
    border-radius: 22px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 20px;
}

.empty-icon {
    font-family: 'Instrument Serif', serif;
    font-style: italic;
    font-size: 40px;
    color: var(--vermilion);
    margin-bottom: 10px;
}

.empty-title {
    font-family: 'Instrument Serif', serif;
    font-size: 24px;
    color: var(--ink);
}

.empty-description {
    font-size: 13px;
    color: var(--muted);
    margin-top: 6px;
}

/* BUTTONS */

.stButton > button {
    background: var(--ink);
    color: var(--paper);
    border: none;
    border-radius: 999px;
    min-height: 48px;
    font-weight: 600;
    transition: all .2s ease;
}

.stButton > button:hover {
    background: var(--vermilion);
    color: var(--card);
    border: none;
    transform: translateY(-2px);
}

/* METRICS */

[data-testid="stMetric"] {
    background: var(--card);
    border: 1.5px solid var(--line);
    border-radius: 18px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    font-family: 'Instrument Serif', serif;
    color: var(--vermilion);
    font-weight: 400;
}

/* PROGRESS BARS */

[data-testid="stProgress"] [role="progressbar"] > div {
    background-color: var(--vermilion);
}

/* FOOTER */

.app-footer {
    border-top: 1.5px solid var(--line);
    margin-top: 38px;
    padding-top: 18px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    text-align: center;
}

@keyframes appear {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return tf.keras.models.load_model(
        str(model_path),
        compile=False
    )


@st.cache_resource
def load_vocabulary():

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulary not found: {VOCAB_PATH}"
        )

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# CHARACTER PREPROCESSING
# =========================================================

def preprocess_character(image):

    image = image.convert("L")
    img = np.array(image)

    if np.mean(img) > 127:
        img = 255 - img

    _, img = cv2.threshold(
        img,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(img)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)

    img = img[y:y+h, x:x+w]

    scale = 20.0 / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = img

    canvas = canvas.astype("float32") / 255.0

    return canvas.reshape(1, 28, 28, 1)


# =========================================================
# LINE PREPROCESSING
# =========================================================

def preprocess_line(image):

    image = image.convert("L")
    img = np.array(image)

    if np.mean(img) < 127:
        img = 255 - img

    height, width = img.shape

    target_height = 64
    scale = target_height / height
    new_width = max(1, int(width * scale))

    if new_width > 2560:
        new_width = 2560
        scale = new_width / width
        target_height = max(1, int(height * scale))

    img = cv2.resize(
        img,
        (new_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    padded_width = ((new_width + 3) // 4) * 4

    padded = np.ones(
        (64, padded_width),
        dtype=np.float32
    )

    padded[:target_height, :new_width] = (
        img.astype("float32") / 255.0
    )

    input_length = padded_width // 4

    return padded.reshape(1, 64, padded_width, 1), input_length


# =========================================================
# CHARACTER PREDICTION
# =========================================================

def predict_character(image, mode):

    if mode == "Digit Recognition":

        model = load_model(MNIST_PATH)
        class_names = [str(i) for i in range(10)]

    else:

        model = load_model(EMNIST_PATH)
        class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    processed = preprocess_character(image)

    if processed is None:
        raise ValueError("No handwriting detected.")

    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    prediction = class_names[predicted_index]

    model_score = float(probabilities[predicted_index])

    return prediction, model_score, probabilities


# =========================================================
# LINE PREDICTION
# =========================================================

def predict_line(image):

    model = load_model(CRNN_PATH)
    vocabulary = load_vocabulary()

    index_to_char = vocabulary["index_to_char"]

    processed, input_length = preprocess_line(image)

    logits = model.predict(
        processed,
        verbose=0
    )

    decoded, _ = tf.nn.ctc_greedy_decoder(
        tf.transpose(logits, [1, 0, 2]),
        sequence_length=tf.constant([input_length]),
        merge_repeated=True,
        blank_index=0
    )

    decoded_indices = tf.sparse.to_dense(
        decoded[0],
        default_value=-1
    ).numpy()[0]

    result = "".join(
        index_to_char.get(str(int(index)), "")
        for index in decoded_indices
        if int(index) > 0
    )

    return result.strip()


# =========================================================
# RESULT DISPLAY HELPERS
# =========================================================

def show_empty_result(message):

    st.markdown(
        f"""
        <div class="empty-result">
            <div class="empty-icon">✧</div>
            <div class="empty-title">Prediction output</div>
            <div class="empty-description">{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_line_result(prediction):

    displayed_text = html.escape(
        prediction if prediction else "No text detected"
    )

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Recognized text</div>
            <div class="result-value">{displayed_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_character_result(prediction):

    displayed_character = html.escape(prediction)

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">Predicted character</div>
            <div class="character-value">{displayed_character}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="app-header">
    <div class="app-logo">✍</div>
    <div>
        <div class="app-name">Handwriting Recognition</div>
    </div>
</div>

<div class="page-title">Recognition Workspace</div>

<div class="page-description">
    Select a recognition task, provide handwriting, and view the model prediction.
</div>
""", unsafe_allow_html=True)


# =========================================================
# RECOGNITION NAVIGATION
# =========================================================

workspace = st.radio(
    "Recognition task",
    [
        "Digit Recognition",
        "Character Recognition",
        "Line Recognition"
    ],
    horizontal=True,
    key="recognition_workspace"
)

st.write("")


# =========================================================
# LINE RECOGNITION
# =========================================================

if workspace == "Line Recognition":

    st.markdown(
        '<div class="section-heading">Handwritten line recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Upload an image containing a handwritten word or line of text.'
        '</div>',
        unsafe_allow_html=True
    )

    input_col, output_col = st.columns(
        [1, 1],
        gap="large"
    )

    with input_col:

        st.markdown('<div class="input-panel">', unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(
                '<div class="panel-heading">01 &nbsp; Input image</div>',
                unsafe_allow_html=True
            )

            uploaded_line = st.file_uploader(
                "Choose a handwriting image",
                type=["png", "jpg", "jpeg", "bmp"],
                key="line_upload"
            )

            if uploaded_line:

                line_image = Image.open(
                    uploaded_line
                ).convert("RGB")

                st.image(
                    line_image,
                    caption="Input handwriting",
                    use_container_width=True
                )

                if st.button(
                    "Recognize text",
                    use_container_width=True,
                    key="line_predict"
                ):

                    with st.spinner("Recognizing handwriting..."):

                        try:

                            prediction = predict_line(line_image)

                            st.session_state["line_prediction"] = prediction
                            st.session_state["line_source"] = uploaded_line.name

                        except Exception as error:

                            st.error(f"Recognition error: {error}")

            else:

                st.info("Upload a handwritten image to begin.")

        st.markdown('</div>', unsafe_allow_html=True)

    with output_col:

        st.markdown('<div class="output-panel">', unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(
                '<div class="panel-heading">02 &nbsp; Recognition result</div>',
                unsafe_allow_html=True
            )

            if "line_prediction" in st.session_state:

                show_line_result(
                    st.session_state["line_prediction"]
                )

                st.caption(
                    "Input file: "
                    + st.session_state.get("line_source", "")
                )

                if st.button(
                    "Clear result",
                    key="clear_line"
                ):

                    st.session_state.pop("line_prediction", None)
                    st.session_state.pop("line_source", None)
                    st.rerun()

            else:

                show_empty_result(
                    "Your recognized text will appear here."
                )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# DIGIT AND CHARACTER RECOGNITION
# =========================================================

else:

    is_digit = workspace == "Digit Recognition"

    if is_digit:

        st.markdown(
            '<div class="section-heading">Handwritten digit recognition</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Draw or upload a single handwritten digit from 0 to 9.'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="section-heading">Handwritten character recognition</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'Draw or upload a single uppercase English letter.'
            '</div>',
            unsafe_allow_html=True
        )

    input_col, output_col = st.columns(
        [1, 1],
        gap="large"
    )

    character_image = None

    with input_col:

        st.markdown('<div class="input-panel">', unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(
                '<div class="panel-heading">01 &nbsp; Input handwriting</div>',
                unsafe_allow_html=True
            )

            input_method = st.radio(
                "Input method",
                ["Draw", "Upload"],
                horizontal=True,
                key=f"input_method_{workspace}"
            )

            if input_method == "Draw":

                canvas_result = st_canvas(
                    fill_color="rgba(255,255,255,0)",
                    stroke_width=15,
                    stroke_color="#1B2A4A",
                    background_color="#FFFFFF",
                    height=250,
                    width=450,
                    drawing_mode="freedraw",
                    key=f"canvas_{workspace}",
                    display_toolbar=True
                )

                if canvas_result.image_data is not None:

                    canvas_array = canvas_result.image_data.astype("uint8")
                    rgb = canvas_array[:, :, :3]

                    if np.any(np.min(rgb, axis=2) < 245):

                        character_image = Image.fromarray(
                            rgb
                        ).convert("RGB")

            else:

                uploaded_character = st.file_uploader(
                    "Choose an image",
                    type=["png", "jpg", "jpeg", "bmp"],
                    key=f"character_upload_{workspace}"
                )

                if uploaded_character:

                    character_image = Image.open(
                        uploaded_character
                    ).convert("RGB")

                    st.image(
                        character_image,
                        caption="Input handwriting",
                        use_container_width=True
                    )

            predict_button = st.button(
                "Recognize character",
                use_container_width=True,
                key=f"character_predict_{workspace}"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with output_col:

        st.markdown('<div class="output-panel">', unsafe_allow_html=True)

        with st.container(border=True):

            st.markdown(
                '<div class="panel-heading">02 &nbsp; Recognition result</div>',
                unsafe_allow_html=True
            )

            if predict_button:

                if character_image is None:

                    st.warning(
                        "Please draw or upload a character first."
                    )

                else:

                    with st.spinner("Analyzing handwriting..."):

                        try:

                            prediction, model_score, probabilities = (
                                predict_character(
                                    character_image,
                                    workspace
                                )
                            )

                            st.session_state["character_prediction"] = prediction
                            st.session_state["character_score"] = model_score
                            st.session_state["character_probabilities"] = probabilities
                            st.session_state["character_mode"] = workspace

                        except Exception as error:

                            st.error(f"Prediction error: {error}")

            if (
                "character_prediction" in st.session_state
                and st.session_state.get("character_mode") == workspace
            ):

                prediction = st.session_state["character_prediction"]
                model_score = st.session_state["character_score"]
                probabilities = st.session_state["character_probabilities"]

                show_character_result(prediction)

                st.write("")

                st.metric(
                    "Model score",
                    f"{model_score * 100:.2f}%"
                )

                st.caption(
                    "Model output score; not a calibrated measure of reliability."
                )

                st.markdown("**Other top predictions**")

                top_indices = np.argsort(probabilities)[-3:][::-1]

                for index in top_indices:

                    if workspace == "Digit Recognition":
                        class_name = str(index)
                    else:
                        class_name = chr(ord("A") + int(index))

                    score = float(probabilities[index])

                    st.progress(
                        score,
                        text=f"{class_name} — {score * 100:.1f}%"
                    )

                if st.button(
                    "Clear result",
                    key=f"clear_character_{workspace}"
                ):

                    st.session_state.pop("character_prediction", None)
                    st.session_state.pop("character_score", None)
                    st.session_state.pop("character_probabilities", None)
                    st.session_state.pop("character_mode", None)

                    st.rerun()

            else:

                show_empty_result(
                    "Your predicted character will appear here."
                )

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="app-footer">
    Handwriting Recognition · CNN · CRNN · TensorFlow
</div>
""", unsafe_allow_html=True)