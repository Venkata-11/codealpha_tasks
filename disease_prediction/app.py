import random
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HealthSense | Health Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# 2. GLOBAL STYLING AND ANIMATIONS
# ============================================================

GLOBAL_CSS = """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,800;1,500&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #F7F5F0;
    --sage: #7A9E8E;
    --sage-light: #A9C8B8;
    --terracotta: #C77D6B;
    --rose: #E9B7B0;
    --slate: #2F3E46;
    --muted: #718078;
}

html, body {
    background: var(--bg);
}

.stApp {
    background: var(--bg);
    color: var(--slate);
    font-family: 'Outfit', sans-serif;
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4, p, label {
    font-family: 'Outfit', sans-serif;
}

h1, h2, h3, h4 {
    color: var(--slate);
}


/* ------------------------------------------------------------
   ANIMATIONS
------------------------------------------------------------ */

@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-13px);
    }
}

@keyframes floatSlow {
    0%, 100% {
        transform: translateY(0px) rotate(-2deg);
    }
    50% {
        transform: translateY(-18px) rotate(2deg);
    }
}

@keyframes heartbeat {
    0%, 100% {
        transform: scale(1);
    }
    15% {
        transform: scale(1.13);
    }
    30% {
        transform: scale(1);
    }
    45% {
        transform: scale(1.08);
    }
}

@keyframes moveECG {
    from {
        transform: translateX(0);
    }
    to {
        transform: translateX(-50%);
    }
}

@keyframes rotateSlow {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: .45;
        transform: scale(.9);
    }
    50% {
        opacity: 1;
        transform: scale(1.15);
    }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes shine {
    from {
        background-position: -200% center;
    }
    to {
        background-position: 200% center;
    }
}

@keyframes dropMove {
    0%, 100% {
        transform: translateY(0) rotate(-5deg);
    }
    50% {
        transform: translateY(-12px) rotate(5deg);
    }
}

@keyframes ribbonMove {
    0%, 100% {
        transform: rotate(-5deg) translateY(0);
    }
    50% {
        transform: rotate(5deg) translateY(-12px);
    }
}

@keyframes orbit {
    from {
        transform: rotate(0deg) translateX(12px) rotate(0deg);
    }
    to {
        transform: rotate(360deg) translateX(12px) rotate(-360deg);
    }
}

@keyframes reveal {
    from {
        opacity: 0;
        transform: translateY(35px) scale(.97);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}


/* ------------------------------------------------------------
   BACKGROUND
------------------------------------------------------------ */

.background-glow {
    position: fixed;
    width: 380px;
    height: 380px;
    border-radius: 50%;
    filter: blur(90px);
    opacity: .35;
    pointer-events: none;
    z-index: 0;
    animation: floatSlow 15s ease-in-out infinite;
}

.glow-one {
    background: #A9C8B8;
    top: -150px;
    left: -130px;
}

.glow-two {
    background: #E9B7B0;
    right: -150px;
    top: 35%;
    animation-delay: -5s;
}

.glow-three {
    background: #EADFC9;
    bottom: -180px;
    left: 35%;
    animation-delay: -8s;
}


/* ------------------------------------------------------------
   HEADER
------------------------------------------------------------ */

.brand-name {
    font-family: 'Fraunces', serif;
    font-size: 29px;
    font-weight: 800;
    color: var(--slate);
    letter-spacing: -1px;
    display: flex;
    align-items: center;
    gap: 9px;
}

.brand-heart {
    display: inline-block;
    color: var(--terracotta);
    font-size: 27px;
    animation: heartbeat 1.2s infinite;
}

.brand-name span {
    color: var(--sage);
}

.nav-label {
    display: inline-block;
    padding: 9px 16px;
    border-radius: 30px;
    background: rgba(255,255,255,.75);
    border: 1px solid #E6E1D7;
    color: #64796C;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
}

.status-dot {
    display: inline-block;
    height: 8px;
    width: 8px;
    border-radius: 50%;
    background: var(--sage);
    margin-right: 7px;
    animation: pulse 1.5s infinite;
}


/* ------------------------------------------------------------
   HERO SECTION
------------------------------------------------------------ */

.hero {
    position: relative;
    overflow: hidden;
    border-radius: 30px;
    padding: 48px 45px;
    margin-top: 22px;
    margin-bottom: 30px;
    background: linear-gradient(
        120deg,
        #E8EEE7 0%,
        #F3EFE6 55%,
        #F0DDD6 100%
    );
    border: 1px solid #E6E0D6;
    animation: fadeUp .9s ease both;
}

.hero::before {
    content: "";
    position: absolute;
    width: 350px;
    height: 350px;
    right: -130px;
    top: -160px;
    border-radius: 50%;
    background: rgba(122,158,142,.13);
}

.hero-kicker {
    color: var(--sage);
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 17px;
}

.hero-heading {
    font-family: 'Fraunces', serif;
    color: var(--slate);
    font-size: clamp(38px, 5vw, 62px);
    line-height: 1.1;
    letter-spacing: -1.5px;
    font-weight: 800;
    margin-bottom: 20px;
}

.hero-heading .highlight {
    background: linear-gradient(
        90deg,
        #6F9A86,
        #C77D6B,
        #6F9A86
    );
    background-size: 200%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    animation: shine 6s linear infinite;
}

.hero-copy {
    max-width: 570px;
    color: #65746C;
    font-size: 16px;
    line-height: 1.8;
    font-weight: 400;
}

.hero-pill {
    display: inline-block;
    margin-top: 24px;
    padding: 11px 18px;
    border-radius: 30px;
    background: rgba(255,255,255,.75);
    border: 1px solid rgba(122,158,142,.25);
    color: #64796C;
    font-size: 12px;
    font-weight: 700;
}


/* ------------------------------------------------------------
   HERO MEDICAL VISUALS
------------------------------------------------------------ */

.hero-visual {
    position: relative;
    height: 390px;
    display: flex;
    justify-content: center;
    align-items: center;
}

.visual-orbit {
    position: absolute;
    width: 310px;
    height: 310px;
    border: 1.5px dashed rgba(111,154,134,.55);
    border-radius: 50%;
    animation: rotateSlow 35s linear infinite;
}

.visual-orbit-two {
    position: absolute;
    width: 365px;
    height: 365px;
    border: 1px solid rgba(199,125,107,.25);
    border-radius: 50%;
    animation: rotateSlow 48s linear infinite reverse;
}

.visual-center {
    width: 175px;
    height: 175px;
    border-radius: 50%;
    background: radial-gradient(
        circle at 35% 30%,
        #FFFFFF,
        #B8D0C3 55%,
        #7A9E8E
    );
    display: flex;
    justify-content: center;
    align-items: center;
    box-shadow: 0 25px 65px rgba(111,154,134,.35);
    animation: floatSlow 5s ease-in-out infinite;
    z-index: 2;
}

.visual-heart {
    font-size: 90px;
    color: var(--terracotta);
    filter: drop-shadow(0 8px 12px rgba(199,125,107,.3));
    animation: heartbeat 1.2s infinite;
}

.floating-medical {
    position: absolute;
    z-index: 3;
    display: flex;
    justify-content: center;
    align-items: center;
    background: rgba(255,255,255,.88);
    border: 1px solid rgba(255,255,255,.9);
    box-shadow: 0 12px 30px rgba(47,62,70,.12);
    backdrop-filter: blur(10px);
}

.medical-heart {
    width: 66px;
    height: 66px;
    border-radius: 20px;
    top: 20px;
    left: 5%;
    animation: float 4.5s ease-in-out infinite;
    font-size: 34px;
    color: var(--terracotta);
}

.medical-glucose {
    width: 75px;
    height: 75px;
    border-radius: 22px;
    top: 22px;
    right: 2%;
    animation: dropMove 5s ease-in-out infinite;
    font-size: 38px;
    color: #6F9A86;
}

.medical-ribbon {
    width: 70px;
    height: 70px;
    border-radius: 22px;
    bottom: 35px;
    left: 4%;
    animation: ribbonMove 5s ease-in-out infinite;
    font-size: 39px;
    color: #D98EAA;
}

.medical-cell {
    width: 65px;
    height: 65px;
    border-radius: 20px;
    bottom: 25px;
    right: 5%;
    animation: float 5.5s ease-in-out infinite;
    font-size: 32px;
    color: #C77D6B;
}

.visual-caption {
    position: absolute;
    bottom: -4px;
    font-family: 'Fraunces', serif;
    color: var(--slate);
    font-size: 19px;
    font-weight: 700;
    text-align: center;
}


/* ------------------------------------------------------------
   MOVING ECG LINE
------------------------------------------------------------ */

.ecg-container {
    overflow: hidden;
    height: 75px;
    border-radius: 18px;
    background: rgba(255,255,255,.55);
    border: 1px solid rgba(255,255,255,.85);
    margin: 0 0 35px 0;
}

.ecg-track {
    width: 200%;
    height: 100%;
    animation: moveECG 8s linear infinite;
}

.ecg-track svg {
    width: 100%;
    height: 100%;
}

.ecg-track path {
    fill: none;
    stroke: #7A9E8E;
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-linejoin: round;
}


/* ------------------------------------------------------------
   SECTION HEADINGS
------------------------------------------------------------ */

.section-heading {
    font-family: 'Fraunces', serif;
    color: var(--slate);
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -.5px;
    margin-top: 28px;
    margin-bottom: 5px;
}

.section-copy {
    color: #7C8580;
    font-size: 14px;
    line-height: 1.7;
    margin-bottom: 20px;
}


/* ------------------------------------------------------------
   PROJECT METRICS
------------------------------------------------------------ */

.metric-card {
    background: rgba(255,255,255,.8);
    border: 1px solid #E7E3D9;
    border-radius: 20px;
    padding: 23px;
    min-height: 120px;
    backdrop-filter: blur(10px);
    transition: transform .3s ease, box-shadow .3s ease;
    animation: float 7s ease-in-out infinite;
}

.metric-card:hover {
    transform: translateY(-7px);
    box-shadow: 0 18px 35px rgba(47,62,70,.1);
}

.metric-card.m2 {
    animation-delay: -2s;
}

.metric-card.m3 {
    animation-delay: -4s;
}

.metric-number {
    font-family: 'Fraunces', serif;
    color: var(--sage);
    font-size: 40px;
    font-weight: 800;
}

.metric-label {
    color: #7D8780;
    font-size: 13px;
    margin-top: 3px;
}


/* ------------------------------------------------------------
   DISEASE CARDS
------------------------------------------------------------ */

.condition-card {
    position: relative;
    overflow: hidden;
    border-radius: 23px;
    padding: 25px 22px;
    min-height: 210px;
    border: 1.5px solid #E7E3D9;
    margin-bottom: 10px;
    transition: transform .35s ease, box-shadow .35s ease;
}

.condition-card:hover {
    transform: translateY(-9px);
    box-shadow: 0 22px 38px rgba(47,62,70,.13);
}

.condition-card.heart {
    background: #F5E9E4;
    border-color: #EBD6CE;
}

.condition-card.diabetes {
    background: #E8EEE7;
    border-color: #D8E2D7;
}

.condition-card.breast {
    background: #F0EAEF;
    border-color: #E5D6E0;
}

.condition-card.active {
    border: 2px solid var(--sage);
    box-shadow: 0 14px 35px rgba(111,154,134,.22);
}

.condition-icon {
    font-size: 48px;
    margin-bottom: 12px;
    display: inline-block;
}

.condition-icon.heart-icon {
    animation: heartbeat 1.3s infinite;
}

.condition-icon.diabetes-icon {
    animation: dropMove 4s ease-in-out infinite;
}

.condition-icon.breast-icon {
    animation: ribbonMove 4s ease-in-out infinite;
}

.condition-title {
    font-family: 'Fraunces', serif;
    color: var(--slate);
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 8px;
}

.condition-description {
    color: #69766F;
    font-size: 13.5px;
    line-height: 1.65;
}


/* ------------------------------------------------------------
   INPUT FORM
------------------------------------------------------------ */

div[data-testid="stForm"] {
    background: rgba(255,255,255,.85);
    border: 1px solid #E7E3D9;
    border-radius: 24px;
    padding: 28px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(47,62,54,.05);
}

label {
    color: var(--slate) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-baseweb="input"],
div[data-baseweb="select"] {
    border-radius: 12px;
    transition: box-shadow .25s, transform .25s;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    box-shadow: 0 0 0 4px rgba(111,154,134,.22);
    transform: translateY(-2px);
}

div[data-testid="stNumberInput"] input {
    color: var(--slate);
}


/* ------------------------------------------------------------
   BUTTONS
------------------------------------------------------------ */

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #6F9A86, #86B3A0);
    color: white;
    border: 0;
    border-radius: 14px;
    min-height: 48px;
    font-weight: 700;
    font-size: 14px;
    transition: all .25s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    transform: translateY(-3px);
    box-shadow: 0 12px 24px rgba(111,154,134,.35);
}


/* ------------------------------------------------------------
   RESULT CARD
------------------------------------------------------------ */

.result-card {
    background: rgba(255,255,255,.9);
    border: 1px solid #E5E0D6;
    border-radius: 26px;
    padding: 32px;
    margin-top: 20px;
    box-shadow: 0 8px 28px rgba(47,62,54,.06);
    display: flex;
    gap: 28px;
    align-items: center;
    flex-wrap: wrap;
    animation: reveal .8s ease both;
}

.result-symbol {
    width: 125px;
    height: 125px;
    flex: none;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: #E8EEE7;
    font-size: 65px;
    animation: float 4s ease-in-out infinite;
}

.result-symbol.positive {
    background: #F5E9E4;
}

.result-body {
    flex: 1;
    min-width: 260px;
}

.result-kicker {
    color: #89928B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.result-heading {
    font-family: 'Fraunces', serif;
    font-size: 29px;
    font-weight: 800;
    margin-top: 10px;
}

.result-positive {
    color: var(--terracotta);
}

.result-negative {
    color: var(--sage);
}

.result-copy {
    color: #748078;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 10px;
}

.result-meta {
    display: inline-block;
    margin-top: 16px;
    padding: 10px 14px;
    background: var(--bg);
    color: #69766F;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
}


/* ------------------------------------------------------------
   FOOTER
------------------------------------------------------------ */

.app-footer {
    text-align: center;
    border-top: 1px solid #E6E1D7;
    margin-top: 55px;
    padding-top: 25px;
    color: #89928B;
    font-size: 12px;
    line-height: 1.8;
}


/* ------------------------------------------------------------
   RESPONSIVE AND ACCESSIBILITY
------------------------------------------------------------ */

@media (max-width: 768px) {
    .hero {
        padding: 30px 22px;
    }

    .hero-heading {
        font-size: 38px;
    }

    .section-heading {
        font-size: 26px;
    }

    .hero-visual {
        height: 340px;
    }

    .visual-orbit {
        width: 260px;
        height: 260px;
    }

    .visual-orbit-two {
        width: 300px;
        height: 300px;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
    }
}

</style>
"""

st.html(GLOBAL_CSS)


# ============================================================
# 3. ANIMATED BACKGROUND
# ============================================================

st.html("""
<div class="background-glow glow-one"></div>
<div class="background-glow glow-two"></div>
<div class="background-glow glow-three"></div>
""")


# ============================================================
# 4. DATASET CONFIGURATION
# ============================================================

DISEASES = {
    "Heart Disease": {
        "folder": "heart_disease",
        "prefix": "heart_disease",
        "preprocessor": "heart_preprocessor.pkl",
        "csv": "heart_disease_raw.csv",
        "target": "num",
        "categorical": [
            "sex", "cp", "fbs", "restecg",
            "exang", "slope", "ca", "thal"
        ],
    },

    "Diabetes": {
        "folder": "diabetes",
        "prefix": "diabetes",
        "preprocessor": "diabetes_preprocessor.pkl",
        "csv": "diabetes_raw.csv",
        "target": "class",
        "categorical": [
            "gender", "polyuria", "polydipsia",
            "sudden_weight_loss", "weakness",
            "polyphagia", "genital_thrush",
            "visual_blurring", "itching",
            "irritability", "delayed_healing",
            "partial_paresis", "muscle_stiffness",
            "alopecia", "obesity",
        ],
    },

    "Breast Cancer": {
        "folder": "breast_cancer",
        "prefix": "breast_cancer",
        "preprocessor": "breast_preprocessor.pkl",
        "csv": "breast_cancer_raw.csv",
        "target": "Diagnosis",
        "categorical": [],
    },
}


ALGORITHMS = {
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}


HEART_LABELS = {
    "sex": {
        0: "Female",
        1: "Male"
    },
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-Anginal Pain",
        4: "Asymptomatic"
    },
    "fbs": {
        0: "No",
        1: "Yes"
    },
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    },
    "exang": {
        0: "No",
        1: "Yes"
    },
    "slope": {
        1: "Upsloping",
        2: "Flat",
        3: "Downsloping"
    },
    "ca": {
        0: "0 major vessels",
        1: "1 major vessel",
        2: "2 major vessels",
        3: "3 major vessels"
    },
    "thal": {
        3: "Normal",
        6: "Fixed Defect",
        7: "Reversible Defect"
    },
}


# ============================================================
# 5. HEADER
# ============================================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.html("""
    <div class="brand-name">
        <span class="brand-heart">♥</span>
        Health<span>Sense</span>
    </div>
    """)

with header_right:
    st.html("""
    <div style="text-align:right;padding-top:7px;">
        <span class="nav-label">
            <span class="status-dot"></span>
            HEALTHCARE SCREENING
        </span>
    </div>
    """)


# ============================================================
# 6. UNIFIED HERO SECTION
# ============================================================

hero_left, hero_right = st.columns([1.25, 1])

with hero_left:
    st.html("""
    <section class="hero">
        <div class="hero-kicker">
            A more thoughtful approach to health
        </div>

        <div class="hero-heading">
            Your health.<br>
            Your awareness.<br>
            <span class="highlight">Your first step.</span>
        </div>

        <div class="hero-copy">
            Explore machine-learning-based screening for
            heart disease, diabetes and breast cancer using
            structured medical datasets.
        </div>

        <div class="hero-pill">
            03 Screening Areas &nbsp; · &nbsp; 04 ML Algorithms
        </div>
    </section>
    """)

with hero_right:
    st.html("""
    <section class="hero" style="padding:15px 10px;min-height:445px;">

        <div class="hero-visual">

            <div class="visual-orbit-two"></div>
            <div class="visual-orbit"></div>

            <div class="visual-center">
                <div class="visual-heart">♥</div>
            </div>

            <div class="floating-medical medical-heart">
                ♥
            </div>

            <div class="floating-medical medical-glucose">
                💧
            </div>

            <div class="floating-medical medical-ribbon">
                🎗
            </div>

            <div class="floating-medical medical-cell">
                ◉
            </div>

            <div class="visual-caption">
                Small steps. Better awareness.
            </div>

        </div>
    </section>
    """)


# ============================================================
# 7. CONTINUOUS ECG ANIMATION
# ============================================================

ecg_path = "M0 38"

for x in range(0, 1600, 400):
    ecg_path += (
        f" L{x+50} 38"
        f" L{x+70} 32"
        f" L{x+85} 38"
        f" L{x+110} 38"
        f" L{x+125} 48"
        f" L{x+140} 5"
        f" L{x+155} 68"
        f" L{x+170} 38"
        f" L{x+210} 38"
        f" L{x+235} 25"
        f" L{x+255} 38"
    )

st.html(f"""
<div class="ecg-container">
    <div class="ecg-track">
        <svg viewBox="0 0 1600 75" preserveAspectRatio="none">
            <path d="{ecg_path}"></path>
        </svg>
    </div>
</div>
""")


# ============================================================
# 8. PROJECT HIGHLIGHTS
# ============================================================

st.html("""
<div class="section-heading">
    One platform, three screening areas
</div>

<div class="section-copy">
    Explore the medical datasets and trained machine-learning
    models used in this project.
</div>
""")

metric_columns = st.columns(3)

metrics = [
    (3, "Medical datasets"),
    (4, "Classification algorithms"),
    (12, "Trained model experiments"),
]

for idx, (column, (number, label)) in enumerate(
    zip(metric_columns, metrics),
    start=1
):
    with column:
        st.html(f"""
        <div class="metric-card m{idx}">
            <div class="metric-number">{number:02d}</div>
            <div class="metric-label">{label}</div>
        </div>
        """)


# ============================================================
# 9. SELECT SCREENING AREA
# ============================================================

st.html("""
<div class="section-heading">
    Choose your screening area
</div>

<div class="section-copy">
    Select one of the available medical datasets to begin.
</div>
""")

if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = "Heart Disease"


conditions = [
    {
        "name": "Heart Disease",
        "class": "heart",
        "icon": "♥",
        "icon_class": "heart-icon",
        "description": (
            "Heart-related measurements including age, "
            "blood pressure, cholesterol and heart rate."
        ),
    },

    {
        "name": "Diabetes",
        "class": "diabetes",
        "icon": "💧",
        "icon_class": "diabetes-icon",
        "description": (
            "Age, gender and the symptoms included "
            "in the diabetes dataset."
        ),
    },

    {
        "name": "Breast Cancer",
        "class": "breast",
        "icon": "🎗",
        "icon_class": "breast-icon",
        "description": (
            "Cell measurements such as radius, texture, "
            "perimeter and area."
        ),
    },
]


condition_columns = st.columns(3)

for index, condition in enumerate(conditions):

    with condition_columns[index]:

        active = (
            " active"
            if st.session_state.selected_disease == condition["name"]
            else ""
        )

        st.html(f"""
        <div class="condition-card {condition['class']}{active}">

            <div class="condition-icon {condition['icon_class']}">
                {condition['icon']}
            </div>

            <div class="condition-title">
                {condition['name']}
            </div>

            <div class="condition-description">
                {condition['description']}
            </div>

        </div>
        """)

        button_text = (
            "Currently selected ✓"
            if st.session_state.selected_disease == condition["name"]
            else "Select this area"
        )

        if st.button(
            button_text,
            key=f"condition_{index}",
            use_container_width=True
        ):
            st.session_state.selected_disease = condition["name"]
            st.rerun()


disease = st.session_state.selected_disease
selected = DISEASES[disease]


# ============================================================
# 10. LOAD MATCHING MODEL AND PREPROCESSOR
# ============================================================

st.html("""
<div class="section-heading">
    Patient information
</div>

<div class="section-copy">
    Enter the features required by the selected dataset.
    The form uses the original dataset columns.
</div>
""")

algorithm = st.selectbox(
    "Choose a machine learning algorithm",
    list(ALGORITHMS.keys()),
    key="algorithm_choice",
)

model_filename = (
    f"{selected['prefix']}_{ALGORITHMS[algorithm]}.pkl"
)

model_path = MODEL_DIR / model_filename

preprocessor_path = (
    DATA_DIR
    / selected["folder"]
    / selected["preprocessor"]
)

csv_path = (
    DATA_DIR
    / selected["folder"]
    / selected["csv"]
)

required_files = [
    model_path,
    preprocessor_path,
    csv_path
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    st.error(
        "Some required project files are missing:\n\n"
        + "\n".join(missing_files)
    )
    st.stop()


try:
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    raw_df = pd.read_csv(csv_path)

except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


target_column = selected["target"]

if target_column not in raw_df.columns:
    st.error(
        f"Target column '{target_column}' "
        "was not found in the selected dataset."
    )
    st.stop()


feature_columns = [
    column
    for column in raw_df.columns
    if column != target_column
]

if len(feature_columns) == 0:
    st.error("No input features were found in the dataset.")
    st.stop()


# ============================================================
# 11. INPUT FIELD LABELS
# ============================================================

def get_feature_label(feature):

    if disease != "Breast Cancer":
        return feature.replace("_", " ").title()

    suffix_names = {
        "1": "Mean",
        "2": "Standard Error",
        "3": "Worst"
    }

    suffix = feature[-1]
    base_name = feature[:-1].replace("_", " ").title()

    if suffix in suffix_names:
        return f"{base_name} — {suffix_names[suffix]}"

    return feature.replace("_", " ").title()


def get_category_label(feature, value):

    if disease == "Heart Disease":

        labels = HEART_LABELS.get(feature, {})

        try:
            numeric_value = float(value)

            if numeric_value.is_integer():
                numeric_value = int(numeric_value)

            return labels.get(numeric_value, str(value))

        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, str):
        return value.strip().title()

    return str(value)


# ============================================================
# 12. INPUT WIDGET CREATION
# ============================================================

def create_input_widget(feature, input_values):

    series = raw_df[feature]
    label = get_feature_label(feature)

    # Categorical columns
    if feature in selected["categorical"]:

        available_values = series.dropna().unique().tolist()

        if not available_values:
            raise ValueError(
                f"No available values found for {feature}."
            )

        default_value = series.mode().iloc[0]
        default_index = available_values.index(default_value)

        input_values[feature] = st.selectbox(
            label,
            options=available_values,
            index=default_index,
            format_func=lambda value: get_category_label(
                feature,
                value
            ),
            key=f"{disease}_{feature}",
        )

    # Numerical columns
    else:

        numeric_series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if numeric_series.empty:
            raise ValueError(
                f"No numerical values found for {feature}."
            )

        minimum = float(numeric_series.min())
        maximum = float(numeric_series.max())
        default_value = float(numeric_series.median())

        is_integer = pd.api.types.is_integer_dtype(series)

        if is_integer:

            kwargs = {}

            if minimum < maximum:
                kwargs["min_value"] = int(minimum)
                kwargs["max_value"] = int(maximum)

            input_values[feature] = st.number_input(
                label,
                value=int(round(default_value)),
                step=1,
                key=f"{disease}_{feature}",
                **kwargs,
            )

        else:

            kwargs = {}

            if minimum < maximum:
                kwargs["min_value"] = minimum
                kwargs["max_value"] = maximum

            input_values[feature] = st.number_input(
                label,
                value=default_value,
                step=0.01,
                format="%.4f",
                key=f"{disease}_{feature}",
                **kwargs,
            )


# ============================================================
# 13. PATIENT INPUT FORM
# ============================================================

input_values = {}

with st.form(
    key=f"prediction_form_{disease}",
    clear_on_submit=False
):

    if disease == "Breast Cancer":

        measurement_groups = {
            "Mean measurements": [
                f for f in feature_columns if f.endswith("1")
            ],

            "Standard error measurements": [
                f for f in feature_columns if f.endswith("2")
            ],

            "Worst measurements": [
                f for f in feature_columns if f.endswith("3")
            ],
        }

        for group_name, group_features in measurement_groups.items():

            st.markdown(f"### {group_name}")

            left, right = st.columns(2)

            for index, feature in enumerate(group_features):

                with (left if index % 2 == 0 else right):
                    create_input_widget(feature, input_values)

            st.markdown("---")

    else:

        left, right = st.columns(2)

        for index, feature in enumerate(feature_columns):

            with (left if index % 2 == 0 else right):
                create_input_widget(feature, input_values)

    predict_button = st.form_submit_button(
        "Generate screening result",
        use_container_width=True
    )


# ============================================================
# 14. PREDICTION
# ============================================================

if predict_button:

    try:

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        # Apply the exact preprocessing fitted during training
        processed_input = preprocessor.transform(input_df)

        # Predict using the selected trained model
        prediction = int(model.predict(processed_input)[0])

        positive = prediction == 1

        # Result labels for each dataset
        if disease == "Heart Disease":

            result_title = (
                "Possible presence of heart disease"
                if positive
                else "No heart disease predicted"
            )

            result_description = (
                "The selected model predicts the positive class "
                "for the supplied heart-related features."
                if positive
                else
                "The selected model predicts the negative class "
                "for the supplied heart-related features."
            )

            result_icon = "♥"

        elif disease == "Diabetes":

            result_title = (
                "Positive diabetes prediction"
                if positive
                else "Negative diabetes prediction"
            )

            result_description = (
                "The selected model predicts the positive diabetes "
                "class for the supplied features."
                if positive
                else
                "The selected model predicts the negative diabetes "
                "class for the supplied features."
            )

            result_icon = "💧"

        else:

            result_title = (
                "Malignant class predicted"
                if positive
                else "Benign class predicted"
            )

            result_description = (
                "The selected model classifies the supplied cell "
                "measurements as malignant."
                if positive
                else
                "The selected model classifies the supplied cell "
                "measurements as benign."
            )

            result_icon = "🎗"

        result_style = (
            "result-positive"
            if positive
            else "result-negative"
        )

        symbol_style = "positive" if positive else ""

        st.html(f"""
        <div class="section-heading">
            Your screening result
        </div>

        <div class="result-card">

            <div class="result-symbol {symbol_style}">
                {result_icon}
            </div>

            <div class="result-body">

                <div class="result-kicker">
                    MODEL PREDICTION
                </div>

                <div class="result-heading {result_style}">
                    {result_title}
                </div>

                <div class="result-copy">
                    {result_description}
                </div>

                <div class="result-meta">
                    Screening: {disease}
                    &nbsp; | &nbsp;
                    Algorithm: {algorithm}
                </div>

            </div>

        </div>
        """)

        st.warning(
            "This is an educational machine-learning prediction, "
            "not a medical diagnosis. It should not be used to make "
            "healthcare decisions. Please consult a qualified "
            "healthcare professional for medical advice."
        )

    except Exception as error:

        st.error(
            f"Prediction could not be generated. Error details: {error}"
        )


# ============================================================
# 15. FOOTER
# ============================================================

st.html("""
<div class="app-footer">

    <div style="
        font-family:'Fraunces',serif;
        color:#2F3E46;
        font-size:22px;
        font-weight:800;
        margin-bottom:7px;
    ">
        Health<span style="color:#7A9E8E;">Sense</span>
    </div>

    Healthcare · Machine Learning · Data Analytics

    <br><br>

    Developed for educational and research purposes.

    <br>

    Not intended for clinical diagnosis.

</div>
""")