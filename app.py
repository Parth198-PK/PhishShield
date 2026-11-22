import html
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from indicators import (
    analyze,
    CAT_URGENCY, CAT_LINKS, CAT_CREDENTIALS,
    CAT_FINANCIAL, CAT_PRIZE, CAT_THREATS,
)

# set_page_config must be the first Streamlit call
st.set_page_config(page_title="PhishShield", page_icon="🛡️", layout="wide")

# ======================================================================
# Paths, constants, palette
# ======================================================================
BASE_DIR = Path(__file__).parent
MODEL_FILE = BASE_DIR / "models" / "model.joblib"
METRICS_FILE = BASE_DIR / "models" / "metrics.json"

MAX_CHARS = 5000          # longest message we accept
BORDERLINE = (0.35, 0.65)  # scores in this range are flagged as "close call"

# Single source of truth for colors (CSS and charts both use this)
C = {
    "bg":        "#0A0712",  # midnight black
    "card":      "#181126",  # deep purple-black
    "primary":   "#B6FF00",  # neon green
    "secondary": "#A855F7",  # electric purple
    "glow":      "#C084FC",  # purple highlight
    "text":      "#F5F3FF",
    "muted":     "#A49AB8",
    "threat":    "#FF4566",  # danger
}

MODEL_LABELS = {
    "naive_bayes": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
}

esc = html.escape  # always escape user text before putting it in HTML


# ======================================================================
# Theme: CSS + Matplotlib style
# ======================================================================
CSS = f"""
<style>
.stApp {{ background: {C['bg']}; color: {C['text']}; }}
[data-testid="stSidebar"] {{ background: {C['card']}; border-right: 1px solid {C['secondary']}44; }}
[data-testid="stHeader"] {{ background: transparent; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; max-width: 1150px; }}
hr {{ border-color: {C['secondary']}44; }}

/* Typography */
.ps-title {{ font-size: 2.1rem; font-weight: 800; letter-spacing: -0.01em; margin: 0; }}
.ps-title span {{ color: {C['primary']}; text-shadow: 0 0 18px {C['primary']}55; }}
.ps-kicker {{ color: {C['glow']}; font-size: .75rem; letter-spacing: .18em;
              text-transform: uppercase; margin-bottom: .2rem; }}
.ps-sub {{ color: {C['muted']}; margin: .2rem 0 1.2rem 0; }}
.ps-section {{ font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .5rem 0; color: {C['text']}; }}
.ps-muted {{ color: {C['muted']}; font-size: .9rem; }}

/* Cards */
.ps-card {{ background: {C['card']}; border: 1px solid {C['secondary']}55; border-radius: 12px;
            padding: 1rem 1.25rem; margin-bottom: .9rem; box-shadow: 0 0 14px {C['secondary']}22; }}
.ps-card h4 {{ margin: 0; color: {C['glow']}; font-size: 1.02rem; }}
.ps-card p {{ margin: .5rem 0 0 0; }}
.ps-card ul {{ margin: .3rem 0 0 1.1rem; padding: 0; }}
.ps-card li {{ margin-bottom: .35rem; }}
.ps-card-ml   {{ border-left: 3px solid {C['primary']}; }}
.ps-card-rule {{ border-left: 3px solid {C['secondary']}; }}
.ps-row {{ display: flex; justify-content: space-between; align-items: center; gap: .75rem; }}

/* Layer banners: make the two analysis layers visibly different */
.ps-layer {{ margin: 1.8rem 0 .7rem 0; padding-bottom: .5rem; }}
.ps-layer .tag {{ font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }}
.ps-layer .name {{ font-size: 1.3rem; font-weight: 800; margin-top: .1rem; }}
.ps-layer-ml   {{ border-bottom: 1px solid {C['primary']}66; }}
.ps-layer-ml .tag {{ color: {C['primary']}; }}
.ps-layer-rule {{ border-bottom: 1px solid {C['secondary']}88; }}
.ps-layer-rule .tag {{ color: {C['glow']}; }}

/* Badges */
.ps-badge {{ display: inline-block; padding: .25rem .75rem; border-radius: 999px;
             font-weight: 700; letter-spacing: .04em; font-size: .8rem; white-space: nowrap; }}
.ps-badge-threat  {{ color: {C['threat']}; border: 1px solid {C['threat']};
                     background: {C['threat']}18; box-shadow: 0 0 10px {C['threat']}44; }}
.ps-badge-ok      {{ color: {C['primary']}; border: 1px solid {C['primary']};
                     background: {C['primary']}14; box-shadow: 0 0 10px {C['primary']}33; }}
.ps-badge-caution {{ color: {C['glow']}; border: 1px solid {C['glow']}; background: {C['glow']}18; }}
.ps-badge-low     {{ color: {C['muted']}; border: 1px solid {C['muted']}88; background: {C['muted']}14; }}

/* Probability bar */
.ps-bignum {{ font-size: 2.6rem; font-weight: 800; line-height: 1; margin: .5rem 0 .2rem 0; }}
.ps-bar {{ height: 10px; background: {C['bg']}; border: 1px solid {C['secondary']}55;
           border-radius: 999px; overflow: hidden; margin: .6rem 0 .3rem 0; }}
.ps-bar-fill {{ height: 100%; border-radius: 999px; }}

/* Chips (words and evidence) */
.ps-chip {{ display: inline-block; font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size: .82rem; padding: .15rem .6rem; border-radius: 6px; margin: .2rem .3rem .2rem 0; }}
.ps-chip b {{ margin-left: .35rem; }}
.ps-chip-spam {{ color: {C['threat']}; background: {C['threat']}18; border: 1px solid {C['threat']}66; }}
.ps-chip-ham  {{ color: {C['primary']}; background: {C['primary']}14; border: 1px solid {C['primary']}55; }}
.ps-chip-ev   {{ color: {C['text']}; background: {C['secondary']}26; border: 1px solid {C['secondary']}88; }}

/* Highlighted phrases inside the message */
mark.ps-hit {{ background: {C['threat']}33; color: {C['text']};
               border-bottom: 2px solid {C['threat']}; border-radius: 3px; padding: 0 2px; }}
.ps-msg {{ line-height: 1.7; word-break: break-word; }}

/* Notes and disclaimer */
.ps-note {{ background: {C['card']}; border: 1px dashed {C['glow']}99; border-radius: 12px;
            padding: .9rem 1.2rem; margin: 1.2rem 0; }}
.ps-disclaimer {{ background: {C['threat']}12; border: 1px solid {C['threat']}66;
                  border-radius: 12px; padding: .9rem 1.2rem; margin-top: 1.4rem; }}

/* Dashboard stat cards and table */
.ps-stat {{ background: {C['card']}; border: 1px solid {C['secondary']}55; border-radius: 12px;
            padding: .9rem 1rem; box-shadow: 0 0 14px {C['secondary']}22; }}
.ps-stat-label {{ color: {C['muted']}; font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }}
.ps-stat-value {{ color: {C['primary']}; font-size: 1.7rem; font-weight: 800; margin: .15rem 0; }}
table.ps-table {{ width: 100%; border-collapse: collapse; background: {C['card']}; border-radius: 12px;
                  overflow: hidden; font-size: .92rem; }}
table.ps-table th {{ background: {C['secondary']}33; color: {C['text']}; text-align: left; padding: .6rem .8rem; }}
table.ps-table td {{ padding: .6rem .8rem; border-top: 1px solid {C['secondary']}33; color: {C['text']}; }}

/* Inputs and buttons */
.stTextArea textarea {{ background: {C['card']}; color: {C['text']}; border: 1px solid {C['secondary']}88; }}
.stTextArea textarea:focus {{ border-color: {C['primary']}; box-shadow: 0 0 10px {C['primary']}55; }}
.stButton > button[data-testid="stBaseButton-secondary"] {{
    background: {C['card']}; color: {C['text']}; border: 1px solid {C['secondary']}88; }}
.stButton > button[data-testid="stBaseButton-secondary"]:hover {{
    border-color: {C['primary']}; color: {C['primary']}; }}
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {{
    background: {C['primary']}; color: {C['bg']}; font-weight: 700; border: none;
    box-shadow: 0 0 14px {C['primary']}55; }}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: {C['primary']}; color: {C['bg']}; box-shadow: 0 0 22px {C['primary']}88; }}
</style>
"""


def apply_theme():
    """Inject the CSS and give Matplotlib the same palette."""
    st.markdown(CSS, unsafe_allow_html=True)
    plt.rcParams.update({
        "figure.facecolor": C["card"], "axes.facecolor": C["card"],
        "axes.edgecolor": C["secondary"], "axes.labelcolor": C["text"],
        "text.color": C["text"], "xtick.color": C["muted"], "ytick.color": C["muted"],
        "grid.color": C["secondary"], "grid.alpha": 0.25, "font.size": 10,
    })


def html_block(markup):
    """Render an HTML string. Keep markup free of blank lines and leading indentation."""
    st.markdown(markup, unsafe_allow_html=True)


def section_title(text):
    html_block(f'<div class="ps-section">{esc(text)}</div>')


def layer_banner(kind, tag, name, description):
    css = "ps-layer-ml" if kind == "ml" else "ps-layer-rule"
    html_block(
        f'<div class="ps-layer {css}"><div class="tag">{esc(tag)}</div>'
        f'<div class="name">{esc(name)}</div>'
        f'<div class="ps-muted">{esc(description)}</div></div>'
    )


def show_fig(fig):
    """Display a Matplotlib figure and free its memory."""
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ======================================================================
# Loading the saved model and metrics
# ======================================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the saved sklearn Pipeline (TF-IDF + classifier) once.
    Also pre-compute the vocabulary and per-word weights used for the
    'which words influenced the model' explanation."""
    model = joblib.load(MODEL_FILE)
    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    words = np.array(vectorizer.get_feature_names_out())

    if hasattr(clf, "coef_"):  # Logistic Regression
        weights = clf.coef_[0]
    else:                      # Naive Bayes: log P(word|spam) - log P(word|ham)
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

    return {"model": model, "words": words, "weights": weights,
            "model_name": type(clf).__name__}


@st.cache_data(show_spinner=False)
def load_metrics():
    """Read the metrics saved by train_model.py."""
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def try_load_model():
    """Returns (artifacts, error_message). Never raises."""
    if not MODEL_FILE.exists():
        return None, "Trained model not found. Run `python train_model.py` first."
    try:
        return load_artifacts(), None
    except Exception as error:
        return None, f"Could not load the model: {error}"


def try_load_metrics():
    """Returns (metrics, error_message). Never raises."""
    if not METRICS_FILE.exists():
        return None, "Metrics file not found. Run `python train_model.py` first."
    try:
        return load_metrics(), None
    except Exception as error:
        return None, f"Could not read metrics.json: {error}"


def model_label(key):
    return MODEL_LABELS.get(key, str(key).replace("_", " ").title())


def fmt_int(value):
    return f"{value:,}" if isinstance(value, (int, float)) else "n/a"


def fmt_pct(value):
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "n/a"


# ======================================================================
# Analyzer helpers
# ======================================================================

CATEGORY_TIPS = {
    CAT_URGENCY: "Pause before acting. Real organisations give you time; pressure to hurry is a scam tactic.",
    CAT_LINKS: "Do not tap the link. Type the official website address yourself or use the official app.",
    CAT_CREDENTIALS: "Never share an OTP, PIN, password or CVV. Genuine bank or company staff will never ask for it.",
    CAT_FINANCIAL: "Do not send money or card/bank details. Verify any payment request in the official app or on a number you trust.",
    CAT_PRIZE: "You cannot win a contest you never entered. Ignore it and never pay a fee to 'claim' a prize.",
    CAT_THREATS: "Threats are a pressure tactic. Check your account status directly with the organisation, not via the message.",
}
CONCERN_TIPS = [
    "Do not reply, click links, or call numbers given in the message.",
    "Contact the organisation using its official app, website, or the phone number printed on your card or statement.",
    "If you already shared an OTP, password or card details, contact your bank immediately and change your passwords.",
    "Block the sender and report the scam. In India: cybercrime.gov.in or helpline 1930. Elsewhere, use your country's cybercrime reporting service.",
]
CALM_TIPS = [
    "No strong warning signs were found, but that is not proof the message is safe.",
    "Be careful with unexpected messages that contain links or ask for personal information.",
    "When in doubt, verify through the organisation's official app, website or phone number.",
]


def predict_message(message, art):
    """Spam probability (0 to 1) from the saved model."""
    clf = art["model"].named_steps["clf"]
    spam_column = list(clf.classes_).index(1)  # label 1 = spam
    return float(art["model"].predict_proba([message])[0][spam_column])


def word_contributions(message, art, top_n=8):
    """Words in the message that pushed the model toward spam or legitimate.
    contribution = word's TF-IDF value in this message x model's learned weight.
    (The model's baseline/intercept is not included.)"""
    vectorizer = art["model"].named_steps["tfidf"]
    row = vectorizer.transform([message]).tocoo()
    pairs = [(str(art["words"][col]), float(val * art["weights"][col]))
             for col, val in zip(row.col, row.data)]
    toward_spam = sorted([p for p in pairs if p[1] > 0], key=lambda p: -p[1])[:top_n]
    toward_ham = sorted([p for p in pairs if p[1] < 0], key=lambda p: p[1])[:top_n]
    return toward_spam, toward_ham, len(pairs)


def highlight_text(text, spans):
    """Return HTML of the message with matched phrases wrapped in <mark>."""
    parts, pos = [], 0
    for start, end in spans:
        parts.append(esc(text[pos:start]))
        parts.append(f'<mark class="ps-hit">{esc(text[start:end])}</mark>')
        pos = end
    parts.append(esc(text[pos:]))
    return "".join(parts).replace("\n", "<br>")


def severity_badge(severity):
    css = {"High": "ps-badge-threat", "Medium": "ps-badge-caution"}.get(severity, "ps-badge-low")
    return f'<span class="ps-badge {css}">{esc(severity.upper())}</span>'


def level_badge(level):
    if level == "None":
        return '<span class="ps-badge ps-badge-ok">NO INDICATORS FOUND</span>'
    css = {"High": "ps-badge-threat", "Medium": "ps-badge-caution"}.get(level, "ps-badge-low")
    return f'<span class="ps-badge {css}">{esc(level.upper())} EVIDENCE</span>'


# ----------------------------------------------------------------------
# Layer 1: ML model
# ----------------------------------------------------------------------
def render_ml_layer(message, art):
    """Show the ML model's verdict. Returns the spam probability."""
    layer_banner("ml", "Layer 1 · Machine learning", "Model prediction",
                 "A trained classifier (TF-IDF + " + art["model_name"] + ") scored the wording of the message.")

    p_spam = predict_message(message, art)
    is_spam = p_spam >= 0.5
    confidence = p_spam if is_spam else 1 - p_spam
    badge = ('<span class="ps-badge ps-badge-threat">MODEL VERDICT: SUSPICIOUS / SPAM</span>' if is_spam
             else '<span class="ps-badge ps-badge-ok">MODEL VERDICT: LEGITIMATE</span>')
    bar_color = C["threat"] if is_spam else C["primary"]
    close_call = BORDERLINE[0] < p_spam < BORDERLINE[1]
    close_note = ('<p class="ps-muted"><b>Close call:</b> the score is near the decision boundary, '
                  'so treat this verdict as low confidence.</p>') if close_call else ""

    html_block(
        '<div class="ps-card ps-card-ml">'
        f'<div class="ps-row"><h4>Model score</h4>{badge}</div>'
        f'<div class="ps-bignum" style="color:{bar_color}">{p_spam * 100:.1f}%</div>'
        '<div class="ps-muted">spam-like score (probability estimate from the model)</div>'
        f'<div class="ps-bar"><div class="ps-bar-fill" style="width:{p_spam * 100:.1f}%;background:{bar_color}"></div></div>'
        f'<div class="ps-muted">Model confidence in this verdict: {confidence * 100:.1f}%</div>'
        f'{close_note}'
        '<p class="ps-muted">This is a statistical estimate, not a certainty. The model was trained on older SMS '
        'data, and its probabilities can be over- or under-confident.</p>'
        '</div>'
    )

    # Words that influenced the model
    spam_words, ham_words, n_known = word_contributions(message, art)
    if n_known == 0:
        body = '<p class="ps-muted">None of the words in this message are in the model\'s vocabulary.</p>'
    else:
        spam_chips = "".join(f'<span class="ps-chip ps-chip-spam">{esc(w)}<b>+{v:.2f}</b></span>'
                             for w, v in spam_words) or '<span class="ps-muted">none</span>'
        ham_chips = "".join(f'<span class="ps-chip ps-chip-ham">{esc(w)}<b>{v:.2f}</b></span>'
                            for w, v in ham_words) or '<span class="ps-muted">none</span>'
        body = (f'<p class="ps-muted" style="margin-top:.6rem">Pushed toward SPAM</p><div>{spam_chips}</div>'
                f'<p class="ps-muted" style="margin-top:.6rem">Pushed toward LEGITIMATE</p><div>{ham_chips}</div>')
    html_block(
        '<div class="ps-card ps-card-ml"><h4>Words that influenced the model</h4>'
        '<div class="ps-muted">Contribution = the word\'s TF-IDF value in this message × the weight the model learned. '
        'Positive numbers push toward spam, negative toward legitimate. Single words and two-word phrases are both used.</div>'
        f'{body}</div>'
    )
    return p_spam


# ----------------------------------------------------------------------
# Layer 2: rule-based indicators
# ----------------------------------------------------------------------
def render_rule_layer(message):
    """Show rule-based indicators. Returns the dictionary from analyze()."""
    layer_banner("rule", "Layer 2 · Rule-based checks", "Scam indicators",
                 "Keyword and pattern rules. This is NOT machine learning and does not affect the model's score.")

    result = analyze(message)
    findings = result["findings"]

    html_block(
        '<div class="ps-card ps-card-rule">'
        f'<div class="ps-row"><h4>Indicator evidence level</h4>{level_badge(result["evidence_level"])}</div>'
        f'<p>{esc(result["summary"])}</p>'
        f'<p class="ps-muted">{len(findings)} categor{"y" if len(findings) == 1 else "ies"} matched · '
        f'rule points: {result["score"]} (Low = 1, Medium = 2, High = 3 per category)</p>'
        '</div>'
    )

    if findings:
        html_block(
            '<div class="ps-card ps-card-rule"><h4>Your message with matched phrases highlighted</h4>'
            f'<p class="ps-msg">{highlight_text(message, result["highlight_spans"])}</p></div>'
        )
        for f in findings:
            chips = "".join(f'<span class="ps-chip ps-chip-ev">{esc(e)}</span>' for e in f["evidence"])
            html_block(
                '<div class="ps-card ps-card-rule">'
                f'<div class="ps-row"><h4>{esc(f["category"])}</h4>{severity_badge(f["severity"])}</div>'
                f'<div style="margin-top:.5rem"><span class="ps-muted">Matched text: </span>{chips}</div>'
                f'<p class="ps-muted">{esc(f["explanation"])}</p>'
                '</div>'
            )
    return result


# ----------------------------------------------------------------------
# Comparison note, tips, disclaimer
# ----------------------------------------------------------------------
def comparison_text(p_spam, rules):
    """Plain-language note on how the two layers relate. NOT a combined score."""
    model_flags = p_spam >= 0.5
    level = rules["evidence_level"]
    rules_strong = level in ("Medium", "High")

    if model_flags and rules_strong:
        return "Both layers raise concerns: the model scored the message as suspicious and the rules found scam-style language."
    if model_flags and level == "None":
        return ("The model flagged this message, but no rule matched. The model learns wording patterns from its training data, "
                "while the rules only cover specific listed phrases. Treat the message with caution.")
    if model_flags:
        return "The model flagged this message and the rules found only weak signals. Treat the message with caution."
    if rules_strong:
        return ("The layers disagree: the model scored the message as legitimate, but the rules found scam-style language. "
                "The model was trained on older SMS data and can miss modern scams, so verify this message before acting on it.")
    return "Neither layer raised a strong concern. That still does not guarantee the message is safe."


def render_comparison_and_tips(p_spam, rules):
    if p_spam is not None and rules is not None:
        html_block(
            '<div class="ps-note"><b style="color:' + C["glow"] + '">How the two layers compare</b> '
            '<span class="ps-muted">(a reading aid, not a combined score)</span>'
            f'<p style="margin:.4rem 0 0 0">{esc(comparison_text(p_spam, rules))}</p></div>'
        )

    concerned = (p_spam is not None and p_spam >= 0.5) or \
                (rules is not None and rules["evidence_level"] in ("Medium", "High"))
    tips = []
    if concerned:
        if rules is not None:
            for category in rules["categories_found"]:
                tip = CATEGORY_TIPS.get(category)
                if tip and tip not in tips:
                    tips.append(tip)
        tips += CONCERN_TIPS
    else:
        tips = CALM_TIPS

    items = "".join(f"<li>{esc(t)}</li>" for t in tips)
    title = "Recommended safety steps" if concerned else "General safety reminders"
    html_block(f'<div class="ps-card"><h4>{title}</h4><ul>{items}</ul></div>')


def render_disclaimer():
    html_block(
        '<div class="ps-disclaimer"><b style="color:' + C["threat"] + '">Disclaimer.</b> '
        'PhishShield gives a statistical estimate and rule-based hints. <b>It is not a guarantee.</b> '
        'A "Legitimate" result does not mean a message is safe, and a "Suspicious" result does not prove fraud. '
        'When in doubt, verify through official channels, and do not rely on this tool as your only protection.</div>'
    )


# ======================================================================
# Page 1: Message Analyzer
# ======================================================================
def page_analyzer(art, model_error):
    html_block('<div class="ps-kicker">AI-powered scam detection</div>'
               '<div class="ps-title">Message <span>Analyzer</span></div>'
               '<div class="ps-sub">Paste a suspicious SMS or email text. You get two separate opinions: '
               'a trained ML model and transparent rule-based checks.</div>')

    if model_error:
        st.error(model_error)
        return

    st.text_area("Message to scan", key="message_input", height=170, max_chars=MAX_CHARS,
                 placeholder="Paste the message here...")

    scan = st.button("🔍 Scan message", type="primary")

    if not scan:
        render_disclaimer()
        return

    message = st.session_state.get("message_input", "").strip()
    if not message:
        st.warning("Please paste a message first. The text box is empty.")
        return

    # --- Layer 1: ML model (errors here do not stop Layer 2) ---
    p_spam = None
    try:
        p_spam = render_ml_layer(message, art)
    except Exception as error:
        st.error(f"The ML model could not score this message: {error}")

    # --- Layer 2: rules ---
    rules = None
    try:
        rules = render_rule_layer(message)
    except Exception as error:
        st.error(f"The rule-based checks failed: {error}")

    render_comparison_and_tips(p_spam, rules)
    render_disclaimer()


# ======================================================================
# Page 2: Dashboard (all numbers come from metrics.json)
# ======================================================================
def stat_card(label, value, sub=""):
    return (f'<div class="ps-stat"><div class="ps-stat-label">{esc(label)}</div>'
            f'<div class="ps-stat-value">{esc(value)}</div><div class="ps-muted">{esc(sub)}</div></div>')


def bar_with_labels(ax, labels, values, colors, fmt):
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt(value),
                ha="center", va="bottom", color=C["text"], fontsize=9)
    style_axes(ax)


def word_chart(words, color):
    names = [w["word"] for w in words][::-1]          # reversed so the biggest is on top
    values = [abs(w["weight"]) for w in words][::-1]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.barh(names, values, color=color)
    ax.set_xlabel("Weight strength (absolute value)")
    style_axes(ax)
    show_fig(fig)


def page_dashboard(metrics, metrics_error):
    html_block('<div class="ps-kicker">Model transparency</div>'
               '<div class="ps-title">Dataset &amp; model <span>dashboard</span></div>'
               '<div class="ps-sub">Every number below was computed by train_model.py and read from models/metrics.json.</div>')
    if metrics_error:
        st.error(metrics_error)
        return

    ds = metrics.get("dataset", {})
    models = metrics.get("models", {})
    selected = metrics.get("selected_model")
    clean = ds.get("class_counts_clean", {})
    test_counts = ds.get("test_class_counts", {})

    # ---- Dataset overview ----
    section_title("Dataset overview")
    cards = st.columns(5)
    cards[0].markdown(stat_card("Raw messages", fmt_int(ds.get("total_messages")), "UCI SMS Spam Collection"), unsafe_allow_html=True)
    cards[1].markdown(stat_card("Duplicates removed", fmt_int(ds.get("duplicates_removed")), "before splitting"), unsafe_allow_html=True)
    cards[2].markdown(stat_card("After cleaning", fmt_int(ds.get("messages_after_cleaning")), "unique messages"), unsafe_allow_html=True)
    cards[3].markdown(stat_card("Training set", fmt_int(ds.get("train_size")), "80% (stratified)"), unsafe_allow_html=True)
    cards[4].markdown(stat_card("Test set", fmt_int(ds.get("test_size")), "20%, used once"), unsafe_allow_html=True)

    # ---- Class distribution + message length ----
    section_title("Class distribution and message length")
    left, right = st.columns(2)
    with left:
        if clean:
            ham, spam = clean.get("ham", 0), clean.get("spam", 0)
            total = max(ham + spam, 1)
            fig, ax = plt.subplots(figsize=(5, 3.4))
            bar_with_labels(ax, ["Legitimate (ham)", "Spam"], [ham, spam], [C["primary"], C["threat"]],
                            lambda v: f"{int(v):,} ({v / total * 100:.1f}%)")
            ax.set_title("Messages per class (after removing duplicates)", color=C["text"], fontsize=10)
            ax.set_ylim(0, max(ham, spam) * 1.15)
            show_fig(fig)
            st.caption("The classes are imbalanced, so accuracy alone can look good "
                       "even for a weak spam detector. Precision and recall matter more.")
    with right:
        lengths = ds.get("avg_length_chars", {})
        if lengths:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            bar_with_labels(ax, ["Legitimate (ham)", "Spam"], [lengths.get("ham", 0), lengths.get("spam", 0)],
                            [C["primary"], C["threat"]], lambda v: f"{v:.0f} chars")
            ax.set_title("Average message length", color=C["text"], fontsize=10)
            ax.set_ylim(0, max(lengths.values()) * 1.15)
            show_fig(fig)

    # ---- Model comparison ----
    section_title("Model comparison")
    if models:
        rows = []
        for key, m in models.items():
            rows.append({
                "Model": model_label(key) + ("  ★ selected" if key == selected else ""),
                "CV F1 (selection)": fmt_pct(m.get("cv_f1")),
                "Test accuracy": fmt_pct(m.get("accuracy")),
                "Test precision (spam)": fmt_pct(m.get("precision")),
                "Test recall (spam)": fmt_pct(m.get("recall")),
                "Test F1 (spam)": fmt_pct(m.get("f1")),
            })
        table = pd.DataFrame(rows).to_html(index=False, border=0, classes="ps-table", escape=True)
        html_block(table)

        criterion = metrics.get("selection_criterion", "cross-validation on training data")
        st.caption(f"Selected by: {criterion}. The test-set columns were computed once, after selection, for reporting only.")
        params = "; ".join(f"{model_label(k)}: {m.get('best_params', {})}" for k, m in models.items())
        st.caption(f"Settings chosen by cross-validation: {params}")

        # Grouped bar chart of the four test metrics
        metric_keys = [("accuracy", "Accuracy"), ("precision", "Precision"),
                       ("recall", "Recall"), ("f1", "F1")]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        names = list(models.keys())
        x = np.arange(len(metric_keys))
        width = 0.8 / max(len(names), 1)
        palette = [C["primary"], C["secondary"], C["glow"]]
        all_values = [models[n].get(k, 0) for n in names for k, _ in metric_keys]
        lower = max(0.0, np.floor((min(all_values) - 0.05) * 20) / 20)
        for i, name in enumerate(names):
            vals = [models[name].get(k, 0) for k, _ in metric_keys]
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
                          label=model_label(name), color=palette[i % len(palette)])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v * 100:.1f}",
                        ha="center", va="bottom", fontsize=8, color=C["text"])
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metric_keys])
        ax.set_ylim(lower, 1.04)
        ax.set_ylabel("Score (spam = positive class)")
        ax.legend(facecolor=C["card"], edgecolor=C["secondary"], labelcolor=C["text"], fontsize=8, loc="lower right")
        style_axes(ax)
        show_fig(fig)
        st.caption(f"Note: the y-axis starts at {lower:.2f}, not 0, so small differences look bigger than they are.")

        n_spam_test = test_counts.get("spam")
        if n_spam_test:
            st.caption(f"The test set has only {n_spam_test} spam messages, so one message changes recall by about "
                       f"{100 / n_spam_test:.1f} percentage points. Do not over-interpret small gaps between models.")
    else:
        st.info("No model results found in metrics.json.")

    # ---- Confusion matrix ----
    section_title("Confusion matrix (test set)")
    keys = [k for k in models if models[k].get("confusion_matrix")]
    cm_source, cm_key = None, None
    if keys:
        default_index = keys.index(selected) if selected in keys else 0
        cm_key = st.radio("Model", keys, index=default_index, format_func=model_label, horizontal=True)
        cm_source = models[cm_key]["confusion_matrix"]
    elif metrics.get("confusion_matrix"):
        cm_source, cm_key = metrics["confusion_matrix"], selected

    if cm_source:
        cm = np.array(cm_source)
        left, right = st.columns([1, 1])
        with left:
            cmap = LinearSegmentedColormap.from_list("ps", [C["card"], C["secondary"]])
            fig, ax = plt.subplots(figsize=(4.4, 3.8))
            ax.imshow(cm, cmap=cmap)
            cell_names = [["Correct: legitimate", "False alarm"], ["Missed spam", "Correct: spam"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i - 0.1, f"{cm[i, j]:,}", ha="center", va="center",
                            color=C["text"], fontsize=18, fontweight="bold")
                    ax.text(j, i + 0.25, cell_names[i][j], ha="center", va="center",
                            color=C["muted"], fontsize=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Legitimate", "Spam"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Legitimate", "Spam"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(model_label(cm_key), color=C["text"], fontsize=10)
            show_fig(fig)
        with right:
            html_block(
                '<div class="ps-card"><h4>How to read this</h4><ul>'
                f'<li><b>{cm[0][1]:,}</b> legitimate message(s) were wrongly flagged as spam (false alarms).</li>'
                f'<li><b>{cm[1][0]:,}</b> spam message(s) were missed and passed as legitimate.</li>'
                f'<li><b>{cm[1][1]:,}</b> spam messages were caught and <b>{cm[0][0]:,}</b> legitimate messages were correctly passed.</li>'
                '</ul><p class="ps-muted">Both error types matter: missed spam is risky, '
                'and too many false alarms make a tool annoying.</p></div>'
            )

    # ---- Top words ----
    spam_words = metrics.get("top_spam_words", [])
    ham_words = metrics.get("top_ham_words", [])
    if spam_words or ham_words:
        section_title("Most informative words (selected model)")
        left, right = st.columns(2)
        with left:
            st.markdown("**Words pushing toward spam**")
            if spam_words:
                word_chart(spam_words, C["threat"])
        with right:
            st.markdown("**Words pushing toward legitimate**")
            if ham_words:
                word_chart(ham_words, C["primary"])
        st.caption("Read from the saved model's learned weights. These show patterns in the training data, not rules about real scams.")

    # ---- Training info ----
    info = metrics.get("trained_with", {})
    if info:
        st.caption(
            f"Trained {str(info.get('trained_at', 'n/a')).replace('T', ' ')} · "
            f"scikit-learn {info.get('sklearn_version', 'n/a')} · "
            f"Python {info.get('python_version', 'n/a')} · random seed {info.get('random_state', 'n/a')}"
        )


# ======================================================================
# Page 3: About
# ======================================================================
def page_about(metrics):
    ds = (metrics or {}).get("dataset", {})
    selected = (metrics or {}).get("selected_model")
    selected_text = model_label(selected) if selected else "the model selected during training"
    total = fmt_int(ds.get("total_messages"))

    html_block('<div class="ps-kicker">Project</div>'
               '<div class="ps-title">About <span>PhishShield</span></div>'
               '<div class="ps-sub">AI-powered digital scam detection · Fundamentals of AI and Machine Learning</div>')

    html_block(
        '<div class="ps-card"><h4>Overview</h4>'
        '<p>PhishShield helps people judge whether a text message or email might be a scam. '
        'It gives two independent opinions and keeps them separate:</p><ul>'
        f'<li><b style="color:{C["primary"]}">Layer 1, machine learning:</b> a TF-IDF text model trained on a labeled dataset '
        f'({esc(selected_text)} was selected by cross-validation) returns a spam probability.</li>'
        f'<li><b style="color:{C["glow"]}">Layer 2, rule-based checks:</b> transparent keyword and pattern rules look for urgency, '
        'threats, suspicious links, OTP/password requests, financial requests and prize language.</li>'
        '</ul><p class="ps-muted">The rules never change the model\'s score, and the two are never merged into one number.</p></div>'
    )

    html_block(
        '<div class="ps-card"><h4>How the model was built</h4><ul>'
        f'<li>Dataset: UCI SMS Spam Collection ({esc(total)} labeled messages before cleaning).</li>'
        '<li>Exact duplicate messages were removed before splitting, so no message appears in both training and test data.</li>'
        '<li>Stratified 80/20 train/test split (random seed 42).</li>'
        '<li>TF-IDF features with single words and two-word phrases.</li>'
        '<li>Multinomial Naive Bayes and Logistic Regression were compared with 5-fold cross-validation on the training data only; '
        'the test set was used once for final reporting.</li>'
        '<li>Evaluation: accuracy, precision, recall, F1 and confusion matrix (see the Dashboard).</li></ul></div>'
    )

    section_title("Technology stack")
    stack = [
        ("Python", "Language; the rule engine uses only the standard library (re)"),
        ("Streamlit", "Web interface"),
        ("scikit-learn", "TF-IDF, classifiers, evaluation"),
        ("pandas + NumPy", "Data processing"),
        ("Matplotlib", "Dashboard charts"),
        ("joblib", "Saving and loading the trained model"),
    ]
    cols = st.columns(3)
    for i, (name, desc) in enumerate(stack):
        cols[i % 3].markdown(stat_card(name, "", desc).replace('<div class="ps-stat-value"></div>', ""), unsafe_allow_html=True)

    html_block(
        '<div class="ps-card" style="margin-top:1rem"><h4>Limitations</h4><ul>'
        '<li><b>Old, narrow training data.</b> The dataset is English SMS from the 2000s. The model can be weaker on modern scams '
        'such as UPI/KYC alerts, delivery scams, or long phishing emails.</li>'
        '<li><b>Probabilities are estimates.</b> The score is not a calibrated certainty and can be over- or under-confident.</li>'
        '<li><b>Rules are simple.</b> They match listed phrases and patterns, so new wording can slip through, and normal messages '
        '(for example a genuine OTP notice) can trigger weak signals.</li>'
        '<li><b>Links are judged by their shape only.</b> There is no web lookup, so a dangerous link with a clean-looking address is not detected.</li>'
        '<li><b>Small test set.</b> Only a limited number of spam messages were held out, so metrics are rough estimates.</li>'
        '<li><b>Text only.</b> Sender identity, images, attachments and phone numbers are not analyzed.</li></ul></div>'
    )
    render_disclaimer()


# ======================================================================
# Main: sidebar navigation and page routing
# ======================================================================
def main():
    apply_theme()
    art, model_error = try_load_model()
    metrics, metrics_error = try_load_metrics()

    with st.sidebar:
        html_block('<div class="ps-title" style="font-size:1.5rem">🛡️ Phish<span>Shield</span></div>'
                   '<div class="ps-muted" style="margin-bottom:1rem">Scam detection system</div>')
        page = st.radio("Navigation", ["Message Analyzer", "Dashboard", "About"],
                        label_visibility="collapsed")
        st.divider()
        if art and metrics:
            trained = str(metrics.get("trained_with", {}).get("trained_at", "n/a")).replace("T", " ")
            st.caption(f"Model: {model_label(metrics.get('selected_model'))}")
            st.caption(f"Trained: {trained}")
        elif model_error:
            st.caption("Model not loaded.")
        st.caption("Predictions are estimates, not guarantees.")

    if page == "Message Analyzer":
        page_analyzer(art, model_error)
    elif page == "Dashboard":
        page_dashboard(metrics, metrics_error)
    else:
        page_about(metrics)


main()