"""
Run with:
streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

from backend import get_backend


st.set_page_config(
    page_title="AI Scream Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(0, 214, 255, 0.14), transparent 28%),
            radial-gradient(circle at bottom left, rgba(0, 255, 163, 0.10), transparent 25%),
            linear-gradient(150deg, #02040a 0%, #07111f 52%, #03060d 100%);
        color: #ebf4ff;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }
    .glass-card {
        background: rgba(6, 13, 24, 0.86);
        border: 1px solid rgba(100, 170, 255, 0.16);
        border-radius: 26px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset, 0 24px 80px rgba(0, 0, 0, 0.38);
        backdrop-filter: blur(14px);
    }
    .hero-card {
        padding: 1.7rem 2rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.7rem;
        font-weight: 900;
        letter-spacing: 0.03em;
        line-height: 1.05;
    }
    .hero-subtitle {
        margin-top: 0.35rem;
        color: #87a4c7;
        font-size: 1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .hero-badge {
        margin-top: 1rem;
        display: inline-block;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(0, 247, 167, 0.28);
        background: rgba(0, 247, 167, 0.12);
        color: #bdfbe5;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
    }
    .card {
        padding: 1.3rem 1.35rem;
        height: 100%;
    }
    .section-label {
        color: #7890b0;
        font-size: 0.75rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        margin-bottom: 0.85rem;
    }
    .status-shell {
        border-radius: 24px;
        padding: 1.5rem 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        transition: all 0.25s ease;
    }
    .status-normal {
        background: linear-gradient(135deg, rgba(0, 255, 163, 0.14), rgba(0, 125, 82, 0.10));
        box-shadow: 0 0 30px rgba(0,255,163,0.18);
    }
    .status-warning {
        background: linear-gradient(135deg, rgba(255, 203, 70, 0.16), rgba(160, 96, 0, 0.10));
        box-shadow: 0 0 30px rgba(255,203,70,0.18);
    }
    .status-alert {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.18), rgba(136, 0, 0, 0.14));
        box-shadow: 0 0 34px rgba(255,68,68,0.26);
    }
    .status-text {
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: 0.08em;
    }
    .status-caption {
        margin-top: 0.45rem;
        font-size: 0.95rem;
        color: #9eb4d2;
    }
    .gauge-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 0.3rem 0 0.8rem;
    }
    .gauge {
        width: 190px;
        height: 190px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: conic-gradient(from 180deg, #16ff9d 0deg, #ffd84f 180deg, #ff4c4c 360deg);
        position: relative;
        box-shadow: 0 0 34px rgba(0, 214, 255, 0.12);
    }
    .gauge::before {
        content: "";
        position: absolute;
        inset: 14px;
        border-radius: 50%;
        background: radial-gradient(circle, #07101d 30%, #03070e 100%);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .gauge-fill {
        position: absolute;
        inset: 0;
        border-radius: 50%;
    }
    .gauge-center {
        position: relative;
        z-index: 2;
        text-align: center;
    }
    .gauge-value {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1;
    }
    .gauge-label {
        margin-top: 0.3rem;
        font-size: 0.8rem;
        color: #87a4c7;
        text-transform: uppercase;
        letter-spacing: 0.14em;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.9rem;
        margin-top: 0.8rem;
    }
    .metric-tile {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(11, 20, 34, 0.92);
        border: 1px solid rgba(120, 144, 176, 0.12);
    }
    .metric-name {
        color: #7890b0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 800;
    }
    .metric-num {
        margin-top: 0.35rem;
        font-size: 1.5rem;
        font-weight: 800;
    }
    .signal-chip {
        display: inline-block;
        margin: 0.25rem 0.35rem 0 0;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(92, 136, 198, 0.16);
        border: 1px solid rgba(92, 136, 198, 0.18);
        color: #cfe0ff;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .alert-banner {
        margin-top: 1rem;
        padding: 1.25rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(255, 31, 31, 0.94), rgba(145, 0, 0, 0.94));
        border: 1px solid rgba(255, 140, 140, 0.45);
        color: #fff;
        text-align: center;
        font-size: 1.55rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        animation: pulseAlert 1.05s infinite alternate;
        box-shadow: 0 0 40px rgba(255, 61, 61, 0.36);
    }
    @keyframes pulseAlert {
        from { transform: scale(1); }
        to { transform: scale(1.02); }
    }
    .table-title {
        color: #ebf4ff;
        font-size: 0.95rem;
        font-weight: 700;
    }
    .note {
        color: #8fa7c7;
        font-size: 0.92rem;
    }
    div[data-testid="stButton"] > button {
        width: 100%;
        border-radius: 15px;
        min-height: 3rem;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(145deg, #0c1730, #122042);
        color: #eef6ff;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: rgba(104, 173, 255, 0.42);
        box-shadow: 0 0 24px rgba(61, 137, 255, 0.22);
    }
    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }
</style>
"""


def get_status_style(status):
    if status == "SCREAM DETECTED":
        return "status-alert", "Critical event confirmed by the AI monitoring pipeline."
    if status == "POSSIBLE SCREAM":
        return "status-warning", "Elevated scream probability detected. Monitoring persistence."
    return "status-normal", "Ambient environment is stable and within the normal range."


def downsample_waveform(waveform, target_points=400):
    if len(waveform) <= target_points:
        return waveform
    indices = np.linspace(0, len(waveform) - 1, target_points).astype(int)
    return waveform[indices]


def render_hero(snapshot):
    badge = "LIVE MONITORING ACTIVE" if snapshot["running"] else "SYSTEM STANDBY"
    st.markdown(
        f"""
        <div class="glass-card hero-card">
            <div class="hero-title">AI Scream Detection System</div>
            <div class="hero-subtitle">Real-Time Audio Monitoring</div>
            <div class="hero-badge">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(snapshot):
    status_class, caption = get_status_style(snapshot["state"])
    st.markdown(
        f"""
        <div class="glass-card card">
            <div class="section-label">Status Panel</div>
            <div class="status-shell {status_class}">
                <div class="status-text">{snapshot["state"]}</div>
                <div class="status-caption">{caption}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge(snapshot):
    confidence = float(snapshot["confidence"])
    degrees = int(max(0.0, min(1.0, confidence)) * 360)
    reason_text = snapshot["reasons"] or ["monitoring"]
    chips = "".join(f'<span class="signal-chip">{item}</span>' for item in reason_text)
    st.markdown(
        f"""
        <div class="glass-card card">
            <div class="section-label">Confidence Gauge</div>
            <div class="gauge-wrap">
                <div class="gauge" style="background: conic-gradient(#16ff9d 0deg, #ffd84f 180deg, #ff4c4c {degrees}deg, rgba(24,34,53,0.95) {degrees}deg 360deg);">
                    <div class="gauge-center">
                        <div class="gauge-value">{confidence:.2f}</div>
                        <div class="gauge-label">Scream Score</div>
                    </div>
                </div>
            </div>
            <div class="note">Predicted class: <strong>{snapshot["predicted_class"]}</strong> ({snapshot["predicted_confidence"]:.2f})</div>
            <div style="margin-top:0.55rem;">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_controls(backend, snapshot):
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Control Panel</div>', unsafe_allow_html=True)
    start_col, stop_col = st.columns(2)

    with start_col:
        if st.button("Start Detection", type="primary", use_container_width=True):
            try:
                backend.start()
                st.session_state["dashboard_error"] = ""
            except Exception as exc:
                st.session_state["dashboard_error"] = str(exc)
            st.rerun()

    with stop_col:
        if st.button("Stop Detection", use_container_width=True):
            backend.stop()
            st.rerun()

    if st.button("Play Last Alert", use_container_width=True):
        st.session_state["play_last_alert"] = True

    mode_text = "Monitoring microphone input continuously." if snapshot["running"] else "System idle. Detection stream stopped."
    st.markdown(f'<div class="note" style="margin-top:0.8rem;">{mode_text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_tiles(snapshot):
    event = snapshot["event"]
    st.markdown(
        """
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-name">Scream Raw</div>
                <div class="metric-num">{:.2f}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-name">Adjusted</div>
                <div class="metric-num">{:.2f}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-name">RMS</div>
                <div class="metric-num">{:.3f}</div>
            </div>
            <div class="metric-tile">
                <div class="metric-name">Active Ratio</div>
                <div class="metric-num">{:.2f}</div>
            </div>
        </div>
        """.format(
            snapshot["raw_confidence"],
            snapshot["adjusted_confidence"],
            event["rms"],
            event["active_ratio"],
        ),
        unsafe_allow_html=True,
    )


def render_live_confidence_chart(snapshot):
    chart_df = pd.DataFrame(
        {
            "Frame": list(range(1, len(snapshot["history"]) + 1)),
            "Confidence": snapshot["history"],
        }
    ).set_index("Frame")
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Inference Trend</div>', unsafe_allow_html=True)
    st.line_chart(chart_df, height=260, color="#00f7a7")
    st.markdown("</div>", unsafe_allow_html=True)


def render_waveform(snapshot):
    waveform = downsample_waveform(snapshot["waveform"])
    waveform_df = pd.DataFrame({"Amplitude": waveform})
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Real-Time Audio Waveform</div>', unsafe_allow_html=True)
    st.line_chart(waveform_df, height=260, color="#52b7ff")
    st.markdown("</div>", unsafe_allow_html=True)


def render_spectrogram(snapshot):
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Mel Spectrogram</div>', unsafe_allow_html=True)
    spectrogram = snapshot["spectrogram_image"]
    if spectrogram is not None:
        st.image(spectrogram, use_container_width=True, clamp=True)
    else:
        st.info("Spectrogram will appear once audio streaming starts.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_class_probabilities(snapshot):
    probs = snapshot["class_probabilities"]
    if probs:
        prob_df = pd.DataFrame(
            {"Class": list(probs.keys()), "Probability": list(probs.values())}
        ).set_index("Class")
    else:
        prob_df = pd.DataFrame({"Class": [], "Probability": []}).set_index("Class")
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
    if not prob_df.empty:
        st.bar_chart(prob_df, height=260, color="#ffcf4c")
    else:
        st.info("Class probabilities will update when inference starts.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_alert_history(snapshot):
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Alert History</div>', unsafe_allow_html=True)
    history = snapshot["alert_history"]
    if history:
        df = pd.DataFrame(history)
        df = df.rename(columns={"time": "Time", "confidence": "Confidence", "status": "Status", "class": "Class"})
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No alert events recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_audio_playback(snapshot):
    st.markdown('<div class="glass-card card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Alert Playback</div>', unsafe_allow_html=True)
    if snapshot["last_alert_audio_bytes"]:
        label = snapshot["last_alert_filename"] or "last_alert.wav"
        st.markdown(f'<div class="note">Last captured alert clip: {label}</div>', unsafe_allow_html=True)
        if st.session_state.get("play_last_alert"):
            st.audio(snapshot["last_alert_audio_bytes"], format="audio/wav")
    else:
        st.info("Captured alert audio will appear here after the first confirmed scream.")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("dashboard_error", "")
    st.session_state.setdefault("play_last_alert", False)

    backend = get_backend()
    snapshot = backend.get_snapshot()

    if st.session_state["dashboard_error"]:
        st.error(st.session_state["dashboard_error"])
    if snapshot["last_error"]:
        st.warning(f"Audio stream warning: {snapshot['last_error']}")

    render_hero(snapshot)

    top_left, top_mid, top_right = st.columns([1.25, 1.0, 0.9])
    with top_left:
        render_status(snapshot)
    with top_mid:
        render_gauge(snapshot)
    with top_right:
        render_controls(backend, snapshot)

    render_metric_tiles(snapshot)

    row1_col1, row1_col2 = st.columns([1.15, 0.85])
    with row1_col1:
        render_live_confidence_chart(snapshot)
    with row1_col2:
        render_class_probabilities(snapshot)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        render_waveform(snapshot)
    with row2_col2:
        render_spectrogram(snapshot)

    row3_col1, row3_col2 = st.columns([1.2, 0.8])
    with row3_col1:
        render_alert_history(snapshot)
    with row3_col2:
        render_audio_playback(snapshot)

    if snapshot["state"] == "SCREAM DETECTED":
        st.markdown('<div class="alert-banner">🚨 EMERGENCY DETECTED</div>', unsafe_allow_html=True)

    if snapshot["running"]:
        time.sleep(0.35)
        st.rerun()


if __name__ == "__main__":
    main()
