import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import fuse_session
from audio import analyze_audio
from vision import analyze_video_file, analyze_webcam


st.set_page_config(page_title="Attendance & Engagement Agent", page_icon="📚", layout="wide")
st.title("📚 Multimodal Attendance & Engagement Agent")
st.caption("Local, anonymous session indicators for human review. This tool does not identify or grade students.")

with st.sidebar:
    st.header("Session settings")
    source = st.radio("Video source", ["Upload a video", "Local webcam"])
    threshold = st.slider("Seconds visible to count as present", 1, 60, 5)
    sample_fps = st.select_slider("Video sampling rate", options=[1, 2, 3, 5], value=2)
    webcam_seconds = st.slider("Webcam capture length (seconds)", 10, 120, 30) if source == "Local webcam" else None

video = st.file_uploader("Classroom video (MP4, MOV, AVI)", type=["mp4", "mov", "avi", "mkv"]) if source == "Upload a video" else None
audio_file = st.file_uploader("Optional audio recording (WAV or MP3)", type=["wav", "mp3", "m4a", "ogg"])

if st.button("Analyze session", type="primary", disabled=(source == "Upload a video" and video is None)):
    try:
        with st.spinner("Analyzing locally…"):
            audio_result = None
            if source == "Local webcam":
                vision_result = analyze_webcam(webcam_seconds, threshold, sample_fps)
            else:
                with tempfile.TemporaryDirectory() as work_dir:
                    video_path = Path(work_dir) / f"session{Path(video.name).suffix.lower()}"
                    video_path.write_bytes(video.getvalue())
                    vision_result = analyze_video_file(str(video_path), threshold, sample_fps)
                    if audio_file is not None:
                        audio_path = Path(work_dir) / f"session-audio{Path(audio_file.name).suffix.lower()}"
                        audio_path.write_bytes(audio_file.getvalue())
                        audio_result = analyze_audio(str(audio_path))

            if source == "Local webcam" and audio_file is not None:
                with tempfile.TemporaryDirectory() as work_dir:
                    audio_path = Path(work_dir) / f"session-audio{Path(audio_file.name).suffix.lower()}"
                    audio_path.write_bytes(audio_file.getvalue())
                    audio_result = analyze_audio(str(audio_path))

        agent_result = fuse_session(vision_result, audio_result)
        st.session_state["analysis"] = (vision_result, audio_result, agent_result)
    except Exception as exc:
        st.error(str(exc))

if "analysis" in st.session_state:
    vision_result, audio_result, agent_result = st.session_state["analysis"]
    left, middle, right = st.columns(3)
    left.metric("Present (anonymous tracks)", vision_result["present_count"])
    middle.metric("Tracks observed", vision_result["track_count"])
    right.metric("Session activity indicator", f"{agent_result['score']}/100", agent_result["label"])

    st.subheader("Attendance estimate")
    if vision_result["people"]:
        st.dataframe(pd.DataFrame(vision_result["people"]), use_container_width=True, hide_index=True)
    else:
        st.info("No faces detected. Try a well-lit video with faces visible to the camera.")

    st.subheader("Engagement timeline")
    st.caption("Rule-based session indicator from visible-face presence and optional overall audio activity. It is not an individual attention or learning measurement.")
    st.line_chart(pd.DataFrame(agent_result["timeline"]).set_index("time_seconds")["engagement_score"])
    st.subheader("Instructor prompts")
    for suggestion in agent_result["suggestions"]:
        st.write(f"• {suggestion}")

    if audio_result is not None:
        st.subheader("Audio context")
        st.write(f"Mean signal level: {audio_result['mean_rms']} · Active sound windows: {audio_result['active_ratio']:.0%}")
        st.caption(audio_result["note"])
        if audio_result["windows"]:
            st.line_chart(pd.DataFrame(audio_result["windows"]).set_index("time_seconds")["rms"])

st.divider()
st.caption("Uploads and camera frames are processed in memory by this local app. Results are approximate decision-support cues; review them in context.")
