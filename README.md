# Multimodal Attendance & Engagement Agent

A small local Streamlit demo that analyzes a classroom video (or local webcam) and an optional audio file. It uses OpenCV's built-in face detector, anonymous short-lived track IDs, simple audio features, and transparent rules to create attendance and session summaries.

It does **not** identify people, infer emotion, record or store video, transcribe speech, or grade students. Engagement values are rough session indicators for human review, not assessments of an individual.

## Requirements

- Python 3.9 or newer
- A webcam for live capture (optional)
- An audio file for audio indicators (optional; WAV is easiest)

## Setup (Windows PowerShell)

```powershell
cd multimodal_attendance_agent
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. If you only want video analysis, the app can still run when librosa is unavailable; audio analysis will be disabled.

## Use

1. Choose **Upload a video** and select an MP4/AVI/MOV file, or choose **Local webcam** and start capture.
2. Set the minimum visible time for an anonymous track to count as present.
3. Optionally upload a WAV/MP3 audio recording. The app measures RMS level and estimates active-sound ratio; it cannot reliably distinguish speech from other sounds.
4. Select **Analyze** to see anonymous attendance counts, engagement over time, and rule-based teaching suggestions.

Video and audio are analyzed locally. Uploaded files are briefly staged in a temporary folder so the media decoders can open them; the folder is removed immediately after processing, including on errors. Camera frames are not saved. Camera access is local to the machine running Streamlit.

## Project layout

```text
app.py                  Streamlit interface
vision.py               Face detection, anonymous tracking, presence time
audio.py                Optional audio level/activity features
agent.py                Signal fusion, engagement score, suggestions
requirements.txt        Python dependencies
```

## Limitations

- Haar face detection can miss faces with poor lighting, occlusion, or unusual camera angles.
- Track IDs are temporary and may change after occlusion; they are not identity labels.
- Presence is measured in sampled frames, so results are approximate.
- Audio activity is an environmental cue, not speech recognition or a measure of any person's participation.
- Scores and suggestions are transparent heuristics. Review them in context and do not use them for grading or consequential decisions.
