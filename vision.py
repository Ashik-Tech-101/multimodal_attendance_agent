"""Anonymous face detection and approximate presence tracking."""

from collections import defaultdict

import cv2


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def analyze_frames(frames, fps, min_presence_seconds=5.0):
    """Process BGR frames; return counts, anonymous tracks, and timeline."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV could not load its built-in face detector.")

    tracks = {}
    seen_frames = defaultdict(int)
    next_id = 1
    timeline = []
    last_second = -1

    for frame_index, frame in enumerate(frames):
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        boxes = [tuple(map(int, box)) for box in boxes]
        timestamp = frame_index / max(fps, 1.0)
        used = set()

        for box in boxes:
            candidates = [(_iou(box, old_box), track_id) for track_id, old_box in tracks.items() if track_id not in used]
            score, track_id = max(candidates, default=(0.0, None))
            if score < 0.15:
                track_id = next_id
                next_id += 1
            tracks[track_id] = box
            used.add(track_id)
            seen_frames[track_id] += 1

        current_second = int(timestamp)
        if current_second != last_second:
            timeline.append({"time_seconds": current_second, "visible_faces": len(boxes)})
            last_second = current_second

    people = []
    for track_id, count in seen_frames.items():
        seconds = count / max(fps, 1.0)
        people.append({
            "participant": f"Track {track_id}",
            "visible_seconds": round(seconds, 1),
            "attendance": "Present" if seconds >= min_presence_seconds else "Below threshold",
        })
    duration = (timeline[-1]["time_seconds"] + (1 / max(fps, 1.0))) if timeline else 0
    return {
        "people": people,
        "present_count": sum(p["attendance"] == "Present" for p in people),
        "track_count": len(people),
        "timeline": timeline,
        "duration_seconds": round(duration, 1),
    }


def analyze_video_file(path, min_presence_seconds=5.0, sample_fps=2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open this video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sampled_fps = min(float(sample_fps), fps)
    frames = []
    index = 0
    stride = max(1, round(fps / sampled_fps))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
        index += 1
    cap.release()
    return analyze_frames(frames, sampled_fps, min_presence_seconds)


def analyze_webcam(seconds=60, min_presence_seconds=5.0, sample_fps=2):
    import time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("Could not open the local webcam. Check its permission and connection.")
    frames = []
    start = time.monotonic()
    last_sample = 0.0
    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic() - start
        if now - last_sample >= 1 / max(sample_fps, 0.1):
            frames.append(frame)
            last_sample = now
    cap.release()
    return analyze_frames(frames, float(sample_fps), min_presence_seconds)
