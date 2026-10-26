"""Transparent rules for combining session level vision and audio cues."""


def fuse_session(vision, audio=None):
    timeline = vision.get("timeline", [])
    if not timeline:
        return {"score": 0, "label": "No data", "timeline": [], "suggestions": ["Try a longer video with a clear view of the room."]}

    audio_activity = (audio or {}).get("active_ratio")
    scored = []
    for item in timeline:
        presence = min(item.get("visible_faces", 0) / max(vision.get("track_count", 1), 1), 1.0)
        score = 50 + 35 * presence
        if audio_activity is not None:
            score += 15 * (audio_activity - 0.5)
        score = max(0, min(100, round(score)))
        scored.append({**item, "engagement_score": score})

    average = round(sum(row["engagement_score"] for row in scored) / len(scored))
    label = "Higher activity" if average >= 70 else "Moderate activity" if average >= 45 else "Lower activity"
    suggestions = []
    if average < 55:
        suggestions.append("Consider a brief check-in or interactive question, then review whether participation changes.")
    else:
        suggestions.append("The session shows sustained visual presence; continue checking understanding with short questions.")
    if audio_activity is None:
        suggestions.append("Add an audio file if you want a rough room sound activity cue.")
    elif audio_activity < 0.2:
        suggestions.append("The recording has little active sound; confirm that the microphone captured the room before drawing conclusions.")
    else:
        suggestions.append("Audio shows active sound in some windows; this can include speech or background noise.")
    return {"score": average, "label": label, "timeline": scored, "suggestions": suggestions}
