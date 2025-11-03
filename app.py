import io
import time
import uuid
import cv2
import sqlite3
import numpy as np
from datetime import datetime
from PIL import Image
import streamlit as st
from model import detect_mood, init_model  # renamed imports

# Streamlit configuration
st.set_page_config(page_title="MoodVision App", page_icon="🙂", layout="centered")

st.title("🙂 MoodVision — Image Mood Analyzer")
st.write("Upload or capture a photo and let AI reveal the mood!")

# Tabs
upload_tab, live_tab, log_tab = st.tabs(["📷 Upload Image", "🎥 Live Camera", "🕓 History"])

# Model and database paths
MODEL_DIR = "abhilash88/face-emotion-detection"
DB_FILE = "mood_records.db"

# Emoji mapping
mood_emoji = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Database setup
connection = sqlite3.connect(DB_FILE)
db_cursor = connection.cursor()

# Load model and processor
preproc, net = init_model(MODEL_DIR)


def setup_database():
    db_cursor.execute(
        """CREATE TABLE IF NOT EXISTS mood_logs(
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            mood_label TEXT,
            confidence_score REAL,
            image_blob BLOB,
            created_at TEXT
        )"""
    )


def insert_record(fname, mood, conf, img_data, created=datetime.now().strftime("%Y-%m-%d %H:%M:%S")):
    db_cursor.execute(
        "INSERT INTO mood_logs(filename, mood_label, confidence_score, image_blob, created_at) VALUES (?, ?, ?, ?, ?)",
        (fname, mood, conf, img_data, created),
    )
    connection.commit()


def open_connection():
    return sqlite3.connect(DB_FILE, check_same_thread=False)


def remove_record(record_id):
    with open_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM mood_logs WHERE record_id=?", (record_id,))
        conn.commit()


# Initialize DB
setup_database()

# --------------------------
# 📷 UPLOAD TAB
# --------------------------
with upload_tab:
    file_upload = st.file_uploader("Select an image", type=["jpg", "jpeg", "png", "webp"])

    if file_upload is not None:
        file_bytes = file_upload.getvalue()

        col_img, col_result = st.columns(2)

        with col_img:
            preview_img = Image.open(file_upload)
            st.image(preview_img, caption="Uploaded Photo", width='stretch')

        detect_btn = st.button("Analyze Mood", type="primary")

        if detect_btn:
            with col_result:
                outcome = detect_mood(file_upload, net, preproc)

                if outcome:
                    mood, prob = outcome
                    st.subheader("Analysis Results")
                    st.success(f"**Mood:** {mood.capitalize()} {mood_emoji[mood.lower()]}")
                    st.write(f"Confidence: {(prob * 100):.2f}%")

                    insert_record(file_upload.name, mood, prob, file_bytes)
                else:
                    st.warning("No face detected. Please try a clearer image.")
    else:
        st.info("Upload an image to begin analysis.")


# --------------------------
# 🎥 LIVE CAMERA TAB
# --------------------------
with live_tab:
    if "cam" not in st.session_state:
        st.session_state.cam = None
    if "frame" not in st.session_state:
        st.session_state.frame = None
    if "capture_triggered" not in st.session_state:
        st.session_state.capture_triggered = False

    start_live = st.checkbox("Enable Camera", value=False)
    info_box = st.info("Turn on your camera to start.")
    live_display = st.empty()
    capture_btn_placeholder = st.empty()
    output_placeholder = st.empty()

    if start_live:
        info_box.empty()

        if st.session_state.cam is None:
            st.session_state.cam = cv2.VideoCapture(0)

        webcam = st.session_state.cam
        capture_btn = capture_btn_placeholder.button("🔍 Analyze Mood", key="analyze_button")

        while True:
            success, frame = webcam.read()
            if not success:
                st.warning("Unable to access the webcam.")
                break

            frame = cv2.flip(frame, 1)
            st.session_state.frame = frame
            rgb_view = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_view = Image.fromarray(rgb_view)
            live_display.image(pil_view, channels="RGB", use_container_width=True)

            if capture_btn:
                st.session_state.capture_triggered = True
                break

            time.sleep(0.03)

            if not st.session_state.get("run_checkbox_value", True):
                break

        if st.session_state.capture_triggered:
            snap = st.session_state.frame
            rgb_snap = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
            pil_snap = Image.fromarray(rgb_snap)
            _, buff = cv2.imencode('.jpg', frame)
            img_blob = buff.tobytes()
            fname = f"live_mood_capture{uuid.uuid4()}"

            mood, conf = detect_mood(snap, net, preproc)

            st.success("Captured successfully! Restart to take another.")
            output_placeholder.markdown(
                f"### Mood: **{mood}** {mood_emoji[mood.lower()]} ({conf*100:.1f}%)"
            )
            insert_record(fname, mood, conf, img_blob)

        webcam.release()
        st.session_state.cam = None

    else:
        if st.session_state.cam:
            st.session_state.cam.release()
            st.session_state.cam = None
        live_display.empty()
        capture_btn_placeholder.empty()


# --------------------------
# 🕓 HISTORY TAB
# --------------------------
with log_tab:
    st.subheader("Mood Detection Log")

    db_cursor.execute(
        "SELECT record_id, filename, mood_label, confidence_score, image_blob, created_at FROM mood_logs ORDER BY record_id DESC"
    )
    entries = db_cursor.fetchall()

    if entries:
        st.markdown(f"**Total Entries:** {len(entries)}")

        for rec_id, fname, mood, conf, data, created in entries:
            left, right = st.columns([1, 3])

            with left:
                img = Image.open(io.BytesIO(data))
                st.image(img, width=100)

            with right:
                st.write(f"**Filename:** {fname}")
                st.write(f"**Mood:** {mood.capitalize()} {mood_emoji[mood.lower()]}")
                st.write(f"**Confidence:** {(conf * 100):.2f}%")
                st.write(f"**Timestamp:** {created}")
                st.button("Delete", key=f"remove_{rec_id}", on_click=remove_record, args=(rec_id,))
                st.markdown("---")
    else:
        st.info("No records found. Upload or capture an image to view history.")
