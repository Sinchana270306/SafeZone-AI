
import streamlit as st
import cv2
import json
import os
import tempfile
import numpy as np
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(
    page_title="SafeZone AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafeZone AI")
st.caption("AI-Powered Restricted Area Intrusion Detection")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Video Monitoring",
        "Restricted Zone Setup",
        "AI Detection",
        "Alerts",
        "Reports",
        "Settings"
    ]
)

os.makedirs("output", exist_ok=True)

ALERTS_FILE = "output/alerts.json"
OUTPUT_FILE = "output/restricted_zone.mp4"
MODEL_FILE = "yolo11n.pt"

# Default restricted-zone polygon.
# Coordinates are normalized from 0 to 1.
DEFAULT_ZONE = [
    (0.05, 0.70),
    (0.98, 0.70),
    (0.98, 0.93),
    (0.05, 0.93)
]

def run_detection(video_path, confidence):
    model = YOLO(MODEL_FILE)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    zone = [
        (int(x * width), int(y * height))
        for x, y in DEFAULT_ZONE
    ]

    polygon = np.array(zone, dtype=np.int32)

    writer = cv2.VideoWriter(
        OUTPUT_FILE,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Unable to create output video.")

    alerts = []
    previously_inside = set()
    frame_number = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    status = st.empty()

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            cv2.polylines(
                frame, [polygon], True, (0, 0, 255), 2
            )

            cv2.putText(
                frame, "RESTRICTED AREA", zone[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 2
            )

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                conf=confidence,
                tracker="bytetrack.yaml",
                verbose=False
            )

            inside_now = set()

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    coords = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    x1, y1, x2, y2 = coords

                    person_id = (
                        int(box.id[0])
                        if box.id is not None else -1
                    )

                    foot = (
                        (x1 + x2) // 2,
                        y2
                    )

                    inside = cv2.pointPolygonTest(
                        polygon,
                        (float(foot[0]), float(foot[1])),
                        False
                    ) >= 0

                    color = (
                        (0, 0, 255)
                        if inside else (0, 200, 0)
                    )

                    label = (
                        "INTRUSION"
                        if inside else "Person"
                    )

                    if person_id >= 0:
                        label += f" ID:{person_id}"

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2),
                        color, 2
                    )

                    cv2.putText(
                        frame, label,
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2
                    )

                    if inside:
                        inside_now.add(person_id)

                        if person_id not in previously_inside:
                            alerts.append({
                                "event": "restricted_area_intrusion",
                                "zone_id": "zone_1",
                                "person_id": person_id,
                                "timestamp_seconds": round(
                                    frame_number / fps, 2
                                ),
                                "recorded_at": datetime.now().isoformat(
                                    timespec="seconds"
                                )
                            })

            previously_inside = inside_now
            writer.write(frame)

            if total_frames > 0:
                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            status.write(
                f"Processing frame {frame_number} "
                f"of {total_frames or 'unknown'}"
            )

    finally:
        cap.release()
        writer.release()

    with open(ALERTS_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2)

    progress.progress(1.0)
    status.success("Video processing completed!")

    return alerts


if page == "Dashboard":
    st.header("Dashboard")

    col1, col2, col3 = st.columns(3)

    alert_count = 0
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, encoding="utf-8") as f:
            alert_count = len(json.load(f))

    col1.metric("Detection System", "Ready")
    col2.metric("Restricted Zones", "1")
    col3.metric("Intrusion Events", alert_count)

    st.info("Upload a video to begin AI monitoring.")

elif page == "Video Monitoring":
    st.header("Video Monitoring")

    video = st.file_uploader(
        "Upload surveillance video",
        type=["mp4", "avi", "mov"]
    )

    if video:
        st.video(video)
        st.success(f"Uploaded: {video.name}")

        if st.button("Start AI Detection", type="primary"):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp:
                    temp.write(video.getbuffer())
                    temp_path = temp.name

                with st.spinner("Loading YOLO and processing video..."):
                    alerts = run_detection(
                        temp_path,
                        st.session_state.get("confidence", 0.35)
                    )

                st.success("Detection completed successfully!")

                st.subheader("Detection Results")

                col1, col2 = st.columns(2)
                col1.metric("Intrusion Events", len(alerts))
                col2.metric("Processed Video", "Ready")

                if os.path.exists(OUTPUT_FILE):
                    st.video(OUTPUT_FILE)

                    with open(OUTPUT_FILE, "rb") as f:
                        st.download_button(
                            "Download Annotated Video",
                            f,
                            file_name="safezone_output.mp4",
                            mime="video/mp4"
                        )

                st.subheader("Intrusion Alerts")
                if alerts:
                    st.dataframe(alerts, use_container_width=True)
                else:
                    st.success("No intrusion events detected.")

            except Exception as e:
                st.error(f"Detection failed: {e}")

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

elif page == "Restricted Zone Setup":
    st.header("Restricted Zone Setup")
    st.write("Current provisional restricted-zone coordinates:")
    st.write(DEFAULT_ZONE)
    st.warning(
        "This version uses a fixed polygon. "
        "Interactive zone drawing will be added next."
    )

elif page == "AI Detection":
    st.header("AI Detection")
    st.write("Model: YOLO11n")
    st.write("Tracking: ByteTrack")
    st.write("Detection class: Person")
    st.info("Open Video Monitoring to upload a video and start detection.")

elif page == "Alerts":
    st.header("Intrusion Alerts")

    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, encoding="utf-8") as f:
            alerts = json.load(f)

        if alerts:
            st.dataframe(alerts, use_container_width=True)
        else:
            st.info("No intrusion alerts recorded.")
    else:
        st.info("No alerts available yet.")

elif page == "Reports":
    st.header("Reports")

    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, encoding="utf-8") as f:
            data = f.read()

        st.download_button(
            "Download JSON Report",
            data,
            file_name="safezone_alerts.json",
            mime="application/json"
        )
    else:
        st.info("Run detection to generate a report.")

elif page == "Settings":
    st.header("Detection Settings")

    st.session_state["confidence"] = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=st.session_state.get("confidence", 0.35),
        step=0.05
    )
