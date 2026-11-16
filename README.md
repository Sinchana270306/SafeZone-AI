<div align="center">

# 🛡️ SafeZone AI

### AI-Powered Restricted-Area Intrusion Detection

**Detect people. Track movement. Identify entry into a defined restricted zone.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/Detection-YOLO11-111827)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

</div>

---

## 🎬 Project Demo

This demonstration shows SafeZone AI processing construction-site footage, drawing the configured restricted-zone boundary, identifying people, and marking detected intrusions.

**To display the playable video directly in this GitHub README:** open `README.md` on GitHub, click the pencil (**Edit**) button, and drag `SafeZone-AI-demo.mp4` from your computer into the README editor below this paragraph. GitHub will upload the video and insert a `https://github.com/user-attachments/assets/...` URL. Leave that generated URL on its own line, then commit the README.

<!-- GITHUB_VIDEO_EMBED: Drag SafeZone-AI-demo.mp4 into the GitHub README editor here and replace this comment with the generated user-attachments URL. -->

[Download or view the demo video from the repository](demo/SafeZone-AI-demo.mp4) (if you keep the video in the `demo/` folder).

---

## 📌 Overview

SafeZone AI is a computer-vision prototype for monitoring restricted areas in surveillance footage. It processes an uploaded video, detects people with **YOLO11**, tracks them across frames with **ByteTrack**, and checks whether each person's bottom-centre point enters a configured polygon.

When a tracked person is detected inside the polygon, the application records an intrusion event. The dashboard can show the annotated video and the generated alert report.

> **Current implementation:** the restricted zone is defined by a fixed, normalized polygon in the Python code. Automatic discovery of restricted areas from visual cues is a planned enhancement, not a current feature.

## ✨ Features

- **Person detection** using the YOLO11 model.
- **Multi-object tracking** using ByteTrack tracking IDs.
- **Polygon-based zone checking** using OpenCV's point-in-polygon test.
- **Intrusion event logging** with zone ID, person ID, video timestamp, and recording time.
- **Video annotation** with bounding boxes, tracking IDs, zone boundary, and intrusion labels.
- **Streamlit dashboard** for uploading footage and reviewing results.
- **JSON report export** and annotated-video download.

## 🧭 How It Works

1. Upload an MP4, AVI, or MOV surveillance video in the Streamlit dashboard.
2. OpenCV reads the video frame by frame.
3. YOLO11 detects people in each frame.
4. ByteTrack associates detections across frames and provides tracking IDs.
5. The application calculates each person's bottom-centre point (the approximate foot point).
6. OpenCV checks whether that point is inside the configured restricted-zone polygon.
7. New entries are added to `alerts.json`; annotated frames are written to the processed video.

## 🏗️ System Workflow

```text
Surveillance Video
       │
       ▼
Streamlit Upload
       │
       ▼
OpenCV Frame Processing
       │
       ▼
YOLO11 Person Detection
       │
       ▼
ByteTrack Object Tracking
       │
       ▼
Person Foot-Point Calculation
       │
       ▼
Restricted-Zone Polygon Test
       │
       ├── Outside zone ──► Normal person annotation
       │
       └── Inside zone ───► Intrusion annotation + alert record
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
               Annotated Video             alerts.json
```

## 🧰 Technology Stack

| Technology | Role |
|---|---|
| Python | Application logic |
| Streamlit | Web dashboard |
| Ultralytics YOLO11 | Person detection |
| ByteTrack | Object tracking |
| OpenCV | Video processing, drawing, and polygon testing |
| NumPy | Polygon coordinate handling |
| JSON | Alert storage |

## 📁 Project Structure

```text
SafeZone-AI/
├── app.py                  # Streamlit dashboard and video workflow
├── restricted_zone.py      # Standalone restricted-zone detection script
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── yolo11n.pt              # YOLO model weights (if stored locally)
├── demo/
│   └── SafeZone-AI-demo.mp4
└── output/                 # Generated at runtime; normally not committed
    ├── alerts.json
    └── restricted_zone.mp4
```

The exact files in your local checkout may differ. Keep generated outputs and the Python virtual environment out of version control.

## ⚙️ Installation

### Prerequisites

- Python 3.10 or newer
- `pip`
- Git
- Internet access for installing dependencies and, if needed, downloading model weights

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SafeZone-AI.git
cd SafeZone-AI
```

Replace `YOUR_USERNAME` with the GitHub account that owns the repository.

### 2. Create and activate a virtual environment

**Ubuntu / WSL / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell**

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Check the YOLO model

The application is configured to load `yolo11n.pt` from the project directory. Place the model file there. Ultralytics may also download the model automatically when it is not present and network access is available.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open the local address printed by Streamlit, normally:

```text
http://localhost:8501
```

## 🖥️ Using the Dashboard

1. Select **Video Monitoring** from the sidebar.
2. Upload a supported surveillance video (`.mp4`, `.avi`, or `.mov`).
3. Select **Start AI Detection**.
4. Wait for processing to finish.
5. Review the annotated video and intrusion-event table.
6. Download the processed video or JSON report.

The detection confidence can be adjusted from **Settings**.

## 📄 Example Alert Record

An alert is stored in JSON with fields similar to:

```json
{
  "event": "restricted_area_intrusion",
  "zone_id": "zone_1",
  "person_id": 3,
  "timestamp_seconds": 12.45,
  "recorded_at": "2026-11-15T10:30:00"
}
```

The values above are illustrative; actual values depend on the input video and detection run.

## 🗺️ Restricted-Zone Configuration

The current dashboard version uses a normalized polygon called `DEFAULT_ZONE` in `app.py`. Each coordinate is a fraction of the frame dimensions (from `0` to `1`), so the polygon is scaled to the uploaded video's width and height.

Adjust the polygon to match the camera view. The current method checks the person's bottom-centre point against the polygon; it does not infer a real-world boundary or automatically recognize safety tape, barriers, or warning signs.

## 🚧 Current Scope and Limitations

- The current zone is a predefined polygon; automatic restricted-area recognition is not implemented yet.
- The application processes uploaded videos; live camera-stream monitoring is not included in the current workflow.
- Detection quality depends on camera angle, lighting, video resolution, occlusion, and model performance.
- ByteTrack IDs are tracking identifiers and should not be treated as verified real-world identities.
- This prototype is an aid for reviewing footage, not a certified safety system. Human review is recommended before taking action.

## 🔭 Planned Enhancements

- Interactive polygon drawing in the dashboard.
- Automatic boundary proposals from visible safety tape, barriers, or other visual cues.
- Multiple independently configurable restricted zones.
- Live camera-stream support.
- Configurable alert notifications.
- Detection history, filtering, and analytics.
- Evaluation on varied site footage to measure detection accuracy and false alarms.

## 🤝 Contributing

Suggestions and improvements are welcome. For a substantial change, open an issue first to discuss the proposed feature, then submit a pull request.

## 📜 License

No license has been specified yet. Add a `LICENSE` file before presenting the project as open source or allowing reuse by others.

---

<div align="center">

**Built with Python, Computer Vision, and AI.**

If you find this project useful, consider ⭐ starring the repository.

</div>
