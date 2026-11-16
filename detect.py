from ultralytics import YOLO
import cv2
from pathlib import Path

video_dir = Path.home() / "SafeZone-AI/videos"
videos = sorted(video_dir.glob("*.mp4"))

if not videos:
    raise SystemExit(f"No MP4 videos found in {video_dir}")

video = videos[0]
print(f"Processing video: {video.name}")

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise SystemExit("Cannot open the selected video")

output_dir = Path.home() / "SafeZone-AI/output"
output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / "detected_video.mp4"

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    str(output),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, classes=[0], verbose=False)
        writer.write(result[0].plot())
finally:
    cap.release()
    writer.release()

print(f"Detection complete: {output}")
