VIDEO = "videos/4281405-sd_960_540_25fps.mp4"
import cv2
import json
import os
from datetime import datetime
from ultralytics import YOLO

VIDEO = "videos/rest.mp4"
MODEL = "yolo11n.pt"
OUTPUT = "output/restricted_zone.mp4"
ALERTS = "output/alerts.json"

os.makedirs("output", exist_ok=True)

# Provisional zone for the 426x240 preview.
# Coordinates are scaled to each actual video frame.
ZONE_NORMALIZED = [
    (0.05, 0.70),
    (0.98, 0.70),
    (0.98, 0.93),
    (0.05, 0.93),
]

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

zone = [(int(x * width), int(y * height))
        for x, y in ZONE_NORMALIZED]

writer = cv2.VideoWriter(
    OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (width, height)
)

alerts = []
previously_inside = set()
frame_number = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_number += 1
    polygon = __import__("numpy").array(zone, dtype="int32")
    cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
    cv2.putText(frame, "RESTRICTED AREA", zone[0],
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255), 2)

    results = model.track(
        frame, persist=True, classes=[0],
        conf=0.35, tracker="bytetrack.yaml",
        verbose=False
    )

    inside_now = set()

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            person_id = int(box.id[0]) if box.id is not None else -1

            # Approximate the person's ground-contact point.
            foot = ((x1 + x2) // 2, y2)
            inside = cv2.pointPolygonTest(
                polygon, (float(foot[0]), float(foot[1])), False
            ) >= 0

            color = (0, 0, 255) if inside else (0, 200, 0)
            label = "INTRUSION" if inside else "Person"
            if person_id >= 0:
                label += f" ID:{person_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if inside:
                inside_now.add(person_id)
                if person_id not in previously_inside:
                    alerts.append({
                        "event": "restricted_area_intrusion",
                        "zone_id": "zone_1",
                        "person_id": person_id,
                        "video": VIDEO,
                        "timestamp_seconds": round(frame_number / fps, 2),
                        "recorded_at": datetime.now().isoformat(
                            timespec="seconds"
                        )
                    })

    previously_inside = inside_now
    writer.write(frame)

cap.release()
writer.release()

with open(ALERTS, "w", encoding="utf-8") as f:
    json.dump(alerts, f, indent=2)

print(f"Annotated video: {OUTPUT}")
print(f"Alerts JSON: {ALERTS}")
print(f"Intrusion events: {len(alerts)}")
