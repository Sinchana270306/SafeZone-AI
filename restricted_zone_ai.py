import cv2, json, os
from datetime import datetime
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "rest.mp4"
MODEL_PATH = "yolo11n.pt"
OUTPUT_VIDEO = "output/restricted_zone.mp4"
ALERTS_FILE = "output/alerts.json"
CONFIDENCE = 0.40

# Adjust these coordinates to your video resolution and camera view.
DANGER_POLYGON = [(80, 650), (1615, 650), (1615, 865), (80, 865)]

def inside_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0

def main():
    os.makedirs("output", exist_ok=True)
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    polygon = np.array(DANGER_POLYGON, dtype=np.int32).reshape((-1, 1, 2))
    polygon[:, 0, 0] = np.clip(polygon[:, 0, 0], 0, width - 1)
    polygon[:, 0, 1] = np.clip(polygon[:, 0, 1], 0, height - 1)

    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    alerts, previous_inside, frame_no = [], set(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 classes=[0], conf=CONFIDENCE, verbose=False)[0]

            # Fixed danger polygon is drawn identically on every frame.
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [polygon], True, (0, 0, 255), 3)
            px, py = polygon[0, 0]
            cv2.putText(frame, "DANGER / RESTRICTED ZONE", (int(px), max(25, int(py)-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            current_inside = set()
            boxes = result.boxes
            if boxes is not None and len(boxes):
                coords = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [-1]*len(coords)
                for box, tid in zip(coords, ids):
                    x1, y1, x2, y2 = map(int, box)
                    foot = (int((x1+x2)/2), y2)
                    danger = inside_zone(foot, polygon)
                    color = (0, 0, 255) if danger else (0, 255, 0)
                    label = ("DANGER / INTRUSION" if danger else "SAFE") + (f" ID:{tid}" if tid >= 0 else "")
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.circle(frame, foot, 5, color, -1)
                    cv2.putText(frame, label, (x1,max(25,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    if danger and tid >= 0:
                        current_inside.add(int(tid))
                        if tid not in previous_inside:
                            alerts.append({"track_id": int(tid), "event": "PERSON_ENTERED_DANGER_ZONE",
                                           "frame": frame_no, "timestamp": datetime.now().isoformat(timespec="seconds"),
                                           "foot_point": list(foot)})
            previous_inside = current_inside
            cv2.putText(frame, f"Frame {frame_no} | Alerts {len(alerts)}", (20,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            writer.write(frame)
            cv2.imshow("SafeZone AI - press Q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
        with open(ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)
    print("Video:", OUTPUT_VIDEO, "| Alerts:", ALERTS_FILE, "| Events:", len(alerts))

if __name__ == "__main__":
    main()
