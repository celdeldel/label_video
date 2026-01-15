# tracked_labels.py
# Usage:
#   python tracked_labels.py --in sam_1202666338071409.mp4 --out labeled_video_tracked.mp4
#
# Requirements:
#   pip install ultralytics opencv-python supervision numpy

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

LABELS = ["police man 1", "police man 2", "police man robot"]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = max(1.0, area_a + area_b - inter)
    return inter / union

def draw_label(frame, text, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    tx = (x1 + x2) // 2
    ty = max(25, y1 - 10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    bx1 = max(0, tx - tw // 2 - pad)
    by1 = max(0, ty - th - pad)
    bx2 = min(frame.shape[1] - 1, tx + tw // 2 + pad)
    by2 = min(frame.shape[0] - 1, ty + pad)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
    cv2.putText(frame, text, (tx - tw // 2, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main(inp, outp):
    # Detector: YOLOv8 (person class is robust; robot often also detected as person, so we disambiguate by size/position)
    model = YOLO("yolov8n.pt")  # switch to yolov8s.pt for higher accuracy

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # Tracker
    tracker = sv.ByteTrack(0.25, 30, 0.8, float(fps))

    # We will “lock” identities once we map tracks -> our three labels.
    locked = {}   # label -> track_id
    reverse = {}  # track_id -> label

    # Helper: choose 3 best candidates from detections (prefer large & central for robot + two distinct humans)
    def pick_three(dets_xyxy, confs):
        if len(dets_xyxy) <= 3:
            return list(range(len(dets_xyxy)))
        # score by area + confidence
        areas = (dets_xyxy[:, 2] - dets_xyxy[:, 0]) * (dets_xyxy[:, 3] - dets_xyxy[:, 1])
        scores = 0.7 * confs + 0.3 * (areas / (W * H))
        idx = np.argsort(scores)[::-1][:3]
        return idx.tolist()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # YOLO inference
        res = model(frame, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            writer.write(frame)
            continue

        # Keep only "person" class (COCO id 0)
        cls = boxes.cls.cpu().numpy().astype(int)
        mask = (cls == 0)
        if mask.sum() == 0:
            writer.write(frame)
            continue

        xyxy = boxes.xyxy.cpu().numpy()[mask]
        conf = boxes.conf.cpu().numpy()[mask]

        # keep top 3 candidate persons (usually: 2 policemen + robot)
        keep_idx = pick_three(xyxy, conf)
        xyxy = xyxy[keep_idx]
        conf = conf[keep_idx]

        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.zeros(len(xyxy), dtype=int),
        )

        # tracking update
        tracked = tracker.update_with_detections(detections)

        # On first stable moment, assign our labels to 3 tracks:
        # Rule-of-thumb mapping:
        #   - leftmost -> police man 1
        #   - rightmost -> police man 2
        #   - remaining (often central / largest) -> police man robot
        if len(reverse) < 3 and len(tracked) >= 3:
            # sort by x-center
            centers_x = (tracked.xyxy[:, 0] + tracked.xyxy[:, 2]) / 2.0
            order = np.argsort(centers_x)
            left = int(tracked.tracker_id[order[0]])
            right = int(tracked.tracker_id[order[-1]])
            mid = [int(t) for t in tracked.tracker_id if int(t) not in (left, right)][0]

            # if robot tends to be the largest, swap mid with largest if needed
            areas = (tracked.xyxy[:, 2] - tracked.xyxy[:, 0]) * (tracked.xyxy[:, 3] - tracked.xyxy[:, 1])
            largest_tid = int(tracked.tracker_id[int(np.argmax(areas))])
            if largest_tid not in (left, right):
                mid = largest_tid

            # lock
            cand = {
                "police man 1": left,
                "police man 2": right,
                "police man robot": mid
            }
            locked = cand
            reverse = {v: k for k, v in locked.items()}

        # draw labels for locked tracks (fallback: draw nothing until locked)
        if reverse:
            for i in range(len(tracked)):
                tid = int(tracked.tracker_id[i])
                if tid in reverse:
                    draw_label(frame, reverse[tid], tracked.xyxy[i])

        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Saved: {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()
    main(args.inp, args.outp)
