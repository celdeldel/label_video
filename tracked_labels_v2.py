import argparse
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

LABELS = ["police man 1", "police man 2", "police man robot"]

def crop_safe(img, xyxy, pad=0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def color_hist_feat(bgr_crop, bins=32):
    """Fast, robust-ish appearance feature: HSV histogram."""
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [bins,bins], [0,180,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def cosine_sim(a, b, eps=1e-9):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + eps))

def draw_label(frame, text, xyxy, used_rects):
    """Draw label above bbox, shifting to avoid overlaps with previously drawn labels."""
    x1, y1, x2, y2 = map(int, xyxy)
    cx = (x1 + x2) // 2
    y = max(10, y1 - 10)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = 6

    # initial label box
    bx1 = max(0, cx - tw // 2 - pad)
    by1 = max(0, y - th - pad)
    bx2 = min(frame.shape[1] - 1, cx + tw // 2 + pad)
    by2 = min(frame.shape[0] - 1, y + pad)

    # de-overlap: if intersects existing, shift up until clear (or hit top)
    def intersects(r1, r2):
        ax1, ay1, ax2, ay2 = r1
        bx1, by1, bx2, by2 = r2
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    shift = 0
    rect = (bx1, by1, bx2, by2)
    while any(intersects(rect, r) for r in used_rects) and by1 > 5:
        shift += (th + 2*pad + 4)
        by1 = max(0, by1 - (th + 2*pad + 4))
        by2 = max(by1 + th + 2*pad, by2 - (th + 2*pad + 4))
        rect = (bx1, by1, bx2, by2)
        if shift > 5*(th + 2*pad):  # safety
            break

    used_rects.append(rect)

    # draw
    cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
    cv2.putText(frame, text, (rect[0] + pad, rect[3] - pad),
                font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main(inp, outp, conf=0.25):
    model = YOLO("yolov8s.pt")  # s for better stability than n

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(outp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # We will build template appearance features for each label using the first ~1s.
    # Then each frame we match current tracked boxes to labels by similarity (Hungarian assignment),
    # but still benefit from BoT-SORT's ID stability.
    templates = {lab: [] for lab in LABELS}
    template_feats = {lab: None for lab in LABELS}

    # Label assignment state
    # track_id -> label (preferred); label -> track_id
    tid_to_label = {}
    label_to_tid = {}

    frame_idx = 0
    template_build_frames = int(round(fps * 1.0))  # first second
    last_seen = {}  # tid -> last bbox

    # Use Ultralytics built-in tracking with BoT-SORT (appearance aware)
    # persist=True keeps tracks across frames.
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.track(
            frame,
            conf=conf,
            iou=0.5,
            classes=[0],               # person
            tracker="botsort.yaml",    # appearance-aware
            persist=True,
            verbose=False
        )[0]

        if results.boxes is None or len(results.boxes) == 0 or results.boxes.id is None:
            writer.write(frame)
            continue

        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        tids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        # Keep top 3 persons by confidence/area (your clip has 3 relevant subjects)
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        score = 0.7 * confs + 0.3 * (areas / (W * H))
        keep = np.argsort(score)[::-1][:3]
        xyxy = xyxy[keep]
        tids = tids[keep]

        # update last seen
        for t, b in zip(tids, xyxy):
            last_seen[int(t)] = b

        # --- Template build phase (first second) ---
        if frame_idx <= template_build_frames:
            # Provisional label order: left, right, remaining (robot tends to be middle / large)
            cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
            order = np.argsort(cx)
            left_i = order[0]
            right_i = order[-1]
            mid_i = [i for i in range(len(order)) if i not in (left_i, right_i)][0]

            # If "robot" is largest, prefer that for robot slot
            largest_i = int(np.argmax(areas[keep]))
            robot_i = largest_i if largest_i not in (left_i, right_i) else mid_i
            mid_i = robot_i

            mapping = {
                "police man 1": int(tids[left_i]),
                "police man 2": int(tids[right_i]),
                "police man robot": int(tids[mid_i]),
            }

            # Collect appearance features for each template label
            for lab, tid in mapping.items():
                b = last_seen.get(tid)
                crop = crop_safe(frame, b, pad=6)
                if crop is None:
                    continue
                templates[lab].append(color_hist_feat(crop))

            # finalize template feats at end of build window
            if frame_idx == template_build_frames:
                for lab in LABELS:
                    if len(templates[lab]) == 0:
                        raise RuntimeError(f"Could not build template for {lab} (no crops collected).")
                    template_feats[lab] = np.mean(np.stack(templates[lab], axis=0), axis=0)

                # Initialize tid/label map using the most recent mapping we computed
                tid_to_label = {v: k for k, v in mapping.items()}
                label_to_tid = {k: v for k, v in mapping.items()}

        # --- After templates exist: enforce identity by matching current boxes to labels ---
        if all(template_feats[lab] is not None for lab in LABELS):
            # compute current feats for each detected track
            curr_feats = []
            valid = []
            for t, b in zip(tids, xyxy):
                crop = crop_safe(frame, b, pad=6)
                if crop is None:
                    continue
                curr_feats.append(color_hist_feat(crop))
                valid.append((int(t), b))

            if len(valid) == 3:
                # similarity matrix: labels x detections
                sim = np.zeros((3, 3), dtype=np.float32)
                for i, lab in enumerate(LABELS):
                    for j, (t, b) in enumerate(valid):
                        sim[i, j] = cosine_sim(template_feats[lab], curr_feats[j])

                # Hungarian assignment maximizes similarity => minimize negative similarity
                r, c = linear_sum_assignment(-sim)

                # Update mapping according to best global match
                tid_to_label = {}
                label_to_tid = {}
                for i, j in zip(r, c):
                    lab = LABELS[i]
                    tid = valid[j][0]
                    tid_to_label[tid] = lab
                    label_to_tid[lab] = tid

        # --- Draw ---
        used_rects = []
        for t, b in zip(tids, xyxy):
            t = int(t)
            lab = tid_to_label.get(t)
            if lab:
                draw_label(frame, lab, b, used_rects)

        writer.write(frame)

    writer.release()
    cap.release()
    print(f"Saved: {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    main(args.inp, args.outp, conf=args.conf)
