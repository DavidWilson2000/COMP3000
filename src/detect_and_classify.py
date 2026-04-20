import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def blur_score(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--det_model", required=True)
    parser.add_argument("--cls_model", required=True)

    parser.add_argument("--conf_det", type=float, default=0.25)
    parser.add_argument("--imgsz_det", type=int, default=640)

    parser.add_argument("--pad", type=float, default=0.15)
    parser.add_argument("--min_crop", type=int, default=64)
    parser.add_argument("--blur_thresh", type=float, default=50.0)

    parser.add_argument("--conf_cls", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--imgsz_cls", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source_dir = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(args.det_model)
    classifier = YOLO(args.cls_model)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    img_paths = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in exts])

    if args.limit and args.limit > 0:
        img_paths = img_paths[: args.limit]

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        vis = img.copy()
        h, w = img.shape[:2]

        det = detector.predict(img, imgsz=args.imgsz_det, conf=args.conf_det, verbose=False)[0]
        if det.boxes is None or len(det.boxes) == 0:
            cv2.imwrite(str(out_dir / img_path.name), vis)
            continue

        for box in det.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            bw = x2 - x1
            bh = y2 - y1
            x1 = int(x1 - bw * args.pad)
            y1 = int(y1 - bh * args.pad)
            x2 = int(x2 + bw * args.pad)
            y2 = int(y2 + bh * args.pad)

            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if min(crop.shape[:2]) < args.min_crop:
                continue
            if blur_score(crop) < args.blur_thresh:
                continue

            cls = classifier.predict(crop, imgsz=args.imgsz_cls, verbose=False)[0]
            probs = cls.probs
            top1 = probs.top1
            conf1 = float(probs.top1conf)
            name1 = cls.names[top1]

            conf2 = 0.0
            if hasattr(probs, "top5conf") and len(probs.top5conf) > 1:
                conf2 = float(probs.top5conf[1])

            final_name = name1
            if conf1 < args.conf_cls or (conf1 - conf2) < args.margin:
                final_name = "unknown"

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{final_name} {conf1:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imwrite(str(out_dir / img_path.name), vis)

    print(f"Saved annotated frames to: {out_dir}")

if __name__ == "__main__":
    main()
