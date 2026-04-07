#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Segmentation ONNX export + PT/ONNX parity check.

Design goals:
1. Export `models/yolo11l-seg.pt` -> `models/yolo11l-seg.onnx`.
2. ONNX inference path uses ONLY onnxruntime + numpy/cv2 (no torch/ultralytics).
3. Compare PT (reference) vs ONNX (runtime path) on sample images.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort


def _require_ultralytics_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "ultralytics is required for conversion/reference PT inference."
        ) from exc
    return YOLO


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> Optional[float]:
    if mask_a is None or mask_b is None:
        return None
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b.astype(np.uint8), (mask_a.shape[1], mask_a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "max": float("nan"), "p95": float("nan")}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "p95": float(np.percentile(arr, 95)),
    }


def _collect_sample_images(sample_dirs: Sequence[str], max_samples: int) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    images: List[str] = []
    for folder in sample_dirs:
        if not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    images.append(os.path.join(root, name))
    return sorted(set(images))[:max_samples]


@dataclass
class Detection:
    box: np.ndarray  # xyxy in original image space
    conf: float
    cls: int
    mask: Optional[np.ndarray]


class ORTYoloSeg:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        if providers is None:
            available = set(ort.get_available_providers())
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

    @staticmethod
    def _letterbox(img_bgr: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        h0, w0 = img_bgr.shape[:2]
        new_h, new_w = new_shape
        r = min(new_h / h0, new_w / w0)
        resized_w = int(round(w0 * r))
        resized_h = int(round(h0 * r))

        resized = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        dw = new_w - resized_w
        dh = new_h - resized_h
        dw /= 2.0
        dh /= 2.0

        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))

        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return out, r, (dw, dh)

    def _prepare(self, image_bgr: np.ndarray, imgsz: int) -> Tuple[np.ndarray, Dict[str, object]]:
        lb, ratio, (dw, dh) = self._letterbox(image_bgr, (imgsz, imgsz))
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        meta = {
            "orig_shape": image_bgr.shape[:2],  # h, w
            "img_shape": lb.shape[:2],          # h, w
            "ratio": ratio,
            "pad": (dw, dh),
        }
        return tensor, meta

    @staticmethod
    def _scale_boxes_to_orig(boxes_xyxy: np.ndarray, meta: Dict[str, object]) -> np.ndarray:
        dw, dh = meta["pad"]
        ratio = float(meta["ratio"])
        orig_h, orig_w = meta["orig_shape"]

        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= max(ratio, 1e-8)
        return _clip_boxes(boxes, orig_w, orig_h)

    @staticmethod
    def _decode_masks(
        mask_coeffs: np.ndarray,  # [N, mask_dim]
        proto: np.ndarray,        # [mask_dim, mh, mw]
        boxes_img_xyxy: np.ndarray,  # [N,4] in letterboxed img space
        meta: Dict[str, object],
    ) -> List[np.ndarray]:
        mask_dim, mh, mw = proto.shape
        img_h, img_w = meta["img_shape"]
        orig_h, orig_w = meta["orig_shape"]
        dw, dh = meta["pad"]

        proto_flat = proto.reshape(mask_dim, -1)
        raw = (1.0 / (1.0 + np.exp(-(mask_coeffs @ proto_flat)))).reshape(-1, mh, mw)

        masks: List[np.ndarray] = []
        sx = mw / float(img_w)
        sy = mh / float(img_h)

        for i in range(raw.shape[0]):
            m = raw[i]
            x1, y1, x2, y2 = boxes_img_xyxy[i]
            px1, py1 = int(max(0, np.floor(x1 * sx))), int(max(0, np.floor(y1 * sy)))
            px2, py2 = int(min(mw, np.ceil(x2 * sx))), int(min(mh, np.ceil(y2 * sy)))
            if px2 <= px1 or py2 <= py1:
                masks.append(np.zeros((orig_h, orig_w), dtype=bool))
                continue

            crop_mask = np.zeros_like(m, dtype=np.float32)
            crop_mask[py1:py2, px1:px2] = m[py1:py2, px1:px2]

            up = cv2.resize(crop_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            left = int(round(dw - 0.1))
            right = int(round(dw + 0.1))
            top = int(round(dh - 0.1))
            bottom = int(round(dh + 0.1))
            x_start = max(0, left)
            y_start = max(0, top)
            x_end = img_w - max(0, right)
            y_end = img_h - max(0, bottom)
            unpad = up[y_start:y_end, x_start:x_end]
            if unpad.size == 0:
                masks.append(np.zeros((orig_h, orig_w), dtype=bool))
                continue

            mask_orig = cv2.resize(unpad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            masks.append(mask_orig > 0.5)
        return masks

    def predict(
        self,
        image_path: str,
        imgsz: int = 640,
        conf_thres: float = 0.25,
        class_filter: Optional[Iterable[int]] = None,
    ) -> List[Detection]:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        tensor, meta = self._prepare(img, imgsz=imgsz)
        outs = self.session.run(self.output_names, {self.input_name: tensor})
        if len(outs) != 2:
            raise RuntimeError(f"Unexpected output count: {len(outs)}")

        pred = outs[0][0]   # [C,N] or [N,C]
        proto = outs[1][0]  # [mask_dim, mh, mw]
        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected pred rank: {pred.shape}")

        # Normalize to [N, C]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)

        mask_dim = int(proto.shape[0])
        ch = int(pred.shape[1])
        filter_set = set(int(c) for c in class_filter) if class_filter is not None else None

        # Only support exported nms=True format: [N, 6 + mask_dim]
        if not (ch >= 6 + mask_dim and (ch - 6) == mask_dim):
            raise RuntimeError(
                f"Unexpected ONNX output format: pred shape={pred.shape}, mask_dim={mask_dim}. "
                "Please re-export with nms=True."
            )

        boxes_xyxy_img = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        cls_ids_k = np.round(pred[:, 5]).astype(np.int32)
        mask_coeff_k = pred[:, 6 : 6 + mask_dim].astype(np.float32)

        keep_conf = scores >= conf_thres
        if filter_set is not None:
            keep_cls = np.array([c in filter_set for c in cls_ids_k], dtype=bool)
        else:
            keep_cls = np.ones_like(cls_ids_k, dtype=bool)

        keep = keep_conf & keep_cls
        if not np.any(keep):
            return []

        boxes_xyxy_img = boxes_xyxy_img[keep]
        scores = scores[keep]
        cls_ids_k = cls_ids_k[keep]
        mask_coeff_k = mask_coeff_k[keep]

        masks = self._decode_masks(mask_coeff_k, proto, boxes_xyxy_img, meta)
        boxes_xyxy_orig = self._scale_boxes_to_orig(boxes_xyxy_img, meta)

        out: List[Detection] = []
        for i in range(boxes_xyxy_orig.shape[0]):
            out.append(
                Detection(
                    box=boxes_xyxy_orig[i].astype(np.float32),
                    conf=float(scores[i]),
                    cls=int(cls_ids_k[i]),
                    mask=masks[i],
                )
            )
        return out


def _extract_pt_detections(result, class_filter: Optional[Iterable[int]] = None) -> List[Detection]:
    dets: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    filter_set = set(class_filter) if class_filter is not None else None

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

    masks = None
    if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
        masks = result.masks.data.detach().cpu().numpy()

    for i in range(len(boxes)):
        cls_id = int(classes[i])
        if filter_set is not None and cls_id not in filter_set:
            continue
        mask = None
        if masks is not None and i < len(masks):
            mask = masks[i] > 0.5
        dets.append(Detection(box=boxes[i].astype(np.float32), conf=float(confs[i]), cls=cls_id, mask=mask))
    return dets


def convert_yolo(
    pt_path: str = "models/yolo11l-seg.pt",
    onnx_path: str = "models/yolo11l-seg.onnx",
    imgsz: int = 640,
    opset: int = 13,
    dynamic: bool = True,
    conf: float = 0.25,
    iou: float = 0.7,
) -> str:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model not found: {pt_path}")

    YOLO = _require_ultralytics_yolo()
    model = YOLO(pt_path)
    exported = model.export(
        format="onnx",
        nms=True,
        dynamic=dynamic,
        opset=opset,
        imgsz=imgsz,
        simplify=False,
        conf=conf,
        iou=iou,
        batch=1,
        half=True
    )

    exported_path = str(exported)
    if not os.path.exists(exported_path):
        raise RuntimeError(f"Export finished but ONNX missing: {exported_path}")

    if os.path.abspath(exported_path) != os.path.abspath(onnx_path):
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        os.replace(exported_path, onnx_path)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[YOLO] Exported ONNX: {onnx_path}")
    print("[YOLO] Inputs:")
    for i in sess.get_inputs():
        print(f"  - {i.name}: {i.shape} {i.type}")
    print("[YOLO] Outputs:")
    for o in sess.get_outputs():
        print(f"  - {o.name}: {o.shape} {o.type}")
    return onnx_path


def check_yolo(
    pt_path: str = "models/yolo11l-seg.pt",
    onnx_path: str = "models/yolo11l-seg.onnx",
    sample_dirs: Optional[List[str]] = None,
    max_samples: int = 20,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.7,
) -> Dict[str, object]:
    if sample_dirs is None:
        sample_dirs = ["img", "img_copy"]
    class_filter = [14]

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model not found: {pt_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    images = _collect_sample_images(sample_dirs, max_samples=max_samples)
    if not images:
        raise RuntimeError(f"No test images found in: {sample_dirs}")

    YOLO = _require_ultralytics_yolo()
    pt_model = YOLO(pt_path)
    ort_model = ORTYoloSeg(onnx_path, providers=["CPUExecutionProvider"])

    count_equal = 0
    matched_box_ious: List[float] = []
    matched_conf_abs_diffs: List[float] = []
    matched_mask_ious: List[float] = []
    failures: List[str] = []

    for img_path in images:
        pt_res = pt_model(
            img_path,
            conf=conf,
            iou=iou,
            agnostic_nms=False,
            verbose=False,
            device="cpu",
        )[0]
        pt_dets = _extract_pt_detections(pt_res, class_filter=class_filter)
        ort_dets = ort_model.predict(
            img_path,
            imgsz=imgsz,
            conf_thres=conf,
            class_filter=class_filter,
        )

        if len(pt_dets) == len(ort_dets):
            count_equal += 1
        else:
            failures.append(f"{img_path}: detection count mismatch PT={len(pt_dets)} ORT={len(ort_dets)}")

        used = set()
        per_box_ious: List[float] = []
        per_conf_diffs: List[float] = []
        per_mask_ious: List[float] = []

        for p in sorted(pt_dets, key=lambda x: x.conf, reverse=True):
            best_iou = -1.0
            best_j = -1
            for j, o in enumerate(ort_dets):
                if j in used:
                    continue
                if p.cls != o.cls:
                    continue
                iou_val = _box_iou_xyxy(p.box, o.box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j < 0:
                continue
            used.add(best_j)
            o = ort_dets[best_j]
            per_box_ious.append(best_iou)
            per_conf_diffs.append(abs(float(p.conf) - float(o.conf)))
            mask_iou = _mask_iou(p.mask, o.mask)
            if mask_iou is not None:
                per_mask_ious.append(mask_iou)

        matched_box_ious.extend(per_box_ious)
        matched_conf_abs_diffs.extend(per_conf_diffs)
        matched_mask_ious.extend(per_mask_ious)

    total = len(images)
    count_consistency = count_equal / total if total else 0.0
    box_stats = _summary_stats(matched_box_ious)
    conf_stats = _summary_stats(matched_conf_abs_diffs)
    mask_stats = _summary_stats(matched_mask_ious)

    passed = (
        count_consistency >= 0.90
        and box_stats["mean"] >= 0.90
        and conf_stats["mean"] <= 0.08
        and (np.isnan(mask_stats["mean"]) or mask_stats["mean"] >= 0.85)
    )

    print(f"[YOLO][CHECK] samples={total}")
    print(f"[YOLO][CHECK] count_consistency={count_consistency:.4f} (>=0.90)")
    print(
        f"[YOLO][CHECK] box_iou mean={box_stats['mean']:.4f}, "
        f"p95={box_stats['p95']:.4f}, max={box_stats['max']:.4f} (mean>=0.90)"
    )
    print(
        f"[YOLO][CHECK] conf_abs_diff mean={conf_stats['mean']:.4f}, "
        f"p95={conf_stats['p95']:.4f}, max={conf_stats['max']:.4f} (mean<=0.08)"
    )
    print(
        f"[YOLO][CHECK] mask_iou mean={mask_stats['mean']:.4f}, "
        f"p95={mask_stats['p95']:.4f}, max={mask_stats['max']:.4f} (mean>=0.85)"
    )
    print(f"[YOLO][CHECK] result={'PASS' if passed else 'FAIL'}")

    if failures:
        print("[YOLO][CHECK] mismatch samples (up to 10):")
        for item in failures[:10]:
            print(f"  - {item}")

    return {
        "passed": passed,
        "samples": total,
        "count_consistency": count_consistency,
        "box_iou": box_stats,
        "conf_abs_diff": conf_stats,
        "mask_iou": mask_stats,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO seg to ONNX and validate PT vs ORT parity")
    parser.add_argument("--pt", default="models/yolo11l-seg.pt", help="source PT model path")
    parser.add_argument("--onnx", default="models/yolo11l-seg.onnx", help="target ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="export/infer image size")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset")
    parser.add_argument("--static", action="store_true", help="disable dynamic shape export")
    parser.add_argument("--skip-export", action="store_true", help="skip export and only run check")
    parser.add_argument("--skip-check", action="store_true", help="skip check and only export")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-samples", type=int, default=20, help="max sample images for check")
    parser.add_argument("--sample-dir", action="append", default=None, help="sample image directory (repeatable)")
    args = parser.parse_args()

    if not args.skip_export:
        convert_yolo(
            pt_path=args.pt,
            onnx_path=args.onnx,
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=not args.static,
            conf=args.conf,
            iou=args.iou,
        )

    if not args.skip_check:
        report = check_yolo(
            pt_path=args.pt,
            onnx_path=args.onnx,
            sample_dirs=args.sample_dir,
            max_samples=args.max_samples,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        return 0 if report["passed"] else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
