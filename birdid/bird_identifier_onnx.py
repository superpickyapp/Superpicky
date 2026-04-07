#!/usr/bin/env python3
"""
ONNX bird classifier module.

This module mirrors birdid.bird_identifier public API while replacing
PyTorch classification with ONNX Runtime inference.
"""

__version__ = "1.0.0"

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter
from PIL.ExifTags import GPSTAGS, TAGS

try:
    import cv2
except Exception:
    cv2 = None

try:
    from tools.i18n import t as _t
except ModuleNotFoundError:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from tools.i18n import t as _t

try:
    import rawpy
    import imageio  # noqa: F401

    RAW_SUPPORT = True
except ImportError:
    RAW_SUPPORT = False

YOLO_AVAILABLE = cv2 is not None

BIRDID_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BIRDID_DIR)


def get_birdid_path(relative_path: str) -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "birdid", relative_path)
    return os.path.join(BIRDID_DIR, relative_path)


def get_project_path(relative_path: str) -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(PROJECT_ROOT, relative_path)


def get_user_data_dir() -> str:
    if sys.platform == "darwin":
        user_data_dir = os.path.expanduser("~/Documents/SuperPicky_Data")
    elif sys.platform == "win32":
        user_data_dir = os.path.join(os.path.expanduser("~"), "Documents", "SuperPicky_Data")
    else:
        user_data_dir = os.path.join(os.path.expanduser("~"), "Documents", "SuperPicky_Data")
    os.makedirs(user_data_dir, exist_ok=True)
    return user_data_dir


MODEL_PATH = get_project_path("models/model20240824.onnx")
OSEA_NUM_CLASSES = 11000
DATABASE_PATH = get_birdid_path("data/bird_reference.sqlite")
YOLO_MODEL_PATH = get_project_path("models/yolo11l-seg.onnx")
try:
    from config import config as _global_config

    YOLO_MODEL_PATH = _global_config.ai.get_model_path_onnx()
except Exception:
    pass


_classifier = None
_db_manager = None
_yolo_detector = None
_avonet_filter = None


def _create_session_with_fallback(model_path: str, providers: List[str], label: str) -> ort.InferenceSession:
    try:
        return ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f"[{label}] CUDA init failed, fallback to CPU: {exc}")
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def get_classifier_device() -> List[str]:
    available = set(ort.get_available_providers())
    providers = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


CLASSIFIER_DEVICE = get_classifier_device()

_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_YOLO_INPUT_SIZE = 640


def _resize_shorter_side_to_256(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int(round((256.0 / width) * height))
    else:
        new_height = 256
        new_width = int(round((256.0 / height) * width))
    return image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)


def _center_crop_224(image: Image.Image) -> Image.Image:
    width, height = image.size
    left = max(0, (width - 224) // 2)
    top = max(0, (height - 224) // 2)
    return image.crop((left, top, left + 224, top + 224))


def _to_normalized_nchw(image: Image.Image) -> np.ndarray:
    img_data = np.asarray(image, dtype=np.float32) / 255.0
    img_data = (img_data - _NORM_MEAN) / _NORM_STD
    return np.expand_dims(np.transpose(img_data, (2, 0, 1)), axis=0).astype(np.float32)


def _preprocess_image(image: Image.Image, is_yolo_cropped: bool) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    if is_yolo_cropped:
        processed = image.resize((224, 224), resample=Image.Resampling.LANCZOS)
    else:
        processed = _center_crop_224(_resize_shorter_side_to_256(image))
    return _to_normalized_nchw(processed)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=0, keepdims=True)


def _topk(values: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(-values, k - 1)[:k]
    sorted_idx = idx[np.argsort(-values[idx])]
    return values[sorted_idx], sorted_idx


class ONNXBirdClassifier:
    def __init__(self, model_path: str, providers: List[str]):
        self.model_path = model_path
        self.providers = providers
        self.session = _create_session_with_fallback(model_path, providers, "BirdID")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_logits(self, image: Image.Image, is_yolo_cropped: bool) -> np.ndarray:
        input_tensor = _preprocess_image(image, is_yolo_cropped=is_yolo_cropped)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0][0]


def get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"ONNX classifier model not found: {MODEL_PATH}")
        _classifier = ONNXBirdClassifier(MODEL_PATH, CLASSIFIER_DEVICE)
        print(f"[BirdID] ONNX OSEA model loaded, providers: {CLASSIFIER_DEVICE}")
    return _classifier


def get_bird_model():
    return get_classifier()


def get_database_manager():
    global _db_manager
    if _db_manager is None:
        try:
            try:
                from birdid.bird_database_manager import BirdDatabaseManager
            except Exception:
                from bird_database_manager import BirdDatabaseManager

            if os.path.exists(DATABASE_PATH):
                _db_manager = BirdDatabaseManager(DATABASE_PATH)
        except Exception as exc:
            print(f"[BirdID] database manager load failed: {exc}")
            _db_manager = False
    return _db_manager if _db_manager is not False else None


def get_species_filter():
    global _avonet_filter
    if _avonet_filter is None:
        try:
            try:
                from birdid.avonet_filter import AvonetFilter
            except Exception:
                from avonet_filter import AvonetFilter

            _avonet_filter = AvonetFilter()
            if _avonet_filter.is_available():
                print("[BirdID] avonet filter loaded")
            else:
                _avonet_filter = None
        except Exception as exc:
            print(f"[BirdID] avonet init failed: {exc}")
            return None
    return _avonet_filter


class YOLOBirdDetector:
    def __init__(self, model_path: str = None):
        if not YOLO_AVAILABLE:
            self.session = None
            self.input_name = None
            self.output_names = []
            return

        if model_path is None:
            model_path = YOLO_MODEL_PATH

        try:
            providers = get_classifier_device()
            self.session = _create_session_with_fallback(model_path, providers, "BirdID YOLO")
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
        except Exception as exc:
            print(_t("logs.yolo_load_failed", e=exc))
            self.session = None
            self.input_name = None
            self.output_names = []

    @staticmethod
    def _letterbox(img_bgr: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        h0, w0 = img_bgr.shape[:2]
        new_h, new_w = new_shape
        ratio = min(new_h / h0, new_w / w0)
        resized_w = int(round(w0 * ratio))
        resized_h = int(round(h0 * ratio))
        resized = cv2.resize(img_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        dw = (new_w - resized_w) / 2.0
        dh = (new_h - resized_h) / 2.0
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, ratio, (dw, dh)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
        return boxes

    def _predict_bird_boxes(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, float]]:
        if self.session is None or self.input_name is None:
            return []

        rgb = np.array(image.convert("RGB"))
        img_h, img_w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lb, ratio, (dw, dh) = self._letterbox(bgr, (_YOLO_INPUT_SIZE, _YOLO_INPUT_SIZE))
        input_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(input_rgb, (2, 0, 1))[None, ...].astype(np.float32)

        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        if not outputs:
            return []
        pred = outputs[0][0]
        if pred.ndim != 2:
            return []
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)
        if pred.shape[1] < 6:
            return []

        scores = pred[:, 4].astype(np.float32)
        cls_ids = np.round(pred[:, 5]).astype(np.int32)
        keep = (scores >= confidence_threshold) & (cls_ids == 14)
        if not np.any(keep):
            return []

        boxes = pred[keep, :4].astype(np.float32)
        scores = scores[keep]
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes /= max(ratio, 1e-8)
        boxes = self._clip_boxes(boxes, img_w, img_h)

        detections: List[Dict[str, float]] = []
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                {
                    "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "confidence": float(scores[i]),
                }
            )
        return detections

    def detect_and_crop_bird(
        self,
        image_input,
        confidence_threshold: float = 0.25,
        padding_ratio: float = 0.15,
        fill_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tuple[Optional[Image.Image], str]:
        if self.session is None:
            return None, "YOLO model unavailable"

        try:
            if isinstance(image_input, str):
                image = load_image(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                return None, "Unsupported image input type"

            detections = self._predict_bird_boxes(image, confidence_threshold=confidence_threshold)
            if not detections:
                return None, _t("logs.no_bird_detected")

            best = max(detections, key=lambda x: x["confidence"])
            img_width, img_height = image.size

            x1, y1, x2, y2 = best["bbox"]
            bbox_width = x2 - x1
            bbox_height = y2 - y1

            max_side = max(bbox_width, bbox_height)
            target_side = int(max_side * (1 + padding_ratio))

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half = target_side // 2

            sq_x1 = cx - half
            sq_y1 = cy - half
            sq_x2 = cx + half
            sq_y2 = cy + half

            crop_x1 = max(0, sq_x1)
            crop_y1 = max(0, sq_y1)
            crop_x2 = min(img_width, sq_x2)
            crop_y2 = min(img_height, sq_y2)

            cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            crop_w, crop_h = cropped.size

            if crop_w != crop_h:
                sq_size = max(crop_w, crop_h)
                square = Image.new("RGB", (sq_size, sq_size), fill_color)
                paste_x = (sq_size - crop_w) // 2
                paste_y = (sq_size - crop_h) // 2
                square.paste(cropped, (paste_x, paste_y))
                cropped = square

            info = f"conf={best['confidence']:.3f}, size={cropped.size}"
            return cropped, info

        except Exception as exc:
            return None, f"Detection failed: {exc}"


def get_yolo_detector():
    global _yolo_detector
    if _yolo_detector is None and YOLO_AVAILABLE and os.path.exists(YOLO_MODEL_PATH):
        _yolo_detector = YOLOBirdDetector(YOLO_MODEL_PATH)
    return _yolo_detector


def _load_raw_via_exiftool(image_path: str) -> Image.Image:
    possible_paths = []
    if getattr(sys, "frozen", False):
        possible_paths.append(os.path.join(sys._MEIPASS, "exiftools_mac", "exiftool"))
    possible_paths += [
        os.path.join(PROJECT_ROOT, "exiftools_mac", "exiftool"),
        "/opt/homebrew/bin/exiftool",
        "/usr/local/bin/exiftool",
        "exiftool",
    ]
    exiftool = next((p for p in possible_paths if os.path.isfile(p)), "exiftool")

    for tag in ["-JpgFromRaw", "-PreviewImage", "-ThumbnailImage"]:
        result = subprocess.run([exiftool, "-b", tag, image_path], capture_output=True, timeout=15)
        if result.returncode == 0 and result.stdout and len(result.stdout) > 1000:
            return Image.open(BytesIO(result.stdout)).convert("RGB")

    raise Exception(f"Unsupported RAW format: {os.path.basename(image_path)}")


def _load_heif(image_path: str) -> Image.Image:
    try:
        import pillow_heif

        heif_file = pillow_heif.read_heif(image_path)
        return Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw").convert("RGB")
    except ImportError:
        raise Exception("Please install pillow-heif for HIF/HEIC support")


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    raw_extensions = {
        ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".dng", ".raf", ".orf", ".rw2",
        ".pef", ".srw", ".raw", ".rwl", ".3fr", ".fff", ".erf", ".mef", ".mos", ".mrw",
        ".x3f", ".hif", ".heif", ".heic",
    }

    if ext not in raw_extensions:
        return Image.open(image_path).convert("RGB")

    if ext in {".hif", ".heif", ".heic"}:
        return _load_heif(image_path)

    if not RAW_SUPPORT:
        raise ImportError("rawpy is required for RAW formats")

    try:
        with open(image_path, "rb") as f:
            with rawpy.imread(f) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        return Image.open(BytesIO(thumb.data)).convert("RGB")
                    if thumb.format == rawpy.ThumbFormat.BITMAP:
                        return Image.fromarray(thumb.data).convert("RGB")
                except Exception:
                    pass

                rgb = raw.postprocess(
                    use_camera_wb=True,
                    output_bps=8,
                    no_auto_bright=False,
                    auto_bright_thr=0.01,
                    half_size=True,
                )
                return Image.fromarray(rgb)
    except rawpy._rawpy.LibRawFileUnsupportedError:
        return _load_raw_via_exiftool(image_path)


def extract_gps_from_exif(image_path: str) -> Tuple[Optional[float], Optional[float], str]:
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None, "No EXIF"

        gps_info = {}
        for tag, value in exif_data.items():
            decoded_tag = TAGS.get(tag, tag)
            if decoded_tag == "GPSInfo":
                for gps_tag in value:
                    gps_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_decoded] = value[gps_tag]
                break

        if not gps_info:
            return None, None, "No GPS"

        def convert_to_degrees(coord, ref):
            d, m, s = coord
            decimal = d + (m / 60.0) + (s / 3600.0)
            if ref in ["S", "W"]:
                decimal = -decimal
            return decimal

        lat = lon = None
        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = convert_to_degrees(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = convert_to_degrees(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])

        if lat is not None and lon is not None:
            return lat, lon, f"GPS: {lat:.6f}, {lon:.6f}"

        return None, None, "GPS incomplete"

    except Exception:
        try:
            result = subprocess.run(["exiftool", "-json", "-gpslatitude", "-gpslongitude", "-gpslatituderef", "-gpslongituderef", image_path], capture_output=True, timeout=10)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout.decode("utf-8", errors="ignore"))
                if data:
                    item = data[0]
                    lat_str = item.get("GPSLatitude")
                    lon_str = item.get("GPSLongitude")
                    if lat_str and lon_str:
                        def parse_dms(dms_str):
                            match = re.search(r"(\d+)\s*deg\s*(\d+)'\s*([\d.]+)\"?", str(dms_str))
                            if match:
                                d, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
                                return d + m / 60.0 + s / 3600.0
                            return float(dms_str)

                        lat = parse_dms(lat_str)
                        lon = parse_dms(lon_str)
                        if str(item.get("GPSLatitudeRef", "N")).upper().startswith("S"):
                            lat = -lat
                        if str(item.get("GPSLongitudeRef", "E")).upper().startswith("W"):
                            lon = -lon
                        return lat, lon, f"GPS: {lat:.6f}, {lon:.6f}"
        except Exception:
            pass

    return None, None, "GPS parse failed"


def smart_resize(image: Image.Image, target_size: int = 224) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    if max_dim < 1000:
        return image.resize((target_size, target_size), Image.LANCZOS)
    resized = image.resize((256, 256), Image.LANCZOS)
    left = (256 - target_size) // 2
    top = (256 - target_size) // 2
    return resized.crop((left, top, left + target_size, top + target_size))


def apply_enhancement(image: Image.Image, method: str = "unsharp_mask") -> Image.Image:
    if method == "unsharp_mask":
        return image.filter(ImageFilter.UnsharpMask())
    if method == "edge_enhance_more":
        return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
    if method == "contrast_edge":
        enhanced = ImageEnhance.Brightness(image).enhance(1.2)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.3)
        return enhanced.filter(ImageFilter.EDGE_ENHANCE)
    if method == "desaturate":
        return ImageEnhance.Color(image).enhance(0.5)
    return image


def predict_bird(
    image: Image.Image,
    top_k: int = 5,
    species_class_ids: Optional[Set[int]] = None,
    is_yolo_cropped: bool = False,
    name_format: str = None,
) -> List[Dict]:
    model = get_classifier()
    db_manager = get_database_manager()

    logits = model.predict_logits(image, is_yolo_cropped=is_yolo_cropped)
    num_classes = min(10964, int(logits.shape[0]))
    logits = logits[:num_classes]

    probs = _softmax(logits / 0.9)
    k = min(100 if species_class_ids else top_k, len(probs))
    top_probs, top_indices = _topk(probs, k)

    results = []
    for i in range(len(top_indices)):
        class_id = int(top_indices[i])
        confidence = float(top_probs[i]) * 100.0

        min_confidence = 0.3 if species_class_ids else 1.0
        if confidence < min_confidence:
            continue

        cn_name = en_name = scientific_name = ebird_code = description = None

        if db_manager:
            info = db_manager.get_bird_by_class_id(class_id)
            if info:
                cn_name = info.get("chinese_simplified")
                en_name = info.get("english_name")
                scientific_name = info.get("scientific_name")
                ebird_code = info.get("ebird_code")
                description = info.get("short_description_zh")

        if not cn_name:
            cn_name = f"Unknown (ID: {class_id})"
            en_name = f"Unknown (ID: {class_id})"

        if name_format and name_format != "default" and db_manager:
            avilist_info = db_manager.get_avilist_names_by_class_id(class_id)
            if avilist_info and avilist_info.get("match_type") != "no_match":
                if name_format == "scientific":
                    en_name = avilist_info.get("scientific_name_avilist") or scientific_name or en_name
                else:
                    col = f"en_name_{name_format}"
                    alt_name = avilist_info.get(col)
                    if alt_name:
                        en_name = alt_name
                    elif name_format != "avilist" and avilist_info.get("en_name_avilist"):
                        en_name = avilist_info["en_name_avilist"]

        region_match = False
        if species_class_ids:
            if class_id in species_class_ids:
                region_match = True
            else:
                continue

        results.append(
            {
                "class_id": class_id,
                "cn_name": cn_name,
                "en_name": en_name,
                "scientific_name": scientific_name,
                "confidence": confidence,
                "ebird_code": ebird_code,
                "region_match": region_match,
                "description": description or "",
            }
        )

        if len(results) >= top_k:
            break

    return results


def identify_bird(
    image_path: str,
    use_yolo: bool = True,
    use_gps: bool = True,
    use_ebird: bool = True,
    country_code: str = None,
    region_code: str = None,
    top_k: int = 5,
    name_format: str = None,
) -> Dict:
    result = {
        "success": False,
        "image_path": image_path,
        "results": [],
        "yolo_info": None,
        "gps_info": None,
        "ebird_info": None,
        "error": None,
    }

    try:
        image = load_image(image_path)
        is_yolo_cropped = False

        if use_yolo and YOLO_AVAILABLE:
            width, height = image.size
            if max(width, height) > 640:
                detector = get_yolo_detector()
                if detector:
                    cropped, info = detector.detect_and_crop_bird(image)
                    if cropped:
                        image = cropped
                        result["yolo_info"] = info
                        result["cropped_image"] = cropped
                        is_yolo_cropped = True
                    else:
                        result["success"] = True
                        result["results"] = []
                        result["yolo_info"] = {"bird_count": 0}
                        return result

        species_class_ids = None
        lat = lon = None
        species_filter = None

        if use_ebird:
            species_filter = get_species_filter()
            if species_filter:
                if use_gps:
                    lat, lon, gps_msg = extract_gps_from_exif(image_path)
                    if lat and lon:
                        result["gps_info"] = {"latitude": lat, "longitude": lon, "info": gps_msg}
                        species_class_ids = species_filter.get_species_by_gps(lat, lon)

                if species_class_ids is None and (region_code or country_code):
                    effective_region = region_code or country_code
                    try:
                        ebird_ids, _ = species_filter.get_species_by_region_ebird(effective_region)
                        if ebird_ids:
                            species_class_ids = ebird_ids
                    except Exception:
                        pass
                    if not species_class_ids:
                        species_class_ids = species_filter.get_species_by_region(effective_region)

                if species_class_ids:
                    result["ebird_info"] = {
                        "enabled": True,
                        "species_count": len(species_class_ids),
                        "data_source": "avonet.db (offline)",
                        "region_code": region_code or country_code if not result.get("gps_info") else None,
                    }

        results = predict_bird(
            image,
            top_k=top_k,
            species_class_ids=species_class_ids,
            is_yolo_cropped=is_yolo_cropped,
            name_format=name_format,
        )

        if not results and species_class_ids:
            country_cls_ids = None
            country_cc = None
            if lat is not None and lon is not None and species_filter is not None:
                try:
                    country_cls_ids, country_cc = species_filter.get_species_by_country_ebird(lat, lon)
                except Exception:
                    pass

            if country_cls_ids:
                results = predict_bird(
                    image,
                    top_k=top_k,
                    species_class_ids=country_cls_ids,
                    is_yolo_cropped=is_yolo_cropped,
                    name_format=name_format,
                )
                if results:
                    if not result.get("ebird_info"):
                        result["ebird_info"] = {}
                    result["ebird_info"]["country_fallback"] = True
                    result["ebird_info"]["country_code"] = country_cc

            if not results:
                results = predict_bird(
                    image,
                    top_k=top_k,
                    species_class_ids=None,
                    is_yolo_cropped=is_yolo_cropped,
                    name_format=name_format,
                )
                if results and result.get("ebird_info"):
                    result["ebird_info"]["gps_fallback"] = True

        result["success"] = True
        result["results"] = results

    except Exception as exc:
        result["error"] = str(exc)

    return result


def quick_identify(image_path: str, top_k: int = 3) -> List[Dict]:
    return identify_bird(image_path, top_k=top_k).get("results", [])


def _print_topk(title: str, result: Dict, top_k: int) -> None:
    print(f"\n[{title}] top-{top_k}")
    if not result.get("success"):
        print(f"  failed: {result.get('error')}")
        return
    rows = result.get("results", [])
    if not rows:
        print("  (no results)")
        return
    for i, row in enumerate(rows[:top_k], 1):
        print(
            f"  {i}. class_id={row.get('class_id')} "
            f"cn={row.get('cn_name')} en={row.get('en_name')} "
            f"conf={row.get('confidence', 0):.2f}%"
        )


def _compare_result_sets(onnx_result: Dict, torch_result: Dict, top_k: int) -> Dict:
    onnx_rows = onnx_result.get("results", [])[:top_k]
    torch_rows = torch_result.get("results", [])[:top_k]

    onnx_ids = [int(r.get("class_id", -1)) for r in onnx_rows]
    torch_ids = [int(r.get("class_id", -1)) for r in torch_rows]

    common_ids = sorted(set(onnx_ids) & set(torch_ids))
    diffs = []
    for cid in common_ids:
        onnx_conf = next((float(r.get("confidence", 0.0)) for r in onnx_rows if int(r.get("class_id", -1)) == cid), 0.0)
        torch_conf = next((float(r.get("confidence", 0.0)) for r in torch_rows if int(r.get("class_id", -1)) == cid), 0.0)
        diffs.append(abs(onnx_conf - torch_conf))

    overlap = (len(common_ids) / max(1, top_k)) * 100.0
    mean_diff = float(np.mean(diffs)) if diffs else None
    max_diff = float(np.max(diffs)) if diffs else None
    top1_same = bool(onnx_ids and torch_ids and onnx_ids[0] == torch_ids[0])
    passed = top1_same and overlap >= 60.0

    return {
        "top1_same": top1_same,
        "onnx_ids": onnx_ids,
        "torch_ids": torch_ids,
        "common_ids": common_ids,
        "overlap_percent": overlap,
        "mean_abs_conf_diff": mean_diff,
        "max_abs_conf_diff": max_diff,
        "passed": passed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Bird classifier consistency test")
    parser.add_argument("--image", required=True, help="test image path")
    parser.add_argument("--top-k", type=int, default=5, help="return top-k results")
    parser.add_argument("--no-yolo", action="store_true", help="disable yolo crop")
    parser.add_argument("--no-gps", action="store_true", help="disable gps region filter")
    parser.add_argument("--no-ebird", action="store_true", help="disable ebird/avonet filter")
    parser.add_argument("--compare", action="store_true", default=True, help="compare with torch backend if available")
    parser.add_argument("--no-compare", dest="compare", action="store_false", help="skip torch comparison")
    args = parser.parse_args()

    print(f"YOLO available: {YOLO_AVAILABLE}")
    print(f"RAW support: {RAW_SUPPORT}")
    print(f"ONNX model: {MODEL_PATH}")

    onnx_result = identify_bird(
        args.image,
        use_yolo=not args.no_yolo,
        use_gps=not args.no_gps,
        use_ebird=not args.no_ebird,
        top_k=args.top_k,
    )
    _print_topk("ONNX", onnx_result, args.top_k)

    if not args.compare:
        sys.exit(0)

    has_torch = bool(importlib.util.find_spec("torch"))
    if not has_torch:
        print("\n[Compare] torch is unavailable; downgraded to ONNX smoke test only.")
        sys.exit(0)

    try:
        from birdid import bird_identifier as legacy_bird_identifier
    except Exception as exc:
        print(f"\n[Compare] failed to import torch backend: {exc}")
        print("[Compare] downgraded to ONNX smoke test only.")
        sys.exit(0)

    torch_result = legacy_bird_identifier.identify_bird(
        args.image,
        use_yolo=not args.no_yolo,
        use_gps=not args.no_gps,
        use_ebird=not args.no_ebird,
        top_k=args.top_k,
    )
    _print_topk("PyTorch", torch_result, args.top_k)

    compare = _compare_result_sets(onnx_result, torch_result, args.top_k)
    print("\n[Compare Summary]")
    print(json.dumps(compare, ensure_ascii=False, indent=2))
    if compare["passed"]:
        print("[Compare] PASS")
    else:
        print("[Compare] WARNING: predictions diverge beyond baseline rule.")
