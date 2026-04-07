"""
ONNX 版鸟类检测模块。

作为 `ai_model.py` 的 ONNX 替代实现，使用导出的 YOLO-seg 模型配合
onnxruntime 执行推理，同时尽量保持旧版接口、返回值和写库字段不变。
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from config import config
from tools.i18n import get_i18n
from tools.utils import log_message


DEFAULT_ONNX_IMGSZ = 640
_MODEL_INSTANCE: Optional["OnnxYoloSegModel"] = None
_MODEL_LOCK = threading.Lock()


@dataclass
class Detection:
    """单个检测结果。"""

    box: np.ndarray
    conf: float
    cls: int
    mask: Optional[np.ndarray]


def _select_providers(preferred_device: Optional[str] = None) -> List[str]:
    """
    选择 ONNX Runtime providers。

    策略是优先尝试更快的执行后端，但始终保留 CPU 作为最终兜底，
    避免部署环境差异导致模型无法加载。
    """
    available = set(ort.get_available_providers())
    providers: List[str] = []

    preferred = (preferred_device or "").lower()
    if preferred in {"cuda", "gpu"} and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    elif preferred == "mps":
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        elif "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

    if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in providers:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")
    return providers


def _clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """将边界框裁剪到图像范围内。"""
    if boxes.size == 0:
        return boxes
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(width - 1, 0))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(height - 1, 0))
    return boxes


def _build_empty_result_row(image_path: str) -> Dict[str, object]:
    """
    构造“未检测到鸟”时的写库占位数据。

    字段名继续沿用英文列名，兼容历史报表和结果浏览器。
    """
    return {
        "filename": os.path.splitext(os.path.basename(image_path))[0],
        "has_bird": "no",
        "confidence": 0.0,
        "head_sharp": "-",
        "left_eye": "-",
        "right_eye": "-",
        "beak": "-",
        "nima_score": "-",
        "rating": -1,
    }


class OnnxYoloSegModel:
    """YOLO-seg ONNX 推理封装。"""

    def __init__(self, model_path: str, preferred_device: Optional[str] = None, imgsz: int = DEFAULT_ONNX_IMGSZ):
        self.model_path = model_path
        self.imgsz = int(imgsz)
        self.preferred_device = preferred_device or "cpu"
        self.providers = _select_providers(self.preferred_device)
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: List[str] = []
        self.active_provider = "CPUExecutionProvider"
        self._load_session(self.providers)

    def _load_session(self, providers: Sequence[str]) -> None:
        """
        创建 ONNX Runtime Session，并缓存输入输出节点信息。

        当前导出的 YOLO-seg ONNX 固定为 1 个输入、2 个输出：
        检测结果和 mask proto。
        """
        requested_providers = list(providers)
        self.session = ort.InferenceSession(self.model_path, providers=requested_providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or len(outputs) != 2:
            raise RuntimeError("Unexpected YOLO ONNX IO signature")
        self.input_name = inputs[0].name
        self.output_names = [out.name for out in outputs]
        active = self.session.get_providers()
        if active:
            self.active_provider = active[0]
        print(f"[YOLO ONNX] session provider: {self.active_provider} (requested={requested_providers})")

    def close(self) -> None:
        """释放 session 相关引用，便于显式回收。"""
        self.session = None
        self.input_name = None
        self.output_names = []

    @staticmethod
    def _letterbox(img_bgr: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        按比例缩放并补边到推理尺寸。

        后处理依赖返回的 `ratio` 和 `pad`，将检测框和掩码映射回
        `preprocess_image()` 输出图像的坐标系。
        """
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

        out = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return out, ratio, (dw, dh)

    def _prepare_input(self, image_bgr: np.ndarray, imgsz: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        """
        准备 ORT 输入张量。

        业务层图像是 BGR；模型输入需要 RGB、`float32 / 255`、NCHW。
        同时返回用于后处理的元数据。
        """
        infer_size = int(imgsz or self.imgsz)
        letterboxed, ratio, (dw, dh) = self._letterbox(image_bgr, (infer_size, infer_size))
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        meta = {
            "orig_shape": image_bgr.shape[:2],
            "img_shape": letterboxed.shape[:2],
            "ratio": ratio,
            "pad": (dw, dh),
        }
        return tensor, meta

    @staticmethod
    def _scale_boxes_to_processed_image(boxes_xyxy: np.ndarray, meta: Dict[str, object]) -> np.ndarray:
        """
        将 letterbox 坐标系中的框还原到预处理后图像坐标系。

        这里不会恢复到原始文件尺寸，而是对齐 `preprocess_image()` 之后的图像，
        以兼容后续 `photo_processor` 的缩放链路。
        """
        boxes = boxes_xyxy.copy()
        dw, dh = meta["pad"]
        ratio = float(meta["ratio"])
        out_h, out_w = meta["orig_shape"]

        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= max(ratio, 1e-8)
        return _clip_boxes(boxes, out_w, out_h)

    @staticmethod
    def _decode_masks(
        mask_coeffs: np.ndarray,
        proto: np.ndarray,
        boxes_img_xyxy: np.ndarray,
        meta: Dict[str, object],
    ) -> List[np.ndarray]:
        """
        解码 YOLO-seg 掩码。

        YOLO-seg 的分割结果由 mask coefficient 与 proto 线性组合得到，
        之后再裁框、去 padding，并恢复到业务层图像尺寸。
        """
        mask_dim, mask_h, mask_w = proto.shape
        img_h, img_w = meta["img_shape"]
        out_h, out_w = meta["orig_shape"]
        dw, dh = meta["pad"]

        proto_flat = proto.reshape(mask_dim, -1)
        raw_masks = (1.0 / (1.0 + np.exp(-(mask_coeffs @ proto_flat)))).reshape(-1, mask_h, mask_w)

        masks: List[np.ndarray] = []
        scale_x = mask_w / float(img_w)
        scale_y = mask_h / float(img_h)

        for idx in range(raw_masks.shape[0]):
            current = raw_masks[idx]
            x1, y1, x2, y2 = boxes_img_xyxy[idx]
            px1 = int(max(0, np.floor(x1 * scale_x)))
            py1 = int(max(0, np.floor(y1 * scale_y)))
            px2 = int(min(mask_w, np.ceil(x2 * scale_x)))
            py2 = int(min(mask_h, np.ceil(y2 * scale_y)))

            if px2 <= px1 or py2 <= py1:
                masks.append(np.zeros((out_h, out_w), dtype=np.uint8))
                continue

            crop_mask = np.zeros_like(current, dtype=np.float32)
            crop_mask[py1:py2, px1:px2] = current[py1:py2, px1:px2]

            upsampled = cv2.resize(crop_mask, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            left = int(round(dw - 0.1))
            right = int(round(dw + 0.1))
            top = int(round(dh - 0.1))
            bottom = int(round(dh + 0.1))
            x_start = max(0, left)
            y_start = max(0, top)
            x_end = img_w - max(0, right)
            y_end = img_h - max(0, bottom)
            unpadded = upsampled[y_start:y_end, x_start:x_end]

            if unpadded.size == 0:
                masks.append(np.zeros((out_h, out_w), dtype=np.uint8))
                continue

            restored = cv2.resize(unpadded, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            masks.append(((restored > 0.5).astype(np.uint8)) * 255)

        return masks

    def _run_session(self, tensor: np.ndarray) -> List[np.ndarray]:
        """
        执行 ONNX 推理。

        某些 GPU/CoreML provider 可能在 session 创建成功后仍在实际推理时失败，
        这里回退到 CPU，保证主流程可继续。
        """
        if self.session is None or self.input_name is None:
            raise RuntimeError("YOLO ONNX session is not initialized")
        try:
            return self.session.run(self.output_names, {self.input_name: tensor})
        except Exception:
            if self.active_provider != "CPUExecutionProvider":
                self._load_session(["CPUExecutionProvider"])
                return self.session.run(self.output_names, {self.input_name: tensor})
            raise

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_thres: float = 0.25,
        class_filter: Optional[Iterable[int]] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        """
        对单张图像执行检测。

        导出模型要求 `nms=True`，因此 `output0` 已是 NMS 后结果，
        `output1` 为掩码 proto。
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        tensor, meta = self._prepare_input(image_bgr, imgsz=imgsz)
        outputs = self._run_session(tensor)
        pred = outputs[0][0]
        proto = outputs[1][0]

        if pred.ndim != 2:
            raise RuntimeError(f"Unexpected pred rank: {pred.shape}")
        if pred.shape[0] < pred.shape[1]:
            pred = pred.transpose(1, 0)

        mask_dim = int(proto.shape[0])
        if pred.shape[1] < 6 + mask_dim or (pred.shape[1] - 6) != mask_dim:
            raise RuntimeError(
                f"Unexpected ONNX output format: pred shape={pred.shape}, mask_dim={mask_dim}. "
                "Please re-export with nms=True."
            )

        # nms=True 导出格式为 [x1, y1, x2, y2, score, cls, mask coeff...]
        boxes_xyxy_img = pred[:, :4].astype(np.float32)
        scores = pred[:, 4].astype(np.float32)
        class_ids = np.round(pred[:, 5]).astype(np.int32)
        mask_coeffs = pred[:, 6 : 6 + mask_dim].astype(np.float32)

        # 先按业务阈值和类别过滤，再统一封装成 Detection，复用旧版后处理流程
        keep = scores >= float(conf_thres)
        if class_filter is not None:
            filter_set = {int(cls_id) for cls_id in class_filter}
            keep &= np.array([int(cls_id) in filter_set for cls_id in class_ids], dtype=bool)

        if not np.any(keep):
            return []

        boxes_xyxy_img = boxes_xyxy_img[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
        mask_coeffs = mask_coeffs[keep]

        masks = self._decode_masks(mask_coeffs, proto, boxes_xyxy_img, meta)
        boxes_xyxy_out = self._scale_boxes_to_processed_image(boxes_xyxy_img, meta)

        detections: List[Detection] = []
        for idx in range(boxes_xyxy_out.shape[0]):
            detections.append(
                Detection(
                    box=boxes_xyxy_out[idx].astype(np.float32),
                    conf=float(scores[idx]),
                    cls=int(class_ids[idx]),
                    mask=masks[idx],
                )
            )
        return detections


def load_yolo_model(log_callback=None):
    """
    加载全局 YOLO ONNX 模型实例。

    模块级缓存让 GUI 预加载和 `photo_processor` 处理阶段复用同一个
    ONNX session，避免重复初始化。
    """
    global _MODEL_INSTANCE
    model_path = config.ai.get_model_path()
    try:
        from config import get_best_device
        preferred_device = get_best_device()
    except Exception:
        preferred_device = "cpu"

    with _MODEL_LOCK:
        if _MODEL_INSTANCE is None:
            _MODEL_INSTANCE = OnnxYoloSegModel(str(model_path), preferred_device=preferred_device)
        model = _MODEL_INSTANCE

    try:
        i18n = get_i18n()
        provider = model.active_provider
        if provider == "CUDAExecutionProvider":
            msg = i18n.t("ai.using_cuda")
        elif provider == "CoreMLExecutionProvider":
            msg = i18n.t("ai.using_mps")
        else:
            msg = i18n.t("ai.using_cpu")

        if log_callback:
            log_callback(msg, "info")
        else:
            print(msg)
    except Exception as exc:
        i18n = get_i18n()
        error_msg = i18n.t("ai.device_detection_failed", error=str(exc))
        if log_callback:
            log_callback(error_msg, "warning")
        else:
            print(error_msg)

    return model


def preprocess_image(image_path, target_size=None):
    """
    检测前的图像预处理。

    延续旧版 `ai_model.py` 的约定：按最长边缩放到 `TARGET_IMAGE_SIZE`，
    供后续 bbox、img_dims 和掩码链路共同使用。
    """
    if target_size is None:
        target_size = config.ai.TARGET_IMAGE_SIZE

    # img = cv2.imread(image_path)
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    scale = target_size / max(w, h)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def detect_and_draw_birds(image_path, model, output_path, dir, ui_settings, i18n=None, skip_nima=False, focus_point=None, report_db=None):
    """
    检测鸟并输出调试框。

    保持与旧版 `ai_model.py` 相同的 9 元组返回结构，并继续支持：
    - 多鸟场景下基于 `focus_point` 的选鸟语义
    - `yolo_debug_path` 调试图输出
    - 与后续 `photo_processor` 链路兼容的 bbox / img_dims / bird_mask
    """
    # ui_settings 仍沿用旧版顺序，避免调用协议变化
    ai_confidence = ui_settings[0] / 100
    save_crop = ui_settings[3]

    found_bird = False
    bird_result = False
    # NIMA 已改由 photo_processor 后续链路处理，这里继续返回占位 None
    nima_score = None

    # 保持旧版入口的文件类型检查，避免非 JPG 进入后续流程
    if not config.is_jpg_file(image_path):
        log_message("ERROR: not a jpg file", dir)
        return None

    if not os.path.exists(image_path):
        log_message(f"ERROR: in detect_and_draw_birds, {image_path} not found", dir)
        return None

    total_start = time.time()

    try:
        # 先复用项目现有预处理约定，保证后续坐标链路兼容
        image = preprocess_image(image_path)
    except Exception as exc:
        log_message(f"ERROR: preprocess failed for {image_path}: {exc}", dir)
        return None
    height, width, _ = image.shape

    try:
        # ONNX YOLO 推理输出已是 NMS 后结果，后续只做业务筛选和坐标还原
        detections = model.predict(
            image,
            conf_thres=ai_confidence,
            class_filter=[config.ai.BIRD_CLASS_ID],
            imgsz=DEFAULT_ONNX_IMGSZ,
        )
    except Exception as infer_error:
        t = i18n.t if i18n else get_i18n().t
        log_message(t("ai.ai_inference_failed", error=infer_error), dir)
        if report_db:
            report_db.insert_photo(_build_empty_result_row(image_path))
        return found_bird, bird_result, 0.0, 0.0, None, None, None, None, 0

    # 先收集所有鸟，再统一应用多鸟选择策略
    all_birds = []
    for idx, detection in enumerate(detections):
        if int(detection.cls) != config.ai.BIRD_CLASS_ID:
            continue
        x1, y1, x2, y2 = detection.box
        all_birds.append(
            {
                "idx": idx,
                "conf": float(detection.conf),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
            }
        )

    bird_count = len(all_birds)
    bird_idx = -1

    # 多鸟时尽量保持旧版 ai_model 的选鸟语义：优先 focus_point 命中，否则回退最高置信度
    if bird_count == 1:
        bird_idx = all_birds[0]["idx"]
    elif bird_count > 1 and focus_point is not None:
        fx, fy = focus_point
        fx_px, fy_px = int(fx * width), int(fy * height)
        for bird in all_birds:
            x1, y1, x2, y2 = bird["bbox"]
            if x1 <= fx_px <= x2 and y1 <= fy_px <= y2:
                bird_idx = bird["idx"]
                break
        if bird_idx == -1:
            bird_idx = max(all_birds, key=lambda item: item["conf"])["idx"]
    elif bird_count > 1:
        bird_idx = max(all_birds, key=lambda item: item["conf"])["idx"]

    if bird_idx == -1:
        if report_db:
            report_db.insert_photo(_build_empty_result_row(image_path))
        return found_bird, bird_result, 0.0, 0.0, None, None, None, None, 0

    # 锐度不再在此模块计算，但保留默认值，避免后续引用未定义变量
    sharpness = 0.0
    x, y, w, h = 0, 0, 0, 0
    bird_mask = None

    for idx, detection in enumerate(detections):
        # 只处理已选中的那只鸟，保持旧版后续处理流程
        if idx != bird_idx:
            continue

        x1, y1, x2, y2 = detection.box
        x = int(x1)
        y = int(y1)
        w = int(x2 - x1)
        h = int(y2 - y1)

        if int(detection.cls) != config.ai.BIRD_CLASS_ID:
            continue

        found_bird = True
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            log_message(f"ERROR: Invalid crop region for {image_path}", dir)
            found_bird = False
            continue

        crop_img = image[y:y + h, x:x + w]
        if crop_img is None or crop_img.size == 0:
            log_message(f"ERROR: Crop image is empty for {image_path}", dir)
            found_bird = False
            continue

        # 保留红框调试图，便于人工复核检测结果
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        try:
            rel_current_path = os.path.relpath(image_path, dir)
        except ValueError:
            rel_current_path = image_path

        if save_crop and not output_path:
            from tools.file_utils import ensure_hidden_directory

            superpicky_dir = os.path.join(dir, ".superpicky")
            cache_dir = os.path.join(superpicky_dir, "cache")
            # 调试目录统一命名为 yolo_debug，和数据库字段保持一致
            debug_dir = os.path.join(cache_dir, "yolo_debug")
            try:
                ensure_hidden_directory(superpicky_dir)
                ensure_hidden_directory(debug_dir)
                prefix, _ = os.path.splitext(os.path.basename(image_path))
                output_path = os.path.join(debug_dir, f"{prefix}.jpg")
            except Exception:
                pass

        data = {
            "filename": os.path.splitext(os.path.basename(image_path))[0],
            "has_bird": "yes" if found_bird else "no",
            "confidence": float(f"{detection.conf:.2f}"),
            # 这些字段改由 photo_processor 后续补齐，这里保留占位保证结果结构稳定
            "head_sharp": "-",
            "left_eye": "-",
            "right_eye": "-",
            "beak": "-",
            "nima_score": "-",
            "rating": 0,
            # 路径字段供结果浏览器和调试链路复用
            "current_path": rel_current_path,
            "debug_crop_path": None,
            "yolo_debug_path": None,
        }

        if found_bird and save_crop and output_path:
            # 优先保存相对路径；跨平台/跨盘符时回退绝对路径
            try:
                data["yolo_debug_path"] = os.path.relpath(output_path, dir)
            except ValueError:
                data["yolo_debug_path"] = output_path

        if report_db:
            # 写库字段与旧版 ai_model.py 保持一致，让上层无感知切换到 ONNX 后端
            report_db.insert_photo(data)

        if detection.mask is not None:
            # 返回处理后图像尺寸的二值掩码，供 photo_processor 后续再映射回原图
            raw_mask = detection.mask
            if raw_mask.shape != (height, width):
                raw_mask = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            bird_mask = raw_mask.astype(np.uint8)
        break

    # 只有找到鸟且 output_path 有效时才落盘调试图，行为与旧版保持一致
    if found_bird and output_path:
        cv2.imwrite(output_path, image)

    _ = (time.time() - total_start) * 1000

    bird_confidence = float(detections[bird_idx].conf) if bird_idx != -1 else 0.0
    # bbox / img_dims 继续使用“预处理后图像”的坐标系，便于后续缩放换算
    bird_bbox = (x, y, w, h) if found_bird else None
    img_dims = (width, height) if found_bird else None

    return found_bird, bird_result, bird_confidence, sharpness, nima_score, bird_bbox, img_dims, bird_mask, bird_count
