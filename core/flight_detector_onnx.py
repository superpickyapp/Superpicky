#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX 飞行检测模块。

作为 `core.flight_detector` 的 ONNX 替代实现，使用
`superFlier_efficientnet.onnx` 配合 onnxruntime 执行推理。
"""

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import onnxruntime as ort
from PIL import Image


def _create_session_with_fallback(model_path: str, providers: List[str]) -> tuple[ort.InferenceSession, List[str]]:
    """
    创建 ONNX Runtime Session，并在 CUDA 初始化失败时回退到 CPU。

    这里保留“优先 GPU、失败后继续可用”的策略，避免部署环境中因为
    CUDA Runtime 或打包差异导致整个检测器不可用。
    """
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f"[Flight ONNX] CUDA init failed, fallback to CPU: {exc}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session, session.get_providers()


@dataclass
class FlightResult:
    """飞行检测结果。"""

    is_flying: bool
    confidence: float


class ONNXFlightDetector:
    """
    飞行检测器（ONNX 版）。

    与 PyTorch 版本保持相同的输入输出语义，内部改为通过
    onnxruntime 执行 EfficientNet-B3 二分类模型。
    """

    # 模型配置
    IMAGE_SIZE = 384
    THRESHOLD = 0.5

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化检测器。

        Args:
            model_path: ONNX 模型文件路径；为 None 时自动选择默认路径。
        """
        self.session: Optional[ort.InferenceSession] = None
        self.model_loaded = False
        self.providers: List[str] = []
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

        # 确定模型路径，兼容 PyInstaller 打包与开发环境
        if model_path is None:
            import sys

            if hasattr(sys, "_MEIPASS"):
                self.model_path = Path(sys._MEIPASS) / "models" / "superFlier_efficientnet.onnx"
            else:
                project_root = Path(__file__).parent.parent
                self.model_path = project_root / "models" / "superFlier_efficientnet.onnx"
        else:
            self.model_path = Path(model_path)

    @staticmethod
    def _build_providers() -> List[str]:
        """按“CUDA 优先，CPU 保底”的顺序构造 provider 列表。"""
        available = set(ort.get_available_providers())
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _preprocess_single_image(image: Union[np.ndarray, Image.Image, str], image_size: int) -> np.ndarray:
        """
        预处理单张图像。

        与 PyTorch 版本保持相同的输入兼容性和归一化参数，但这里直接用
        numpy/PIL 完成 resize、标准化与 CHW 转换，不依赖 torchvision。
        """
        if isinstance(image, str):
            # 文件路径输入
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.ndim == 3 and arr.shape[2] == 3:
                # 与原版保持一致：numpy 三通道输入默认按 BGR 处理
                arr = arr[:, :, ::-1]
            pil_image = Image.fromarray(arr).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # resize + ToTensor + Normalize 的 ONNX 等价实现
        resized = pil_image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # HWC -> CHW，匹配模型输入布局
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def load_model(self) -> None:
        """
        加载 ONNX 模型并缓存输入输出节点名称。

        `input_name` / `output_name` 由 session 动态读取，避免写死节点名。
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Flight model not found: {self.model_path}")

        # 创建推理会话，并记录实际启用的 providers
        self.providers = self._build_providers()
        try:
            self.session, self.providers = _create_session_with_fallback(str(self.model_path), self.providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as exc:
            raise RuntimeError(f"Failed to load ONNX flight model: {exc}")

        self.model_loaded = True

    def detect(self, image: Union[np.ndarray, Image.Image, str], threshold: float = None) -> FlightResult:
        """
        检测单张图像中的鸟是否处于飞行状态。

        Args:
            image: 输入图像，支持路径、numpy 数组和 PIL.Image。
            threshold: 分类阈值；为 None 时使用默认阈值。
        """
        if not self.model_loaded or self.session is None or self.input_name is None or self.output_name is None:
            raise RuntimeError("Flight model not loaded, call load_model() first")

        if threshold is None:
            threshold = self.THRESHOLD

        # ONNX 输入固定为 NCHW float32，单张图像需补 batch 维
        sample = self._preprocess_single_image(image, self.IMAGE_SIZE)
        batch = np.expand_dims(sample, axis=0).astype(np.float32)

        # 输出为单个概率值，语义与 PyTorch 版本一致
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        prob = float(np.asarray(output).reshape(-1)[0])

        return FlightResult(is_flying=prob > threshold, confidence=prob)

    def detect_batch(self, images: list, threshold: float = None, batch_size: int = 8) -> list:
        """
        批量检测多张图像。

        为保持容错性，单个样本预处理失败时会跳过该样本，继续处理同批其余图像。
        """
        if not self.model_loaded or self.session is None or self.input_name is None or self.output_name is None:
            raise RuntimeError("Flight model not loaded, call load_model() first")

        if threshold is None:
            threshold = self.THRESHOLD

        results: List[FlightResult] = []

        # 分批预处理并推理，避免一次性堆叠过多图像
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            samples = []
            for img in batch:
                try:
                    samples.append(self._preprocess_single_image(img, self.IMAGE_SIZE))
                except Exception:
                    continue

            if not samples:
                continue

            # 堆叠为 ONNX 批输入
            batch_input = np.stack(samples, axis=0).astype(np.float32)
            output = self.session.run([self.output_name], {self.input_name: batch_input})[0]
            probs = np.asarray(output, dtype=np.float32).reshape(-1)

            for prob in probs:
                p = float(prob)
                results.append(FlightResult(is_flying=p > threshold, confidence=p))

        return results


# 全局单例（延迟初始化）
_flight_detector_instance: Optional[ONNXFlightDetector] = None


def get_flight_detector() -> ONNXFlightDetector:
    """获取全局飞行检测器实例。"""
    global _flight_detector_instance
    if _flight_detector_instance is None:
        _flight_detector_instance = ONNXFlightDetector()
    return _flight_detector_instance


def _compare_with_torch(images: List[str], threshold: float, batch_size: int, onnx_detector: ONNXFlightDetector) -> int:
    """
    与 PyTorch 基线实现做结果对比。

    该辅助函数仅用于开发期验证 ONNX 导出结果是否与原模型保持一致，
    不参与业务主流程。
    """
    if not importlib.util.find_spec("torch"):
        print("[Compare] torch unavailable, skip baseline comparison")
        return 0

    try:
        from core.flight_detector import FlightDetector as TorchFlightDetector
    except Exception as exc:
        try:
            import sys

            project_root = Path(__file__).resolve().parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            from core.flight_detector import FlightDetector as TorchFlightDetector
        except Exception:
            print(f"[Compare] failed to import torch detector: {exc}")
            return 0

    torch_detector = TorchFlightDetector()
    torch_detector.load_model()

    # 统计概率误差与分类不一致次数，便于快速评估导出质量
    max_abs_err = 0.0
    mean_abs_err_acc = 0.0
    count = 0
    cls_mismatch = 0

    for path in images:
        onnx_res = onnx_detector.detect(path, threshold=threshold)
        torch_res = torch_detector.detect(path, threshold=threshold)

        err = abs(onnx_res.confidence - torch_res.confidence)
        max_abs_err = max(max_abs_err, err)
        mean_abs_err_acc += err
        count += 1

        if onnx_res.is_flying != torch_res.is_flying:
            cls_mismatch += 1

        print(
            f"[Compare] {path} | "
            f"onnx={onnx_res.confidence:.6f} ({onnx_res.is_flying}) "
            f"torch={torch_res.confidence:.6f} ({torch_res.is_flying}) "
            f"abs_err={err:.6f}"
        )

    if count > 0:
        print("\n[Compare Summary]")
        print(f"samples: {count}")
        print(f"mean_abs_err: {mean_abs_err_acc / count:.6f}")
        print(f"max_abs_err: {max_abs_err:.6f}")
        print(f"class_mismatch: {cls_mismatch}")

    return 0


if __name__ == "__main__":
    # 简单命令行入口：执行 ONNX 推理，并可选对比 PyTorch 基线
    parser = argparse.ArgumentParser(description="ONNX flight detector with optional torch comparison")
    parser.add_argument("images", nargs="+", help="image path(s)")
    parser.add_argument("--threshold", type=float, default=0.5, help="classification threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for detect_batch")
    parser.add_argument("--no-compare", action="store_true", help="skip torch baseline comparison")
    args = parser.parse_args()

    detector = ONNXFlightDetector()
    detector.load_model()
    print(f"ONNX model loaded: {detector.model_path}")
    print(f"Providers: {detector.providers}")

    print("\n[ONNX Single Detect]")
    for img in args.images:
        res = detector.detect(img, threshold=args.threshold)
        print(f"{img} -> confidence={res.confidence:.6f}, is_flying={res.is_flying}")

    print("\n[ONNX Batch Detect]")
    batch_results = detector.detect_batch(args.images, threshold=args.threshold, batch_size=args.batch_size)
    for i, res in enumerate(batch_results):
        print(f"#{i + 1}: confidence={res.confidence:.6f}, is_flying={res.is_flying}")

    if not args.no_compare:
        _compare_with_torch(args.images, args.threshold, args.batch_size, detector)
