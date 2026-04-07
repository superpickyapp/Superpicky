"""
ONNX 关键点检测模块。

作为 `core.keypoint_detector` 的 ONNX 替代实现，使用
`models/cub200_keypoint_resnet50.onnx` 配合 onnxruntime 完成推理。
"""

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def _create_session_with_fallback(model_path: str, providers):
    """
    创建 ONNX Runtime Session，并在 GPU 初始化失败时回退到 CPU。

    打包环境下 CUDA 依赖更容易出现兼容性问题，因此这里优先保证可用性。
    """
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f"[Keypoint ONNX] CUDA init failed, fallback to CPU: {exc}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


@dataclass
class KeypointResult:
    """关键点检测结果。"""

    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    beak: Tuple[float, float]
    left_eye_vis: float
    right_eye_vis: float
    beak_vis: float
    both_eyes_hidden: bool
    all_keypoints_hidden: bool
    best_eye_visibility: float
    visible_eye: Optional[str]
    head_sharpness: float


class ONNXKeypointDetector:
    """鸟类关键点检测器（ONNX 版）。"""

    # 默认配置
    IMG_SIZE = 416
    VISIBILITY_THRESHOLD = 0.3
    RADIUS_MULTIPLIER = 1.2
    NO_BEAK_RADIUS_RATIO = 0.15

    @staticmethod
    def _get_default_model_path() -> str:
        """获取默认模型路径，兼容 PyInstaller 打包环境。"""
        import sys

        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, "models", "cub200_keypoint_resnet50.onnx")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "models", "cub200_keypoint_resnet50.onnx")

    def __init__(self, model_path: str = None):
        """
        初始化关键点检测器。

        Args:
            model_path: ONNX 模型路径；为空时使用默认路径。
        """
        self.model_path = model_path or self._get_default_model_path()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.coords_name: Optional[str] = None
        self.vis_name: Optional[str] = None

    @staticmethod
    def _build_providers():
        """构造 provider 列表，优先尝试 CUDA，再回退 CPU。"""
        available = set(ort.get_available_providers())
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def load_model(self):
        """
        加载 ONNX 模型，并缓存输入输出节点名称。

        该模型预期有两个输出：关键点坐标和可见性分数。
        """
        if self.session is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Keypoint model not found: {self.model_path}")

        # 创建推理会话并读取图结构中的节点名，避免写死导出名称
        providers = self._build_providers()
        self.session = _create_session_with_fallback(self.model_path, providers)
        self.input_name = self.session.get_inputs()[0].name

        outputs = self.session.get_outputs()
        if len(outputs) < 2:
            raise RuntimeError("Unexpected ONNX outputs for keypoint model")
        self.coords_name = outputs[0].name
        self.vis_name = outputs[1].name

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """
        图像预处理。

        与 PyTorch 版本保持相同的 resize 和 Normalize 参数，
        但直接输出 ONNX 所需的 `1 x C x H x W` float32 数组。
        """
        img = pil_image.resize((self.IMG_SIZE, self.IMG_SIZE), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def detect(
        self,
        bird_crop: np.ndarray,
        box: Tuple[int, int, int, int] = None,
        seg_mask: np.ndarray = None,
    ) -> Optional[KeypointResult]:
        """
        检测鸟类关键点并计算头部清晰度。

        Args:
            bird_crop: 裁剪后的鸟区域图像，要求为 RGB。
            box: 检测框 `(x, y, w, h)`，在无喙可见时用于回退半径计算。
            seg_mask: 与裁剪图对齐的分割掩码，用于约束头部区域。
        """
        self.load_model()

        if bird_crop is None or bird_crop.size == 0:
            return None

        # 转为 PIL 后按训练期参数做预处理
        pil_crop = Image.fromarray(bird_crop).convert("RGB")
        batch = self._preprocess(pil_crop)

        # 模型输出三个关键点坐标和对应可见性分数
        coords, vis = self.session.run([self.coords_name, self.vis_name], {self.input_name: batch})
        coords = np.asarray(coords)[0]
        vis = np.asarray(vis)[0]

        # 解析标准化坐标
        left_eye = (float(coords[0, 0]), float(coords[0, 1]))
        right_eye = (float(coords[1, 0]), float(coords[1, 1]))
        beak = (float(coords[2, 0]), float(coords[2, 1]))

        left_eye_vis = float(vis[0])
        right_eye_vis = float(vis[1])
        beak_vis = float(vis[2])

        # 根据阈值判断关键点可见性
        left_visible = left_eye_vis >= self.VISIBILITY_THRESHOLD
        right_visible = right_eye_vis >= self.VISIBILITY_THRESHOLD
        beak_visible = beak_vis >= self.VISIBILITY_THRESHOLD

        # 保留与原版一致的派生状态
        both_eyes_hidden = (not left_visible) and (not right_visible)
        all_keypoints_hidden = (not left_visible) and (not right_visible) and (not beak_visible)

        if left_visible and right_visible:
            visible_eye = "both"
        elif left_visible:
            visible_eye = "left"
        elif right_visible:
            visible_eye = "right"
        else:
            visible_eye = None

        # 仅在至少有一只眼可见时计算头部区域清晰度
        head_sharpness = 0.0
        if visible_eye is not None:
            head_sharpness = self._calculate_head_sharpness(
                bird_crop,
                left_eye,
                right_eye,
                beak,
                left_eye_vis,
                right_eye_vis,
                beak_visible,
                box,
                seg_mask,
            )

        # 取双眼中更高的可见性，供上层排序或过滤逻辑使用
        best_eye_visibility = max(left_eye_vis, right_eye_vis)

        return KeypointResult(
            left_eye=left_eye,
            right_eye=right_eye,
            beak=beak,
            left_eye_vis=left_eye_vis,
            right_eye_vis=right_eye_vis,
            beak_vis=beak_vis,
            both_eyes_hidden=both_eyes_hidden,
            all_keypoints_hidden=all_keypoints_hidden,
            best_eye_visibility=best_eye_visibility,
            visible_eye=visible_eye,
            head_sharpness=head_sharpness,
        )

    def _calculate_head_sharpness(
        self,
        bird_crop: np.ndarray,
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float],
        beak: Tuple[float, float],
        left_eye_vis: float,
        right_eye_vis: float,
        beak_visible: bool,
        box: Tuple[int, int, int, int] = None,
        seg_mask: np.ndarray = None,
    ) -> float:
        """
        计算头部区域清晰度。

        核心思路与原版一致：以选中的眼睛为圆心，根据眼喙距离或回退规则估算
        头部圆形区域，再与分割掩码取交集后计算区域锐度。
        """
        h, w = bird_crop.shape[:2]

        # 若双眼都不可见，仍使用置信度较高的那只眼做 fallback，
        # 并在最终分数上施加惩罚，避免侧脸等情况被误判为极高清晰度
        if left_eye_vis < self.VISIBILITY_THRESHOLD and right_eye_vis < self.VISIBILITY_THRESHOLD:
            eye = left_eye if left_eye_vis >= right_eye_vis else right_eye
            eye_px = (int(eye[0] * w), int(eye[1] * h))
            beak_px = (int(beak[0] * w), int(beak[1] * h))
            # ONNX 版这里显式使用传入的 beak_visible，避免引用未定义变量
            if beak_visible:
                radius = int(self._distance(eye_px, beak_px) * self.RADIUS_MULTIPLIER)
            elif box is not None:
                box_size = max(box[2], box[3])
                radius = int(box_size * self.NO_BEAK_RADIUS_RATIO)
            else:
                radius = int(max(w, h) * self.NO_BEAK_RADIUS_RATIO)
            radius = max(10, min(radius, min(w, h) // 2))
            circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(circle_mask, eye_px, radius, 255, -1)
            if seg_mask is not None and seg_mask.shape[:2] == (h, w):
                head_mask = cv2.bitwise_and(circle_mask, seg_mask)
            else:
                head_mask = circle_mask
            return self._calculate_sharpness(bird_crop, head_mask) * 0.8

        # 双眼都可见时，选择距离喙更远的眼，近似代表更完整的头部侧
        if left_eye_vis >= self.VISIBILITY_THRESHOLD and right_eye_vis >= self.VISIBILITY_THRESHOLD:
            left_dist = self._distance(left_eye, beak)
            right_dist = self._distance(right_eye, beak)
            eye = left_eye if left_dist >= right_dist else right_eye
        elif left_eye_vis >= self.VISIBILITY_THRESHOLD:
            eye = left_eye
        else:
            eye = right_eye

        # 归一化坐标转像素坐标
        eye_px = (int(eye[0] * w), int(eye[1] * h))
        beak_px = (int(beak[0] * w), int(beak[1] * h))

        # 优先使用眼喙距离估算半径；喙不可见时回退到检测框或裁剪尺寸比例
        if beak_visible:
            radius = int(self._distance(eye_px, beak_px) * self.RADIUS_MULTIPLIER)
        elif box is not None:
            box_size = max(box[2], box[3])
            radius = int(box_size * self.NO_BEAK_RADIUS_RATIO)
        else:
            radius = int(max(w, h) * self.NO_BEAK_RADIUS_RATIO)

        # 限制半径范围，避免极小或超出图像边界
        radius = max(10, min(radius, min(w, h) // 2))

        # 构造头部圆形掩码
        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, eye_px, radius, 255, -1)

        # 如有分割掩码，则与圆形区域取交集
        if seg_mask is not None and seg_mask.shape[:2] == (h, w):
            head_mask = cv2.bitwise_and(circle_mask, seg_mask)
        else:
            head_mask = circle_mask

        return self._calculate_sharpness(bird_crop, head_mask)

    def _calculate_sharpness(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        计算掩码区域的锐度分数。

        使用 Tenengrad（Sobel 梯度平方和）作为基础指标，再通过对数映射
        将结果压缩到 0-1000 范围，便于与原版评分体系保持一致。
        """
        if mask.sum() == 0:
            return 0.0

        # 转灰度后计算梯度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = gx ** 2 + gy ** 2

        mask_pixels = mask > 0
        if mask_pixels.sum() == 0:
            return 0.0

        # 只统计掩码覆盖区域
        raw_sharpness = float(gradient_magnitude[mask_pixels].mean())

        min_val = 100.0
        max_val = 154016.0

        if raw_sharpness <= min_val:
            return 0.0
        if raw_sharpness >= max_val:
            return 1000.0

        log_val = math.log(raw_sharpness) - math.log(min_val)
        log_max = math.log(max_val) - math.log(min_val)
        return (log_val / log_max) * 1000.0

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点之间的欧氏距离。"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 全局单例（延迟初始化）
_detector_instance = None


def get_keypoint_detector() -> ONNXKeypointDetector:
    """获取全局关键点检测器实例。"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ONNXKeypointDetector()
    return _detector_instance


def _load_crop(path: str) -> np.ndarray:
    """从磁盘读取鸟裁剪图，并转换为 RGB。"""
    # arr = cv2.imread(path)
    arr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _compare_with_torch(image_path: str, threshold: float = 0.3) -> int:
    """
    与 PyTorch 版关键点检测器做数值对比。

    用于验证 ONNX 导出模型在坐标、可见性和清晰度评分上是否与原版接近。
    """
    if not importlib.util.find_spec("torch"):
        print("[Compare] torch unavailable, skip baseline comparison")
        return 0

    try:
        from core.keypoint_detector import KeypointDetector as TorchKeypointDetector
    except Exception:
        import sys

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from core.keypoint_detector import KeypointDetector as TorchKeypointDetector

    crop = _load_crop(image_path)

    onnx_detector = ONNXKeypointDetector()
    onnx_detector.load_model()
    onnx_result = onnx_detector.detect(crop)

    torch_detector = TorchKeypointDetector()
    torch_detector.load_model()
    torch_result = torch_detector.detect(crop)

    if onnx_result is None or torch_result is None:
        print("[Compare] one side returned None")
        return 1

    # 简单欧氏距离，用于衡量关键点坐标误差
    def d2(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    coord_err_left = d2(onnx_result.left_eye, torch_result.left_eye)
    coord_err_right = d2(onnx_result.right_eye, torch_result.right_eye)
    coord_err_beak = d2(onnx_result.beak, torch_result.beak)

    vis_err_left = abs(onnx_result.left_eye_vis - torch_result.left_eye_vis)
    vis_err_right = abs(onnx_result.right_eye_vis - torch_result.right_eye_vis)
    vis_err_beak = abs(onnx_result.beak_vis - torch_result.beak_vis)

    sharpness_err = abs(onnx_result.head_sharpness - torch_result.head_sharpness)

    print("\n[Compare Summary]")
    print(f"coord_err_left: {coord_err_left:.8f}")
    print(f"coord_err_right: {coord_err_right:.8f}")
    print(f"coord_err_beak: {coord_err_beak:.8f}")
    print(f"vis_err_left: {vis_err_left:.8f}")
    print(f"vis_err_right: {vis_err_right:.8f}")
    print(f"vis_err_beak: {vis_err_beak:.8f}")
    print(f"sharpness_err: {sharpness_err:.8f}")
    print(
        "class_flags: "
        f"onnx(all_hidden={onnx_result.all_keypoints_hidden}, both_hidden={onnx_result.both_eyes_hidden}, visible_eye={onnx_result.visible_eye}) "
        f"torch(all_hidden={torch_result.all_keypoints_hidden}, both_hidden={torch_result.both_eyes_hidden}, visible_eye={torch_result.visible_eye})"
    )

    return 0


if __name__ == "__main__":
    # 简单命令行入口：执行 ONNX 推理，并可选与 PyTorch 基线比较
    parser = argparse.ArgumentParser(description="ONNX keypoint detector with optional torch comparison")
    parser.add_argument("image", help="bird crop image path")
    parser.add_argument("--no-compare", action="store_true", help="skip torch baseline comparison")
    args = parser.parse_args()

    crop = _load_crop(args.image)
    detector = ONNXKeypointDetector()
    detector.load_model()

    result = detector.detect(crop)
    print("[ONNX Result]")
    print(result)

    if not args.no_compare:
        raise SystemExit(_compare_with_torch(args.image))
