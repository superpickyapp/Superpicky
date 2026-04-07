#!/usr/bin/env python3
"""
OSEA ResNet34 鸟类分类器

基于 OSEA 开源模型 (https://github.com/bird-feeder/OSEA)
支持 10,964 种鸟类识别

优化策略 (基于 test_preprocessing.py 实验):
- 中心裁剪预处理 (Resize 256 + CenterCrop 224): 置信度提升 ~15%
- 可选 TTA 模式 (原图 + 水平翻转): 额外提升 ~0.5%，但推理时间翻倍
"""

__version__ = "1.0.0"

import os
import sqlite3
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional, Set
from PIL import Image

def _get_birdid_dir() -> Path:
    """获取 birdid 模块目录"""
    return Path(__file__).parent


def _get_project_root() -> Path:
    """获取项目根目录"""
    return _get_birdid_dir().parent


def _get_resource_path(relative_path: str) -> Path:
    """获取资源路径 (支持 PyInstaller 打包)"""
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = _get_project_root()
    return base / relative_path


# ==================== 设备配置 ====================

DEVICE = ['CUDAExecutionProvider', 'CPUExecutionProvider']


def _create_session_with_fallback(model_path: str, providers):
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f"[OSEA] CUDA init failed, fallback to CPU: {exc}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session

# ==================== OSEA 分类器 ====================

class OSEAONNXClassifier:
    """
    OSEA ResNet34 鸟类分类器

    Attributes:
        model: ResNet34 模型
        bird_info: 物种信息列表 [[cn_name, en_name, scientific_name], ...]，从 bird_reference.sqlite 加载
        transform: 图像预处理 transform
        num_classes: 物种数量 (10964)
    """

    DEFAULT_MODEL_PATH = "models/model20240824.onnx"
    DEFAULT_DB_PATH = "birdid/data/bird_reference.sqlite"

    def __init__(
        self,
        model_path: Optional[str] = None,
        crop_mode: str = "center_crop",
        device: Optional[List] = None,
    ):
        """
        初始化 OSEA 分类器

        Args:
            model_path: 模型文件路径 (默认: models/model20240824.onnx)
            use_center_crop: 是否使用中心裁剪预处理 (推荐: True)
            device: 使用的计算设备 (默认: None)
        """

        self.device = device or DEVICE
        self.crop_mode = crop_mode
        self.model_path = model_path or str(_get_resource_path(self.DEFAULT_MODEL_PATH))
        self.db_path = str(_get_resource_path(self.DEFAULT_DB_PATH))
        self.bird_info = self._load_bird_info()
        self.num_classes = len(self.bird_info)

        # 创建ONNX对话
        self.session = _create_session_with_fallback(self.model_path, self.device)
        print(f"[OSEA] onnx Model loaded: {self.num_classes} species devices: {self.session.get_providers()}")

    def _load_bird_info(self) -> List[List[str]]:
        """从 bird_reference.sqlite 加载物种信息"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件未找到: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                "SELECT model_class_id, chinese_simplified, english_name, scientific_name "
                "FROM BirdCountInfo WHERE model_class_id IS NOT NULL ORDER BY model_class_id"
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        num_classes = 10964
        bird_info: List[List[str]] = [['Unknown', 'Unknown', ''] for _ in range(num_classes)]
        for class_id, cn_name, en_name, scientific_name in rows:
            if 0 <= class_id < num_classes:
                bird_info[class_id] = [
                    cn_name or 'Unknown',
                    en_name or 'Unknown',
                    scientific_name or '',
                ]
        return bird_info

    def _onnx_transform(self, image: Image.Image, mode='baseline'):

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if mode == 'center_crop':
            # 对应CENTER_CROP_TRANSFORM

            # transforms.Resize(256), 短边缩放到256，保持宽高比
            w, h = image.size
            if w < h:
                new_w = 256
                new_h = int(256 * h / w)
            else:
                new_h = 256
                new_w = int(256 * w / h)
            image = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            # transforms.CenterCrop(224), 中心裁剪出224 * 224，(left, top, right, bottom)
            image = image.crop((
                (new_w - 224) // 2, 
                (new_h - 224) // 2, 
                (new_w + 224) // 2, 
                (new_h + 224) // 2
            ))
        elif mode == 'yolo_crop':
            image = image.resize((224, 224), resample=Image.Resampling.LANCZOS)
        else:
            # 对应BASELINE_TRANSFORM

            # transforms.Resize((224, 224)),
            image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        
        # transforms.ToTensor(),转化为numpy
        img_data = np.array(image).astype(np.float32) / 255.0

        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        img_data = (img_data - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        # PIL 转换为 numpy后是(H, W, C)，需要转置为(C, H, W)，再转为(1, C, H, W)
        img_data = np.expand_dims(img_data.transpose((2, 0, 1)), axis=0).astype(np.float32)

        return img_data

    def _onnx_softmax(self, logits: np.ndarray, axis=-1):
        e_x = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def _onnx_topk(self, matrix: np.ndarray, k: int, axis = -1):

        # partition 会把最大的 k 个数放到最后（无序）
        # 我们取负号是为了处理“最大”，并在最后切片
        idx = np.argpartition(-matrix, k, axis=axis)
        topk_indices = np.take(idx, np.arange(k), axis=axis)
        
        # 因为 partition 后的前 k 个依然是无序的，如果需要按顺序排列，还要再排一次
        topk_values = np.take_along_axis(matrix, topk_indices, axis=axis)
        sort_idx = np.argsort(-topk_values, axis=axis)
        
        final_indices = np.take_along_axis(topk_indices, sort_idx, axis=axis)
        final_values = np.take_along_axis(topk_values, sort_idx, axis=axis)
        
        return final_values, final_indices

    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        temperature: float = 1.0,
        ebird_species_set: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        预测鸟类物种

        Args:
            image: PIL Image 对象 (RGB)
            top_k: 返回前 K 个结果
            temperature: softmax 温度参数 (1.0 为标准, <1 更尖锐, >1 更平滑)
            ebird_species_set: eBird 物种代码集合 (用于过滤)

        Returns:
            识别结果列表 [{cn_name, en_name, scientific_name, confidence, class_id}, ...]
        """

        # [1, 3, 224, 224]
        input_tensor = self._onnx_transform(image, mode=self.crop_mode)

        # [batch_size, num_classes]
        output = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})[0]
        output = output[:, :self.num_classes]

        # [batch_size, top_k_classes]
        probs = self._onnx_softmax(output / temperature, axis=-1)
        k = min(100 if ebird_species_set else top_k, self.num_classes)
        top_probs, top_indices = self._onnx_topk(probs, k, axis=-1)

        results = []
        for single_probs, single_indices in zip(top_probs, top_indices):
            
            for i in range(len(single_probs)):

                class_id = single_indices[i].item()
                confidence = single_probs[i].item() * 100

                min_confidence = 0.3 if ebird_species_set else 1.0
                if confidence < min_confidence:
                    continue

                info = self.bird_info[class_id]
                cn_name = info[0]
                en_name = info[1]
                scientific_name = info[2] if len(info) > 2 else None

                ebird_match = False

                results.append({
                    'class_id': class_id,
                    'cn_name': cn_name,
                    'en_name': en_name,
                    'scientific_name': scientific_name,
                    'confidence': confidence,
                    'ebird_match': ebird_match,
                })

                if len(results) >= top_k:
                    break

        return results

    def predict_with_tta(
        self,
        image: Image.Image,
        top_k: int = 5,
        temperature: float = 1.0,
        ebird_species_set: Optional[Set[str]] = None,
    ) -> List[Dict]:
        """
        使用 TTA (Test-Time Augmentation) 预测

        TTA 策略: 原图 + 水平翻转取平均
        推理时间翻倍，但可能提高准确率
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        fliped = image.transpose(Image.FLIP_LEFT_RIGHT)


        if self.crop_mode == 'center_crop':
            input1 = self._onnx_transform(image, mode='center_crop')
            input2 = self._onnx_transform(fliped, mode='center_crop')
        else:
            input1 = self._onnx_transform(image, mode='baseline')
            input2 = self._onnx_transform(fliped, mode='baseline')

        output1 = self.session.run(None, {self.session.get_inputs()[0].name: input1})[0]
        output1 = output1[:, :self.num_classes]

        output2 = self.session.run(None, {self.session.get_inputs()[0].name: input2})[0]
        output2 = output2[:, :self.num_classes]

        avg_output = (output1 + output2) / 2
        probs = self._onnx_softmax(avg_output / temperature)

        k = min(100 if ebird_species_set else top_k, self.num_classes)
        top_probs, top_indices = self._onnx_topk(probs, k)

        results = []
        for single_probs, single_indices in zip(top_probs, top_indices):
            
            for i in range(len(single_probs)):

                class_id = single_indices[i].item()
                confidence = single_probs[i].item() * 100

                min_confidence = 0.3 if ebird_species_set else 1.0
                if confidence < min_confidence:
                    continue

                info = self.bird_info[class_id]
                cn_name = info[0]
                en_name = info[1]
                scientific_name = info[2] if len(info) > 2 else None

                ebird_match = False

                results.append({
                    'class_id': class_id,
                    'cn_name': cn_name,
                    'en_name': en_name,
                    'scientific_name': scientific_name,
                    'confidence': confidence,
                    'ebird_match': ebird_match,
                })

                if len(results) >= top_k:
                    break

        return results


# ==================== 全局单例 ====================

_osea_classifier: Optional[OSEAONNXClassifier] = None


def get_osea_classifier() -> OSEAONNXClassifier:
    """获取 OSEA 分类器单例"""
    global _osea_classifier
    if _osea_classifier is None:
        _osea_classifier = OSEAONNXClassifier()
    return _osea_classifier


# ==================== 便捷函数 ====================

def osea_predict(image: Image.Image, top_k: int = 5) -> List[Dict]:
    """快速 OSEA 预测"""
    classifier = get_osea_classifier()
    return classifier.predict(image, top_k=top_k)


def osea_predict_file(image_path: str, top_k: int = 5) -> List[Dict]:
    """OSEA 预测 (从文件路径)"""
    # from birdid.bird_identifier import load_image
    from birdid.bird_identifier_onnx import load_image
    image = load_image(image_path)
    return osea_predict(image, top_k=top_k)


# ==================== 测试 ====================

if __name__ == "__main__":

    """
        原始分类器
    """

    import argparse

    parser = argparse.ArgumentParser(description="OSEA 鸟类分类器测试")
    parser.add_argument("--image", help="测试图片路径")
    parser.add_argument("--top-k", type=int, default=5, help="返回前 K 个结果")
    parser.add_argument("--tta", action="store_true", help="使用 TTA 模式")
    args = parser.parse_args()

    # from birdid.bird_identifier import load_image
    from birdid.bird_identifier_onnx import load_image
    image = load_image(args.image)
    classifier = OSEAONNXClassifier()

    if args.tta:
        results = classifier.predict_with_tta(image, top_k=args.top_k)
        print(f"\n[OSEA TTA 预测结果] 前 {args.top_k} 名:")
    else:
        results = classifier.predict(image, top_k=args.top_k)
        print(f"\n[OSEA 预测结果] 前 {args.top_k} 名:")

    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['cn_name']} ({r['en_name']})")
        print(f"     学名: {r['scientific_name']}")
        print(f"     置信度: {r['confidence']:.1f}%")

