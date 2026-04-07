#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exposure Detector - 曝光检测器
使用 OpenCV 直方图分析检测鸟区域的过曝/欠曝

V3.8 新增功能
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExposureResult:
    """曝光检测结果"""
    is_overexposed: bool = False      # 是否过曝
    is_underexposed: bool = False     # 是否欠曝
    overexposed_ratio: float = 0.0    # 过曝像素占比 (0-1)
    underexposed_ratio: float = 0.0   # 欠曝像素占比 (0-1)
    
    @property
    def has_exposure_issue(self) -> bool:
        """是否有曝光问题"""
        return self.is_overexposed or self.is_underexposed
    
    @property
    def issue_description(self) -> str:
        """获取曝光问题描述"""
        if self.is_overexposed and self.is_underexposed:
            return f"过曝+欠曝"
        elif self.is_overexposed:
            return f"过曝({self.overexposed_ratio:.0%})"
        elif self.is_underexposed:
            return f"欠曝({self.underexposed_ratio:.0%})"
        return ""


class ExposureDetector:
    """
    曝光检测器
    
    使用 OpenCV 分析直方图，检测极端过曝和欠曝
    
    过曝判定：亮度 >= 250 的像素占比超过阈值
    欠曝判定：亮度 <= 5 的像素占比超过阈值
    """
    
    def __init__(
        self,
        overexpose_threshold: float = 0.10,  # 过曝阈值 10%
        underexpose_threshold: float = 0.10,  # 欠曝阈值 10%
        bright_cutoff: int = 235,  # 过曝亮度阈值
        dark_cutoff: int = 15       # 欠曝亮度阈值
    ):
        """
        初始化曝光检测器
        
        Args:
            overexpose_threshold: 过曝像素占比阈值 (0-1)
            underexpose_threshold: 欠曝像素占比阈值 (0-1)
            bright_cutoff: 判定为过曝的亮度值 (0-255)
            dark_cutoff: 判定为欠曝的亮度值 (0-255)
        """
        self.overexpose_threshold = overexpose_threshold
        self.underexpose_threshold = underexpose_threshold
        self.bright_cutoff = bright_cutoff
        self.dark_cutoff = dark_cutoff
    
    def detect(
        self, 
        image_bgr: np.ndarray, 
        threshold: Optional[float] = None
    ) -> ExposureResult:
        """
        检测图像曝光情况
        
        Args:
            image_bgr: BGR 格式的图像（通常是鸟的裁剪区域）
            threshold: 可选，覆盖默认阈值（过曝和欠曝共用）
            
        Returns:
            ExposureResult 包含曝光检测结果
        """
        if image_bgr is None or image_bgr.size == 0:
            return ExposureResult()
        
        # 使用传入阈值或默认阈值
        over_thresh = threshold if threshold is not None else self.overexpose_threshold
        under_thresh = threshold if threshold is not None else self.underexpose_threshold
        
        # 转换为灰度图（或使用亮度通道）
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        
        total_pixels = gray.size
        if total_pixels == 0:
            return ExposureResult()
        
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # 过曝：亮度 >= bright_cutoff (250) 的像素占比
        overexposed_pixels = np.sum(hist[self.bright_cutoff:]) / total_pixels
        
        # 欠曝：亮度 <= dark_cutoff (5) 的像素占比
        underexposed_pixels = np.sum(hist[:self.dark_cutoff + 1]) / total_pixels
        
        # 判定
        is_overexposed = overexposed_pixels > over_thresh
        is_underexposed = underexposed_pixels > under_thresh
        
        return ExposureResult(
            is_overexposed=is_overexposed,
            is_underexposed=is_underexposed,
            overexposed_ratio=float(overexposed_pixels),
            underexposed_ratio=float(underexposed_pixels)
        )


# 全局实例（单例模式）
_detector_instance: Optional[ExposureDetector] = None


def get_exposure_detector() -> ExposureDetector:
    """获取曝光检测器实例（单例）"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ExposureDetector()
    return _detector_instance
