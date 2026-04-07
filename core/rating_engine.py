#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rating Engine - 评分引擎
负责根据 AI 检测结果和关键点检测计算照片评分

职责：
- 接收原始数据（置信度、锐度、TOPIQ美学分数、关键点检测结果）
- 根据配置计算评分和旗标
- 返回评分结果和原因

评分等级（关键点增强版）：
- -1 = 无鸟（排除）
-  0 = 普通（所有关键点不可见 或 最低标准不通过）
-  1 = 普通（通过最低标准但锐度和美学都不达标）
-  2 = 良好（锐度或美学达标）
-  3 = 优选（锐度+美学双达标）

注意：精选旗标(picked) 在所有照片处理完后由 PhotoProcessor 单独计算
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RatingResult:
    """评分结果"""
    rating: int          # -1=无鸟, 0=普通(问题照片), 1=普通(合格), 2=良好, 3=优选
    pick: int            # 0=无旗标, 1=精选, -1=排除
    reason: str          # 评分原因说明
    
    @property
    def star_display(self) -> str:
        """获取星级显示字符串"""
        # 使用全局i18n实例
        from tools.i18n import get_i18n
        i18n = get_i18n()
        
        if self.rating == 3:
            return "⭐⭐⭐"
        elif self.rating == 2:
            return "⭐⭐"
        elif self.rating == 1:
            return "⭐"
        elif self.rating == 0:
            return "0★"  # Simplified for i18n concern, or keep hardcoded if symbols are universal
        else:  # -1
            return "❌"


class RatingEngine:
    """
    评分引擎（关键点增强版 V3.8）
    
    评分规则：
    1. 无鸟 → -1 (Rejected)
    2. 最低标准不通过 → 0 (普通-问题照片)
    3. 所有关键点不可见（双眼+鸟喙都<0.3） → 0 (普通-角度不佳)
    4. 眼睛可见度封顶：
       - best_eye 0.3-0.5: 3星降为2星, 2星降为1星
       - best_eye >= 0.5: 正常评分
    5. 锐度 >= 阈值 AND TOPIQ >= 阈值 → 3星 (优选)
    6. 锐度 >= 阈值 OR TOPIQ >= 阈值 → 2星 (良好)
    7. 通过最低标准但都不达标 → 1星 (普通-合格)
    """
    
    def __init__(
        self,
        # 最低标准阈值（低于此为 0 星）
        min_confidence: float = 0.50,
        min_sharpness: float = 100,    # 头部区域锐度最低阈值
        min_nima: float = 3.5,         # V4.0: 降低美学最低阈值
        # 2/3星达标阈值
        sharpness_threshold: float = 400,  # 头部区域锐度达标阈值（2星和3星共用）
        nima_threshold: float = 5.0,  # TOPIQ 美学达标阈值
    ):
        """
        初始化评分引擎
        
        Args:
            min_confidence: AI 置信度最低阈值 (0-1)
            min_sharpness: 锐度最低阈值
            min_nima: TOPIQ 美学最低阈值 (0-10)
            sharpness_threshold: 锐度达标阈值（2星和3星共用）
            topiq_threshold: TOPIQ 美学达标阈值 (2/3星)，范围 4.0-7.0
        """
        # 最低标准
        self.min_confidence = min_confidence
        self.min_sharpness = min_sharpness
        self.min_nima = min_nima
        
        # 达标标准（2星和3星共用）
        self.sharpness_threshold = sharpness_threshold
        self.nima_threshold = nima_threshold
        
        # 国际化
        from tools.i18n import get_i18n
        self.i18n = get_i18n()
    
    def calculate(
        self,
        detected: bool,
        confidence: float,
        sharpness: float,
        topiq: Optional[float] = None,
        all_keypoints_hidden: bool = False,
        best_eye_visibility: float = 1.0,  # V3.8: 眼睛最高置信度
        is_overexposed: bool = False,      # V3.8: 是否过曝
        is_underexposed: bool = False,     # V3.8: 是否欠曝
        focus_sharpness_weight: float = 1.0,  # V4.0: 对焦锐度权重
        focus_topiq_weight: float = 1.0,      # V4.0: 对焦美学权重
        is_flying: bool = False,              # V4.0: 是否飞鸟（用于乘法加成）
    ) -> RatingResult:
        """
        计算评分
        
        Args:
            detected: 是否检测到鸟
            confidence: AI 置信度 (0-1)
            sharpness: 归一化锐度值（关键点启用时为头部锐度）
            topiq: TOPIQ 美学评分 (0-10)，可选
            all_keypoints_hidden: 所有关键点是否都不可见（双眼+鸟喙）
            best_eye_visibility: 双眼中较高的置信度，用于封顶逻辑
            is_overexposed: 是否过曝（V3.8）
            is_underexposed: 是否欠曝（V3.8）
            focus_sharpness_weight: 对焦锐度权重（V4.0）
                - 1.1: 对焦在头部圈内
                - 1.0: 对焦在SEG鸟身内
                - 0.7: 对焦在BBox内
                - 0.5: 对焦在BBox外
            focus_topiq_weight: 对焦美学权重（V4.0）
                - 1.0: 头部/SEG
                - 0.9: BBox内
                - 0.8: BBox外
            is_flying: 是否飞鸟（V4.0 乘法加成: 锐度×1.2, 美学×1.1）
            
        Returns:
            RatingResult 包含评分、旗标和原因
        """
        # 第一步：无鸟检查
        if not detected:
            return RatingResult(
                rating=-1,
                pick=-1,
                reason=self.i18n.t("rating_engine.reject_no_bird")
            )
        
        # 第二步：置信度检查（低于阈值给 0星）
        if confidence < self.min_confidence:
            return RatingResult(
                rating=0,
                pick=0,
                reason=self.i18n.t("rating_engine.low_confidence", confidence=confidence, threshold=self.min_confidence)
            )
        # 第三步：关键点可见性检查（V4.0: 先判定眼睛）
        # 如果看不到眼睛/嘴巴，直接给 1 星，不再判断美学
        if all_keypoints_hidden:
            return RatingResult(
                rating=1,
                pick=0,
                reason=self.i18n.t("rating_engine.angle_poor")
            )
        
        # 第四步：锐度检查
        if sharpness < self.min_sharpness:
            return RatingResult(
                rating=0,
                pick=0,
                reason=self.i18n.t("rating_engine.low_sharpness", val=sharpness, threshold=self.min_sharpness)
            )
        
        # 第五步：美学检查（放在眼睛和锐度之后）
        if topiq is not None and topiq < self.min_nima:
            return RatingResult(
                rating=0,
                pick=0,
                reason=self.i18n.t("rating_engine.low_aesthetics", val=topiq, threshold=self.min_nima)
            )
        
        # V4.0: 曝光问题标记
        has_exposure_issue = is_overexposed or is_underexposed
        exposure_suffix = ""
        if is_overexposed and is_underexposed:
            exposure_suffix = self.i18n.t("rating_engine.exposure_issue_both")
        elif is_overexposed:
            exposure_suffix = self.i18n.t("rating_engine.exposure_over")
        elif is_underexposed:
            exposure_suffix = self.i18n.t("rating_engine.exposure_under")
        
        # V4.0: 对焦权重处理 - 先应用对焦权重
        # 锐度权重: 1.1(头部) / 1.0(SEG) / 0.7(BBox) / 0.5(外部)
        # 美学权重: 1.0(头部/SEG) / 0.9(BBox) / 0.8(外部)
        adjusted_sharpness = sharpness * focus_sharpness_weight
        adjusted_topiq = topiq * focus_topiq_weight if topiq is not None else None
        
        # V4.0: 飞鸟乘法加成 - 后应用
        if is_flying:
            adjusted_sharpness = adjusted_sharpness * 1.2
            if adjusted_topiq is not None:
                adjusted_topiq = adjusted_topiq * 1.1
        
        # 设置对焦状态后缀（精焦/合焦/失焦/脱焦 - 全部显示）
        # V4.2: 阈值与 photo_processor.py 保持一致
        focus_suffix = ""
        if focus_sharpness_weight > 1.0:
            focus_suffix = self.i18n.t("rating_engine.focus_best")
        elif focus_sharpness_weight >= 0.9:  # 修复: 0.9 以上为合焦（鸟身对焦）
            focus_suffix = self.i18n.t("rating_engine.focus_good")
        elif focus_sharpness_weight >= 0.7:
            focus_suffix = self.i18n.t("rating_engine.focus_bad")
        else:  # < 0.7
            focus_suffix = self.i18n.t("rating_engine.focus_worst")
        
        # 第五步：基础星级判定（锐度 >= 阈值 AND/OR TOPIQ >= 阈值）
        sharpness_ok = adjusted_sharpness >= self.sharpness_threshold
        topiq_ok = adjusted_topiq is not None and adjusted_topiq >= self.nima_threshold
        
        # 计算基础星级
        if sharpness_ok and topiq_ok:
            base_rating = 3
            base_reason = self.i18n.t("rating_engine.base_double")
        elif sharpness_ok:
            base_rating = 2
            base_reason = self.i18n.t("rating_engine.base_sharp")
        elif topiq_ok:
            base_rating = 2
            base_reason = self.i18n.t("rating_engine.base_topiq")
        else:
            base_rating = 1
            base_reason = self.i18n.t("rating_engine.base_none")
        
        # V4.0: 渐进式眼睛可见度降权
        # visibility_weight = max(0.5, min(1.0, best_eye_visibility * 2))
        # 即: visibility 0.5 时权重 1.0，0.25 时权重 0.5
        visibility_weight = max(0.5, min(1.0, best_eye_visibility * 2))
        rating = round(base_rating * visibility_weight)
        
        # 曝光问题降级
        if has_exposure_issue:
            rating = max(0, rating - 1)
        
        # 构建理由
        rating_names = {
            3: self.i18n.t("rating_engine.rating_excellent"), 
            2: self.i18n.t("rating_engine.rating_good"), 
            1: self.i18n.t("rating_engine.rating_average"), 
            0: self.i18n.t("rating_engine.rating_poor")
        }
        rating_name = rating_names.get(rating, self.i18n.t("rating_engine.rating_average"))
        
        # 可见度降权说明
        visibility_suffix = ""
        if visibility_weight < 1.0:
            visibility_suffix = self.i18n.t("rating_engine.visibility_suffix", val=best_eye_visibility)
        
        # 飞鸟标记
        flying_suffix = self.i18n.t("rating_engine.flying_suffix") if is_flying else ""
        
        return RatingResult(
            rating=rating,
            pick=0,
            reason=self.i18n.t("rating_engine.reason_fmt", 
                rating_name=rating_name,
                base_reason=base_reason,
                exposure=exposure_suffix,
                focus=focus_suffix,
                visibility=visibility_suffix,
                flying=flying_suffix
            )
        )
    
    def update_thresholds(
        self,
        sharpness_threshold: Optional[float] = None,
        nima_threshold: Optional[float] = None,
    ):
        """更新达标阈值（用于 UI 滑块调整）"""
        if sharpness_threshold is not None:
            self.sharpness_threshold = sharpness_threshold
        if nima_threshold is not None:
            self.nima_threshold = nima_threshold


def create_rating_engine_from_config(config) -> RatingEngine:
    """
    从高级配置创建评分引擎
    
    Args:
        config: AdvancedConfig 实例
        
    Returns:
        RatingEngine 实例
    """
    return RatingEngine(
        min_confidence=config.min_confidence,
        min_sharpness=config.min_sharpness,
        min_nima=config.min_nima,
        # 达标阈值（由 UI 滑块覆盖）
        sharpness_threshold=400,  # 锐度达标阈值 (200-600)
        nima_threshold=5.0,       # TOPIQ 美学达标阈值 (4.0-7.0)
    )
