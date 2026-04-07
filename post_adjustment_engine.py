#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky V3.3 - Re-Star Engine
后期评分调整引擎 - 基于已有CSV数据重新计算星级评分
完全重写版本，增强数据处理的健壮性
"""

import os
from typing import List, Dict, Set, Optional, Tuple
from constants import RAW_EXTENSIONS, JPG_EXTENSIONS, IMAGE_EXTENSIONS
from tools.i18n import t
from tools.report_db import ReportDB


def safe_float(value, default=0.0) -> float:
    """
    安全地将值转换为浮点数
    
    Args:
        value: 要转换的值
        default: 如果转换失败时的默认值
        
    Returns:
        浮点数
    """
    if value is None or value == '' or value == '-':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0) -> int:
    """
    安全地将值转换为整数
    
    Args:
        value: 要转换的值
        default: 如果转换失败时的默认值
        
    Returns:
        整数
    """
    if value is None or value == '' or value == '-':
        return default
    try:
        return int(float(value))  # 支持 "3.0" 这种格式
    except (ValueError, TypeError):
        return default


class PostAdjustmentEngine:
    """后期评分调整引擎"""

    def __init__(self, directory: str):
        """
        初始化引擎

        Args:
            directory: 照片目录路径
        """
        self.directory = directory
        self.report_db = None
        self.photos_data: List[Dict] = []
        self.image_extensions = IMAGE_EXTENSIONS

    def load_report(self) -> Tuple[bool, str]:
        """
        加载 report.db

        Returns:
            (成功标志, 错误消息或成功消息)
        """
        db_path = os.path.join(self.directory, ".superpicky", "report.db")
        if not os.path.exists(db_path):
            return False, t("engine.report_not_found", path=db_path)

        try:
            self.report_db = ReportDB(self.directory)
            all_photos = self.report_db.get_all_photos()

            # 只加载有鸟的照片
            self.photos_data = [
                photo for photo in all_photos
                if photo.get('has_bird') == 1
            ]

            total_count = len(all_photos)
            bird_count = len(self.photos_data)

            return True, t("engine.load_success", bird=bird_count, total=total_count)

        except Exception as e:
            return False, t("engine.csv_read_failed", error=str(e))

    def find_image_file(self, filename_without_ext: str) -> Optional[str]:
        """
        根据文件名（无扩展名）查找实际图片文件，支持递归搜索子目录

        Args:
            filename_without_ext: 不含扩展名的文件名

        Returns:
            完整文件路径，或None（如果文件不存在）
        """
        # 优先级：RAW > JPG > DNG
        raw_priority = [ext.lower() for ext in RAW_EXTENSIONS if ext.lower() not in ['.dng']]
        raw_priority += [ext.upper() for ext in RAW_EXTENSIONS if ext.lower() not in ['.dng']]
        secondary_extensions = [ext.lower() for ext in JPG_EXTENSIONS] + [ext.upper() for ext in JPG_EXTENSIONS]
        tertiary_extensions = ['.dng', '.DNG']

        all_extensions = raw_priority + secondary_extensions + tertiary_extensions

        # 先在根目录查找
        for ext in all_extensions:
            file_path = os.path.join(self.directory, filename_without_ext + ext)
            if os.path.exists(file_path):
                return file_path

        # 如果根目录找不到，递归搜索子目录
        for root, dirs, files in os.walk(self.directory):
            for ext in all_extensions:
                target_filename = filename_without_ext + ext
                if target_filename in files:
                    return os.path.join(root, target_filename)

        return None

    def recalculate_ratings(
        self,
        photos: List[Dict],
        min_confidence: float,
        min_sharpness: float,
        min_nima: float,
        sharpness_threshold: float,
        nima_threshold: float
    ) -> List[Dict]:
        """
        根据新阈值重新计算所有照片的星级

        Args:
            photos: 照片数据列表
            min_confidence: 0星阈值 - 置信度
            min_sharpness: 0星阈值 - 锐度
            min_nima: 0星阈值 - 美学
            sharpness_threshold: 2/3星阈值 - 锐度
            nima_threshold: 2/3星阈值 - 美学

        Returns:
            新的照片数据列表（含新星级）
        """
        new_photos = []

        for photo in photos:
            # V4.1: 使用调整后的锐度和美学（如果存在），否则使用原始值
            # 调整后的值包含对焦权重和飞鸟加成，确保重新评星与原始处理一致
            conf = safe_float(photo.get('confidence'), 0.0)
            
            # 优先使用 adj_sharpness，否则使用 head_sharp
            adj_sharpness = safe_float(photo.get('adj_sharpness'), None)
            sharpness = adj_sharpness if adj_sharpness else safe_float(photo.get('head_sharp'), 0.0)
            
            # 优先使用 adj_topiq，否则使用 nima_score
            adj_topiq = safe_float(photo.get('adj_topiq'), None)
            nima_score = adj_topiq if adj_topiq else safe_float(photo.get('nima_score'), None)

            # 判定星级
            # 0星判定（技术质量差）
            if conf < min_confidence or \
               (nima_score is not None and nima_score < min_nima) or \
               sharpness < min_sharpness:
                rating = 0
            # 3星判定（优选：锐度和美学双达标）
            elif sharpness >= sharpness_threshold and \
                 (nima_score is not None and nima_score >= nima_threshold):
                rating = 3
            # 2星判定（良好：锐度或美学达标其一）
            elif sharpness >= sharpness_threshold or \
                 (nima_score is not None and nima_score >= nima_threshold):
                rating = 2
            # 1星（普通）
            else:
                rating = 1

            # 添加新星级到数据
            photo_copy = photo.copy()
            photo_copy['新星级'] = rating
            new_photos.append(photo_copy)

        return new_photos

    def recalculate_picked(
        self,
        star_3_photos: List[Dict],
        picked_percentage: int
    ) -> Set[str]:
        """
        重新计算精选旗标（3星照片的双Top%交集）

        Args:
            star_3_photos: 3星照片列表
            picked_percentage: 精选百分比 (10-50)

        Returns:
            应设置精选旗标的文件名集合（不含扩展名）
        """
        if len(star_3_photos) == 0:
            return set()

        # 计算需要选取的数量（至少1张）
        top_percent = picked_percentage / 100.0
        top_count = max(1, int(len(star_3_photos) * top_percent))

        # 按美学排序，取Top N%
        photos_with_nima = [
            p for p in star_3_photos
            if safe_float(p.get('nima_score'), None) is not None
        ]

        if len(photos_with_nima) == 0:
            return set()

        sorted_by_nima = sorted(
            photos_with_nima,
            key=lambda x: safe_float(x.get('nima_score'), 0.0),
            reverse=True
        )
        nima_top_files = set([photo['filename'] for photo in sorted_by_nima[:top_count]])

        # 按锐度排序，取Top N%（V3.3: 使用新列名 head_sharp）
        photos_with_sharpness = [
            p for p in star_3_photos
            if safe_float(p.get('head_sharp'), 0.0) > 0
        ]
        
        sorted_by_sharpness = sorted(
            photos_with_sharpness,
            key=lambda x: safe_float(x.get('head_sharp'), 0.0),
            reverse=True
        )
        sharpness_top_files = set([photo['filename'] for photo in sorted_by_sharpness[:top_count]])

        # 计算交集（同时在美学和锐度Top N%中的照片）
        picked_files = nima_top_files & sharpness_top_files

        return picked_files

    def get_statistics(self, photos: List[Dict]) -> Dict[str, int]:
        """
        统计各星级照片数量

        Args:
            photos: 照片数据列表（必须包含'新星级'字段）

        Returns:
            {'star_3': 50, 'star_2': 80, 'star_1': 200, 'star_0': 120, 'total': 450}
        """
        stats = {
            'star_0': 0,
            'star_1': 0,
            'star_2': 0,
            'star_3': 0,
            'total': len(photos)
        }

        for photo in photos:
            rating = safe_int(photo.get('新星级', photo.get('rating', 0)), 0)

            if rating == 0:
                stats['star_0'] += 1
            elif rating == 1:
                stats['star_1'] += 1
            elif rating == 2:
                stats['star_2'] += 1
            elif rating == 3:
                stats['star_3'] += 1

        return stats

    def update_report_csv(self, updated_photos: List[Dict], picked_files: set) -> Tuple[bool, str]:
        """
        更新 report.db 中的评分数据

        Args:
            updated_photos: 更新后的照片数据（包含 '新星级' 字段）
            picked_files: 被标记为精选的文件名集合

        Returns:
            (成功标志, 消息)
        """
        if self.report_db is None:
            return False, "Database not loaded"
        
        try:
            updates = []
            for photo in updated_photos:
                filename = photo.get('filename')
                new_rating = photo.get('新星级', 0)
                if filename:
                    updates.append({
                        'filename': filename,
                        'rating': int(new_rating)
                    })
            
            updated_count = self.report_db.update_ratings_batch(updates)
            return True, t("engine.csv_update_success", count=updated_count)

        except Exception as e:
            return False, t("engine.csv_update_failed", error=str(e))
