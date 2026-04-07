# -*- coding: utf-8 -*-
"""
SuperPicky - 重新评星对话框
PySide6 版本 - 极简艺术风格
"""

import os
import json
import shutil
import threading
from datetime import datetime
from typing import Dict, List, Set, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QPushButton, QGroupBox, QTextEdit,
    QMessageBox, QFrame, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

from post_adjustment_engine import PostAdjustmentEngine, safe_int, safe_float
from tools.exiftool_manager import get_exiftool_manager
from advanced_config import get_advanced_config
from tools.i18n import get_i18n
from constants import RATING_FOLDER_NAMES, get_rating_folder_name
from ui.styles import COLORS, FONTS, VALUE_STYLE
from ui.custom_dialogs import StyledMessageBox


class PostAdjustmentDialog(QDialog):
    """重新评星对话框 - 极简艺术风格"""

    # 信号
    progress_updated = Signal(str)
    main_window_log = Signal(str)
    apply_complete = Signal(str)

    def __init__(self, parent, directory: str, current_sharpness: int = 400,
                 current_nima: float = 5.0, on_complete_callback=None, log_callback=None):
        super().__init__(parent)

        self.config = get_advanced_config()
        self.i18n = get_i18n(self.config.language)

        self.directory = directory
        self.on_complete_callback = on_complete_callback
        self.log_callback = log_callback

        if log_callback:
            self.main_window_log.connect(log_callback)

        # 初始化引擎
        self.engine = PostAdjustmentEngine(directory)

        # 阈值变量
        self.min_confidence = int(self.config.min_confidence * 100)
        self.min_sharpness = int(self.config.min_sharpness)
        self.min_nima = int(self.config.min_nima * 10)
        self.sharpness_threshold = current_sharpness
        self.nima_threshold = int(current_nima * 10)  # 默认 5.5 -> 55
        self.picked_percentage = int(self.config.picked_top_percentage)

        # 数据
        self.original_photos: List[Dict] = []
        self.updated_photos: List[Dict] = []
        self.picked_files: Set[str] = set()

        # 统计
        self.current_stats: Optional[Dict] = None
        self.preview_stats: Optional[Dict] = None

        # 防抖定时器
        self._preview_timer = None

        # 信号连接
        self.progress_updated.connect(self._update_progress_label)
        self.apply_complete.connect(self._on_apply_complete)

        self._setup_ui()
        self._load_data()

    def _setup_ui(self):
        """设置 UI"""
        self.setWindowTitle(self.i18n.t("post_adjustment.window_title"))
        self.setMinimumSize(680, 580)
        self.resize(700, 640)
        self.setModal(True)

        # 应用样式
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['bg_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QPushButton {{
                background-color: {COLORS['accent']};
                color: {COLORS['bg_void']};
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
            }}
            QPushButton#secondary {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
            QPushButton#secondary:hover {{
                border-color: {COLORS['text_muted']};
                color: {COLORS['text_primary']};
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {COLORS['bg_input']};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {COLORS['accent']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                width: 16px;
                height: 16px;
                margin: -6px 0;
                background: {COLORS['text_primary']};
                border-radius: 8px;
            }}
            QCheckBox {{
                color: {COLORS['text_secondary']};
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid {COLORS['border']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['accent']};
                border-color: {COLORS['accent']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)

        # 标题
        self._create_header(layout)
        layout.addSpacing(24)

        # 统计对比
        self._create_stats_section(layout)
        layout.addSpacing(20)

        # 阈值调整
        self._create_threshold_section(layout)
        layout.addSpacing(24)

        # 底部按钮
        self._create_button_section(layout)

    def _create_header(self, layout):
        """创建头部"""
        header_layout = QHBoxLayout()

        title = QLabel(self.i18n.t("post_adjustment.header_title"))
        title.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            font-size: 16px;
            font-weight: 600;
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # 进度标签
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 11px;")
        header_layout.addWidget(self.progress_label)

        layout.addLayout(header_layout)

    def _create_stats_section(self, layout):
        """创建统计区域（两列对比）"""
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        # 左列：当前统计
        current_panel = self._create_stats_panel(self.i18n.t("post_adjustment.current").upper(), is_preview=False)
        stats_layout.addWidget(current_panel)

        # 右列：预览
        preview_panel = self._create_stats_panel(self.i18n.t("post_adjustment.preview").upper(), is_preview=True)
        stats_layout.addWidget(preview_panel)

        layout.addLayout(stats_layout)

    def _create_stats_panel(self, title: str, is_preview: bool):
        """创建统计面板"""
        panel = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border-radius: 10px;
            }}
        """)
        panel.setFixedHeight(180)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 14, 16, 14)
        panel_layout.setSpacing(8)

        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 1px;
        """)
        panel_layout.addWidget(title_label)

        # 统计内容
        if is_preview:
            self.preview_stats_text = QLabel("")
            self.preview_stats_text.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 13px;
                font-family: {FONTS['mono']};
                line-height: 1.6;
            """)
            self.preview_stats_text.setWordWrap(True)
            panel_layout.addWidget(self.preview_stats_text, 1)
        else:
            self.current_stats_text = QLabel("")
            self.current_stats_text.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 13px;
                font-family: {FONTS['mono']};
                line-height: 1.6;
            """)
            self.current_stats_text.setWordWrap(True)
            panel_layout.addWidget(self.current_stats_text, 1)

        return panel

    def _create_threshold_section(self, layout):
        """创建阈值调整区域"""
        # 标题
        section_label = QLabel(self.i18n.t("post_adjustment.thresholds").upper())
        section_label.setStyleSheet(f"""
            color: {COLORS['text_tertiary']};
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 1px;
        """)
        layout.addWidget(section_label)
        layout.addSpacing(12)

        # 参数卡片
        params_frame = QFrame()
        params_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_elevated']};
                border-radius: 10px;
            }}
        """)

        params_layout = QVBoxLayout(params_frame)
        params_layout.setContentsMargins(20, 16, 20, 16)
        params_layout.setSpacing(16)

        # 锐度阈值 (200-600)
        self.sharp_slider, self.sharp_label = self._create_slider(
            params_layout,
            self.i18n.t("post_adjustment.sharpness"),
            min_val=200, max_val=600, default=self.sharpness_threshold,
            step=50
        )

        # 美学阈值 (4.0-7.0)
        self.nima_slider, self.nima_label = self._create_slider(
            params_layout,
            self.i18n.t("post_adjustment.aesthetics"),
            min_val=40, max_val=70, default=self.nima_threshold,
            step=1,
            format_func=lambda v: f"{v/10:.1f}"
        )

        # 精选百分比
        self.picked_slider, self.picked_label = self._create_slider(
            params_layout,
            self.i18n.t("post_adjustment.pick_top_percent"),
            min_val=10, max_val=50, default=self.picked_percentage,
            step=5,
            format_func=lambda v: f"{v}%"
        )

        layout.addWidget(params_frame)

    def _create_slider(self, layout, label_text, min_val, max_val, default,
                       step=1, format_func=None):
        """创建滑块"""
        container = QHBoxLayout()
        container.setSpacing(16)

        label = QLabel(label_text)
        label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px; min-width: 80px;")
        container.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setSingleStep(step)
        container.addWidget(slider, 1)

        if format_func is None:
            format_func = lambda v: str(v)

        value_label = QLabel(format_func(default))
        value_label.setStyleSheet(f"""
            color: {COLORS['accent']};
            font-size: 14px;
            font-family: {FONTS['mono']};
            font-weight: 500;
            min-width: 50px;
        """)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        container.addWidget(value_label)

        slider.format_func = format_func

        def on_value_changed(v):
            aligned = round(v / step) * step
            if aligned != v:
                slider.blockSignals(True)
                slider.setValue(aligned)
                slider.blockSignals(False)
                v = aligned
            value_label.setText(format_func(v))
            self._on_threshold_changed()

        slider.valueChanged.connect(on_value_changed)

        layout.addLayout(container)
        return slider, value_label

    def _create_button_section(self, layout):
        """创建按钮区域"""
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.setSpacing(8)

        # 取消
        cancel_btn = QPushButton(self.i18n.t("post_adjustment.cancel"))
        cancel_btn.setObjectName("secondary")
        cancel_btn.setMinimumWidth(100)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        # 应用
        self.apply_btn = QPushButton(self.i18n.t("post_adjustment.apply_changes"))
        self.apply_btn.setMinimumWidth(130)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_new_ratings)
        btn_layout.addWidget(self.apply_btn)

        layout.addLayout(btn_layout)

    def _load_data(self):
        """加载 CSV 数据"""
        success, message = self.engine.load_report()

        if not success:
            StyledMessageBox.critical(self, self.i18n.t("errors.error_title"), message)
            self.reject()
            return

        self.original_photos = self.engine.photos_data.copy()
        print(f"Loaded {len(self.original_photos)} photos")

        self.current_stats = self._get_original_statistics()
        print(f"Current stats = {self.current_stats}")

        self._update_current_stats_display()
        self.apply_btn.setEnabled(True)
        self._on_threshold_changed()

    def _get_original_statistics(self) -> Dict[str, int]:
        """获取原始统计"""
        stats = {
            'star_0': 0, 'star_1': 0, 'star_2': 0, 'star_3': 0,
            'picked': 0, 'total': len(self.original_photos)
        }

        star_3_photos = []

        for photo in self.original_photos:
            rating = safe_int(photo.get('rating', '0'), 0)
            if rating == 0:
                stats['star_0'] += 1
            elif rating == 1:
                stats['star_1'] += 1
            elif rating == 2:
                stats['star_2'] += 1
            elif rating == 3:
                stats['star_3'] += 1
                star_3_photos.append(photo)

        picked_files = self.engine.recalculate_picked(
            star_3_photos, self.picked_percentage
        )
        stats['picked'] = len(picked_files)

        return stats

    def _update_current_stats_display(self):
        """更新当前统计显示"""
        if not self.current_stats:
            return

        stats = self.current_stats
        t = self.i18n.t
        text = t("post_adjustment.total_photos", count=stats['total']) + "\n\n"
        text += f"★★★  {stats['star_3']}\n"
        text += f"  └ {t('post_adjustment.pick_label')}: {stats['picked']}\n"
        text += f"★★   {stats['star_2']}\n"
        text += f"★    {stats['star_1']}\n"
        text += f"0★   {stats['star_0']}"

        self.current_stats_text.setText(text)

    def _on_threshold_changed(self):
        """阈值改变回调（防抖）"""
        if self._preview_timer:
            self._preview_timer.stop()

        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(300)

    @Slot()
    def _update_preview(self):
        """更新预览统计"""
        sharpness_threshold = self.sharp_slider.value()
        nima_threshold = self.nima_slider.value() / 10.0
        picked_percentage = self.picked_slider.value()

        # 使用配置文件中的固定默认值（不再由 UI 滑块控制）
        min_confidence = self.config.min_confidence
        min_sharpness = self.config.min_sharpness
        min_nima = self.config.min_nima

        self.updated_photos = self.engine.recalculate_ratings(
            self.original_photos,
            min_confidence=min_confidence,
            min_sharpness=min_sharpness,
            min_nima=min_nima,
            sharpness_threshold=sharpness_threshold,
            nima_threshold=nima_threshold
        )

        star_3_photos = [p for p in self.updated_photos if p.get('新星级') == 3]
        self.picked_files = self.engine.recalculate_picked(star_3_photos, picked_percentage)

        self.preview_stats = self.engine.get_statistics(self.updated_photos)
        self.preview_stats['picked'] = len(self.picked_files)

        self._update_preview_display()

    def _update_preview_display(self):
        """更新预览显示（带变化量）"""
        if not self.preview_stats or not self.current_stats:
            return

        old = self.current_stats
        new = self.preview_stats

        def format_diff(new_val, old_val):
            diff = new_val - old_val
            if diff == 0:
                return ""
            elif diff > 0:
                return f"  <span style='color: {COLORS['success']};'>[+{diff}]</span>"
            else:
                return f"  <span style='color: {COLORS['error']};'>[{diff}]</span>"

        t = self.i18n.t
        html = t("post_adjustment.total_photos", count=new['total']) + "<br><br>"
        html += f"★★★  {new['star_3']}{format_diff(new['star_3'], old['star_3'])}<br>"
        html += f"  └ {t('post_adjustment.pick_label')}: {new['picked']}{format_diff(new['picked'], old.get('picked', 0))}<br>"
        html += f"★★   {new['star_2']}{format_diff(new['star_2'], old['star_2'])}<br>"
        html += f"★    {new['star_1']}{format_diff(new['star_1'], old['star_1'])}<br>"
        html += f"0★   {new['star_0']}{format_diff(new['star_0'], old['star_0'])}"

        self.preview_stats_text.setText(html)

    @Slot(str)
    def _update_progress_label(self, text):
        """更新进度标签"""
        self.progress_label.setText(text)

    @Slot()
    def _apply_new_ratings(self):
        """应用新评星"""
        if not self.updated_photos:
            StyledMessageBox.warning(
                self,
                self.i18n.t("messages.hint"),
                self.i18n.t("post_adjustment.no_data_warning")
            )
            return

        changed_photos = []
        for photo in self.updated_photos:
            new_rating = photo.get('新星级', 0)
            old_rating = int(photo.get('rating', 0))
            if new_rating != old_rating:
                changed_photos.append(photo)

        if not changed_photos:
            StyledMessageBox.information(
                self,
                self.i18n.t("messages.hint"),
                self.i18n.t("post_adjustment.no_changes")
            )
            return

        msg = self.i18n.t("post_adjustment.confirm_msg", count=len(changed_photos), total=len(self.updated_photos))
        reply = StyledMessageBox.question(
            self,
            self.i18n.t("post_adjustment.confirm_title"),
            msg,
            yes_text=self.i18n.t("labels.yes"),
            no_text=self.i18n.t("labels.no")
        )

        if reply != StyledMessageBox.Yes:
            return

        self.apply_btn.setEnabled(False)

        def process():
            try:
                self._do_apply(changed_photos)
            except Exception as e:
                self.progress_updated.emit(f"Error: {e}")

        threading.Thread(target=process, daemon=True).start()

    def _do_apply(self, changed_photos):
        """执行应用（后台线程）"""
        total = len(changed_photos)
        batch_data = []
        not_found = 0

        def log(msg):
            self.progress_updated.emit(msg)
            self.main_window_log.emit(msg)

        log(self.i18n.t("post_adjustment.starting"))

        for i, photo in enumerate(changed_photos):
            filename = photo['filename']
            file_path = self.engine.find_image_file(filename)

            if not file_path:
                not_found += 1
            else:
                rating = photo.get('新星级', 0)
                pick = 1 if filename in self.picked_files else 0
                batch_data.append({
                    'file': file_path,
                    'rating': rating,
                    'pick': pick
                })

            if (i + 1) % 10 == 0 or i == total - 1:
                self.progress_updated.emit(self.i18n.t("post_adjustment.finding_files", current=i+1, total=total))

        if not batch_data:
            log(self.i18n.t("post_adjustment.no_files_found"))
            QTimer.singleShot(0, lambda: self.apply_btn.setEnabled(True))
            return

        log(self.i18n.t("post_adjustment.writing_exif", count=len(batch_data)))
        exiftool_mgr = get_exiftool_manager()
        total_files = len(batch_data)
        batch_size = 20
        success_count = 0
        failed_count = 0

        for i in range(0, total_files, batch_size):
            batch = batch_data[i:i+batch_size]
            current = min(i + batch_size, total_files)
            self.progress_updated.emit(self.i18n.t("post_adjustment.writing_exif_progress", current=current, total=total_files))

            stats = exiftool_mgr.batch_set_metadata(batch)
            success_count += stats['success']
            failed_count += stats['failed']

        log(self.i18n.t("post_adjustment.exif_result", success=success_count, failed=failed_count))

        log(self.i18n.t("post_adjustment.updating_csv"))
        csv_success, csv_msg = self.engine.update_report_csv(
            changed_photos, self.picked_files
        )

        log(self.i18n.t("post_adjustment.reorganizing"))
        moved_count = 0

        for photo in changed_photos:
            filename = photo['filename']
            new_rating = photo.get('新星级', 0)
            old_rating = safe_int(photo.get('rating', '0'), 0)

            if new_rating == old_rating:
                continue

            file_path = self.engine.find_image_file(filename)
            if not file_path:
                continue

            target_folder = get_rating_folder_name(new_rating)
            target_dir = os.path.join(self.directory, target_folder)
            actual_filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, actual_filename)

            if os.path.dirname(file_path) == target_dir:
                continue

            try:
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                if not os.path.exists(target_path):
                    shutil.move(file_path, target_path)
                    moved_count += 1
            except Exception:
                pass

        if moved_count > 0:
            log(self.i18n.t("post_adjustment.moved_files", count=moved_count))

        log(self.i18n.t("post_adjustment.complete"))
        self.progress_updated.emit(self.i18n.t("post_adjustment.complete"))

        if moved_count > 0:
            result_msg = self.i18n.t("post_adjustment.result_with_moved", success=success_count, failed=failed_count, moved=moved_count)
        else:
            result_msg = self.i18n.t("post_adjustment.result_msg", success=success_count, failed=failed_count)
        result_msg += "\n\n" + self.i18n.t("post_adjustment.tip_lightroom")

        self.apply_complete.emit(result_msg)

    @Slot(str)
    def _on_apply_complete(self, result_msg: str):
        """应用完成后显示结果"""
        StyledMessageBox.information(
            self,
            self.i18n.t("post_adjustment.result_title"),
            result_msg
        )
        if self.on_complete_callback:
            self.on_complete_callback()
        self.accept()
