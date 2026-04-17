# -*- coding: utf-8 -*-
"""
SuperPicky - 首次启动欢迎向导 + 初始化流程
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, cast

from PySide6.QtCore import Qt, QObject, QTimer, Signal, QEasingCurve
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from advanced_config import get_advanced_config
from core.initialization_manager import InitializationManager
from ui.custom_dialogs import StyledMessageBox
from ui.skill_level_dialog import SkillLevelCard
from ui.styles import COLORS, FONTS

UPDATE_OPTION_KEYS = ("enabled", "disabled")
SKILL_LEVEL_KEYS = ("beginner", "intermediate", "master")
FULL_FEATURE_SET = ("core_detection", "quality", "keypoint", "flight", "birdid")

SELECTABLE_CARD_TITLE_STYLE = f"""
    color: {COLORS['text_primary']};
    font-size: 15px;
    font-weight: 600;
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
"""

SELECTABLE_CARD_DESC_STYLE = f"""
    color: {COLORS['text_secondary']};
    font-size: 12px;
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
"""

SELECTABLE_CARD_SELECTED_STYLE = f"""
    QFrame#updateOptionCard {{
        background-color: {COLORS['bg_elevated']};
        border: 2px solid {COLORS['accent']};
        border-radius: 8px;
    }}
"""

SELECTABLE_CARD_UNSELECTED_STYLE = f"""
    QFrame#updateOptionCard {{
        background-color: {COLORS['bg_elevated']};
        border: 1px solid transparent;
        border-radius: 8px;
    }}
    QFrame#updateOptionCard:hover {{
        border-color: {COLORS['border']};
    }}
"""

DIALOG_STYLE = f"""
    QDialog {{
        background-color: {COLORS['bg_primary']};
        border-radius: 14px;
    }}
    QLabel {{
        color: {COLORS['text_primary']};
        background: transparent;
        font-family: {FONTS['sans']};
    }}
    QPushButton {{
        background-color: {COLORS['accent']};
        color: {COLORS['bg_void']};
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: 600;
        font-family: {FONTS['sans']};
    }}
    QPushButton:hover {{
        background-color: {COLORS['accent_hover']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['accent_pressed']};
    }}
    QPushButton#secondary {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
    }}
    QPushButton#secondary:hover {{
        background-color: {COLORS['bg_elevated']};
        color: {COLORS['text_primary']};
        border-color: {COLORS['text_tertiary']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_muted']};
        border: 1px solid {COLORS['border_subtle']};
    }}
    QCheckBox {{
        color: {COLORS['text_primary']};
        font-size: 13px;
        spacing: 8px;
    }}
    QTextEdit {{
        background-color: {COLORS['bg_card']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 8px;
        font-family: {FONTS['sans']};
        font-size: 12px;
    }}
"""

PAGE_TITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['text_primary']};
        font-size: 24px;
        font-weight: 700;
    }}
"""

BODY_SUBTITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 13px;
    }}
"""

HINT_STYLE = f"""
    QLabel {{
        color: {COLORS['text_tertiary']};
        font-size: 12px;
    }}
"""

DOT_ACTIVE_STYLE = f"background-color: {COLORS['accent']}; border-radius: 5px;"
DOT_INACTIVE_STYLE = f"background-color: {COLORS['border']}; border-radius: 5px;"
ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
POINTING_HAND_CURSOR = Qt.CursorShape.PointingHandCursor


class _SelectableCardLike(Protocol):
    def set_selected(self, selected: bool) -> None:
        ...


class _PostInitializationFlowHost(Protocol):
    def _resume_post_initialization_flow(self) -> None:
        ...


@dataclass(frozen=True)
class _NavState:
    prev_enabled: bool
    next_text: str
    next_enabled: bool
    background_visible: bool
    retry_visible: bool


class SelectableCard(QFrame):
    clicked = Signal(str)

    def __init__(self, option_key: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self.option_key = option_key
        self._selected = False
        self.setObjectName("updateOptionCard")
        self.setCursor(POINTING_HAND_CURSOR)
        self.setFixedSize(260, 150)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        layout.setAlignment(ALIGN_CENTER)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(ALIGN_CENTER)
        self.title_label.setStyleSheet(SELECTABLE_CARD_TITLE_STYLE)
        layout.addWidget(self.title_label)

        self.desc_label = QLabel(description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(ALIGN_CENTER)
        self.desc_label.setStyleSheet(SELECTABLE_CARD_DESC_STYLE)
        layout.addWidget(self.desc_label)
        self._apply_style()

    def set_selected(self, selected: bool):
        if self._selected == selected:
            return
        self._selected = selected
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            SELECTABLE_CARD_SELECTED_STYLE if self._selected else SELECTABLE_CARD_UNSELECTED_STYLE
        )

    def mousePressEvent(self, event):
        self.clicked.emit(self.option_key)
        super().mousePressEvent(event)


class LockedFeatureCheckBox(QCheckBox):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setChecked(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def nextCheckState(self) -> None:
        return

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def keyPressEvent(self, event) -> None:
        event.accept()


class StatusBulletLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setStyleSheet(
            f"""
            color: {COLORS['text_primary']};
            font-size: 13px;
            font-weight: 600;
            background: transparent;
            padding-left: 4px;
            """
        )


class RoundedProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 100
        self._value = 0
        self.setMinimumHeight(14)

    def setRange(self, minimum: int, maximum: int) -> None:
        self._minimum = minimum
        self._maximum = max(minimum + 1, maximum)
        self.update()

    def setValue(self, value: int) -> None:
        bounded = max(self._minimum, min(self._maximum, value))
        if bounded == self._value:
            return
        self._value = bounded
        self.update()

    def setTextVisible(self, _visible: bool) -> None:
        # Kept for compatibility with the previous QProgressBar calls.
        return

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = self.rect().adjusted(0, 0, -1, -1)
        radius = min(rect.height() / 2.0, 8.0)

        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["bg_card"]))
        painter.drawRoundedRect(rect, radius, radius)

        span = max(1, self._maximum - self._minimum)
        progress_ratio = (self._value - self._minimum) / span
        if progress_ratio <= 0:
            return

        fill_width = max(rect.height(), int(rect.width() * progress_ratio))
        fill_width = min(rect.width(), fill_width)
        fill_rect = rect.adjusted(1, 1, -(rect.width() - fill_width), -1)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["accent"]))
        painter.drawRoundedRect(fill_rect, radius - 1, radius - 1)


class InitializationProgressBinder(QObject):
    RUNTIME_PHASE_MAX = 30
    DOWNLOAD_PHASE_MAX = 100
    RUNTIME_SIM_DURATION_SECONDS = 60.0
    DOWNLOAD_SIM_DURATION_SECONDS = 36.0
    MIN_VISIBLE_PROGRESS = 4

    def __init__(
        self,
        manager: InitializationManager,
        *,
        set_stage_text: Callable[[str], None],
        set_progress_value: Callable[[int], None],
        append_log: Callable[[str], None],
        on_success: Callable[[object], None],
        on_failure: Callable[[object], None],
        parent=None,
    ):
        super().__init__(parent)
        self._set_stage_text = set_stage_text
        self._set_progress_value = set_progress_value
        self._append_log = append_log
        self._on_success = on_success
        self._on_failure = on_failure
        self._display_progress = 0
        self._runtime_phase_active = False
        self._download_phase_active = False
        self._download_actual_progress = 0
        self._runtime_phase_started_at = 0.0
        self._download_phase_started_at = 0.0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(80)
        self._progress_timer.timeout.connect(self._advance_progress_animation)
        self._runtime_curve = QEasingCurve(QEasingCurve.Type.OutCubic)
        self._download_curve = QEasingCurve(QEasingCurve.Type.InOutCubic)
        manager.stage_changed.connect(self._handle_stage_changed)
        manager.progress_changed.connect(self._handle_progress_changed)
        manager.item_status_changed.connect(self._handle_item_status_changed)
        manager.finished.connect(self._handle_finished)

    def reset(self) -> None:
        self._runtime_phase_active = False
        self._download_phase_active = False
        self._download_actual_progress = 0
        self._runtime_phase_started_at = 0.0
        self._download_phase_started_at = 0.0
        self._display_progress = 0
        self._progress_timer.stop()
        self._push_progress(0)

    def _push_progress(self, value: int) -> None:
        normalized = max(0, min(100, value))
        if 0 < normalized < self.MIN_VISIBLE_PROGRESS:
            normalized = self.MIN_VISIBLE_PROGRESS
        self._display_progress = normalized
        self._set_progress_value(normalized)

    def _start_runtime_phase(self) -> None:
        self._runtime_phase_active = True
        self._download_phase_active = False
        self._runtime_phase_started_at = time.monotonic()
        self._push_progress(0)
        self._progress_timer.start()

    def _start_download_phase(self) -> None:
        if self._download_phase_active:
            return
        self._runtime_phase_active = False
        self._download_phase_active = True
        self._download_actual_progress = max(0, self._download_actual_progress)
        self._download_phase_started_at = time.monotonic()
        self._push_progress(max(self._display_progress, self.RUNTIME_PHASE_MAX))
        self._progress_timer.start()

    def _stop_progress_animation(self) -> None:
        self._runtime_phase_active = False
        self._download_phase_active = False
        self._progress_timer.stop()

    def _handle_stage_changed(self, stage: str, message: str) -> None:
        self._set_stage_text(message)
        self._append_log(f"[{stage}] {message}")
        if stage == "preparing_runtime":
            self._start_runtime_phase()
        elif stage == "downloading_resources":
            self._start_download_phase()
        elif stage in {"verifying", "ready", "failed"}:
            self._stop_progress_animation()

    def _handle_progress_changed(self, percent: int, _current_item: str, _done: int, _total: int) -> None:
        self._start_download_phase()
        self._download_actual_progress = max(0, min(100, percent))
        self._advance_progress_animation()

    def _handle_item_status_changed(self, resource_id: str, status: str, detail: str) -> None:
        if resource_id in {"updates", "runtime"}:
            self._append_log(f"{resource_id}: {detail}")
            return
        self._append_log(f"{resource_id} [{status}] {detail}")

    def _handle_finished(self, success: bool, summary: object) -> None:
        self._stop_progress_animation()
        if success:
            self._push_progress(100)
            self._on_success(summary)
            return
        self._on_failure(summary)

    def _advance_progress_animation(self) -> None:
        if self._runtime_phase_active:
            self._advance_runtime_phase()
            return
        if self._download_phase_active:
            self._advance_download_phase()
            return
        self._progress_timer.stop()

    def _advance_runtime_phase(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._runtime_phase_started_at)
        progress_ratio = min(1.0, elapsed / self.RUNTIME_SIM_DURATION_SECONDS)
        simulated = int(self._runtime_curve.valueForProgress(progress_ratio) * self.RUNTIME_PHASE_MAX)
        self._push_progress(max(self._display_progress, simulated))
        if progress_ratio >= 1.0:
            self._runtime_phase_active = False
            if not self._download_phase_active:
                self._progress_timer.stop()

    def _advance_download_phase(self) -> None:
        elapsed = max(0.0, time.monotonic() - self._download_phase_started_at)
        progress_ratio = min(1.0, elapsed / self.DOWNLOAD_SIM_DURATION_SECONDS)
        simulated_tail = int(self._download_curve.valueForProgress(progress_ratio) * (self.DOWNLOAD_PHASE_MAX - self.RUNTIME_PHASE_MAX))
        simulated = self.RUNTIME_PHASE_MAX + simulated_tail
        actual = self.RUNTIME_PHASE_MAX + int(self._download_actual_progress * 0.7)
        combined = max(self._display_progress, simulated, actual)
        self._push_progress(combined)
        if progress_ratio >= 1.0 and self._download_actual_progress >= 100:
            self._download_phase_active = False
            self._progress_timer.stop()


class EnvironmentRepairDialog(QDialog):
    def __init__(self, i18n, config, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self.config = config
        self.manager = InitializationManager(self)
        self._setup_ui()
        self._progress = InitializationProgressBinder(
            self.manager,
            set_stage_text=self.stage_label.setText,
            set_progress_value=self.progress_bar.setValue,
            append_log=self.log_view.append,
            on_success=self._on_repair_success,
            on_failure=self._on_repair_failure,
            parent=self,
        )
        self.retry_btn.clicked.connect(self.start_repair)

    def _setup_ui(self) -> None:
        self.setWindowTitle(self.i18n.t("repair.window_title"))
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)

        title = QLabel(self.i18n.t("repair.window_title"))
        title.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {COLORS['text_primary']};")
        layout.addWidget(title)

        summary = QLabel(self.i18n.t("repair.summary"))
        summary.setWordWrap(True)
        summary.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 13px;")
        layout.addWidget(summary)

        self.stage_label = QLabel(self.i18n.t("repair.start"))
        self.stage_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(self.stage_label)

        self.progress_bar = RoundedProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.retry_btn = QPushButton(self.i18n.t("repair.retry"))
        self.retry_btn.setObjectName("secondary")
        self.retry_btn.hide()
        btn_row.addWidget(self.retry_btn)

        close_btn = QPushButton(self.i18n.t("update.close"))
        close_btn.setObjectName("secondary")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _repair_options(self) -> dict:
        return {
            "runtime_variant": self.config.selected_runtime_variant or "auto",
            "runtime_install_location": self.config.runtime_install_location_preference,
            "features": list(FULL_FEATURE_SET),
            "auto_update_enabled": self.config.auto_check_updates,
        }

    def start_repair(self) -> None:
        self.retry_btn.hide()
        self._progress.reset()
        self.stage_label.setText(self.i18n.t("repair.running"))
        self.log_view.append(self.i18n.t("repair.log_retry"))
        self.manager.start_repair(self._repair_options())

    def _on_repair_success(self, _summary: object) -> None:
        self.stage_label.setText(self.i18n.t("repair.success"))
        self.log_view.append(f"[done] {self.i18n.t('repair.success')}")
        parent = self.parent()
        if parent is not None and hasattr(parent, "_resume_post_initialization_flow"):
            cast(_PostInitializationFlowHost, parent)._resume_post_initialization_flow()

    def _on_repair_failure(self, summary: object) -> None:
        self.retry_btn.show()
        error_text = (
            summary.get("error", self.i18n.t("repair.failed"))
            if isinstance(summary, dict)
            else self.i18n.t("repair.failed")
        )
        self.stage_label.setText(error_text)
        self.log_view.append(f"[failed] {error_text}")


class WelcomeOnboardingDialog(QDialog):
    onboarding_completed = Signal(str, bool)

    def __init__(self, i18n, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self.config = get_advanced_config()
        self.current_page = 0
        self.selected_level = self.config.skill_level or "intermediate"
        self.auto_update_enabled = True
        self._dots: list[QLabel] = []
        self._skill_cards: dict[str, SkillLevelCard] = {}
        self._update_cards: dict[str, SelectableCard] = {}
        self._feature_boxes: dict[str, LockedFeatureCheckBox] = {}
        self._runtime_status_labels: list[QLabel] = []
        self._initialization_complete = False
        self._initialization_running = False
        self._closing_after_interrupt = False

        self.initialization_manager = InitializationManager(self)
        self.selected_runtime_install_location = (
            self.initialization_manager.choose_runtime_install_location().key
        )

        self.setModal(True)
        self.setWindowTitle(self.i18n.t("onboarding.window_title"))
        self.setFixedSize(640, 520)
        self.setStyleSheet(DIALOG_STYLE)

        self._setup_ui()
        self._progress = InitializationProgressBinder(
            self.initialization_manager,
            set_stage_text=self.stage_label.setText,
            set_progress_value=self.progress_bar.setValue,
            append_log=self.log_view.append,
            on_success=self._on_initialization_succeeded,
            on_failure=self._on_initialization_failed,
            parent=self,
        )
        self._sync_defaults()
        self._set_current_page(0, force=True)

    def get_selected_options(self) -> dict:
        return {
            "skill_level": self.selected_level,
            "auto_update_enabled": self.auto_update_enabled,
            "runtime_variant": self.config.selected_runtime_variant or "auto",
            "runtime_install_location": self.selected_runtime_install_location,
            "features": list(FULL_FEATURE_SET),
        }

    def _create_page_widget(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)
        return page, layout

    def _create_text_label(self, text: str, style: str, *, word_wrap: bool = True) -> QLabel:
        label = QLabel(text)
        label.setAlignment(ALIGN_CENTER)
        label.setWordWrap(word_wrap)
        label.setStyleSheet(style)
        return label

    def _create_nav_button(self, text: str, handler: Callable[[], None], *, secondary: bool = False) -> QPushButton:
        button = QPushButton(text)
        if secondary:
            button.setObjectName("secondary")
        button.setFixedSize(120, 38)
        button.clicked.connect(handler)
        return button

    def _create_card_row(self, cards: list[QWidget], *, spacing: int = 12) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(spacing)
        row.setAlignment(ALIGN_CENTER)
        for card in cards:
            row.addWidget(card)
        return row

    def _page_count(self) -> int:
        return self.stack.count()

    def _is_initialization_page(self, page_index: int) -> bool:
        return page_index == self._page_count() - 1

    def _is_preparation_page(self, page_index: int) -> bool:
        return page_index == self._page_count() - 2

    def _nav_state_for_page(self, page_index: int) -> _NavState:
        is_init_page = self._is_initialization_page(page_index)
        if self._initialization_complete and is_init_page:
            next_text = self.i18n.t("onboarding.finish")
        elif self._is_preparation_page(page_index) and self._preparation_can_finish():
            next_text = self.i18n.t("onboarding.finish")
        elif self._is_preparation_page(page_index):
            next_text = self.i18n.t("onboarding.start_initialization")
        else:
            next_text = self.i18n.t("onboarding.next")
        return _NavState(
            prev_enabled=page_index > 0 and not is_init_page,
            next_text=next_text,
            next_enabled=not is_init_page or self._initialization_complete,
            background_visible=False,
            retry_visible=is_init_page and not self._initialization_complete and self.retry_btn.isVisible(),
        )

    def _apply_nav_state(self, state: _NavState) -> None:
        self.prev_btn.setEnabled(state.prev_enabled)
        self.next_btn.setText(state.next_text)
        self.next_btn.setEnabled(state.next_enabled)
        self.retry_btn.setVisible(state.retry_visible)

    def _refresh_nav_state(self) -> None:
        self._apply_nav_state(self._nav_state_for_page(self.current_page))

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        self.stack = QStackedWidget()
        for page_builder in (
            self._build_welcome_page,
            self._build_update_page,
            self._build_skill_level_page,
            self._build_feature_page,
            self._build_runtime_status_page,
            self._build_initialization_page,
        ):
            self.stack.addWidget(page_builder())
        root.addWidget(self.stack, 1)

        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(10)
        dots_layout.setAlignment(ALIGN_CENTER)
        for _ in range(self._page_count()):
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dots_layout.addWidget(dot)
            self._dots.append(dot)
        root.addLayout(dots_layout)

        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(ALIGN_CENTER)
        nav_layout.setSpacing(12)

        self.prev_btn = self._create_nav_button(self.i18n.t("onboarding.previous"), self._go_previous, secondary=True)
        nav_layout.addWidget(self.prev_btn)

        self.retry_btn = self._create_nav_button(self.i18n.t("repair.retry"), self._retry_initialization, secondary=True)
        self.retry_btn.hide()
        nav_layout.addWidget(self.retry_btn)

        self.next_btn = self._create_nav_button(self.i18n.t("onboarding.next"), self._go_next)
        nav_layout.addWidget(self.next_btn)

        root.addLayout(nav_layout)

    def _build_welcome_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addStretch()
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.lite_welcome_title"), PAGE_TITLE_STYLE))
        layout.addWidget(
            self._create_text_label(self.i18n.t("onboarding.lite_welcome_subtitle"), BODY_SUBTITLE_STYLE)
        )
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.lite_welcome_hint"), HINT_STYLE))
        layout.addStretch()
        return page

    def _build_update_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.update_title"), PAGE_TITLE_STYLE))
        layout.addWidget(
            self._create_text_label(self.i18n.t("onboarding.update_subtitle"), BODY_SUBTITLE_STYLE)
        )
        cards = []
        for option_key in UPDATE_OPTION_KEYS:
            card = SelectableCard(
                option_key,
                self.i18n.t(f"onboarding.update_{option_key}_title"),
                self.i18n.t(f"onboarding.update_{option_key}_desc"),
            )
            card.clicked.connect(self._on_update_option_clicked)
            self._update_cards[option_key] = card
            cards.append(card)
        layout.addLayout(self._create_card_row(cards))
        layout.addStretch()
        return page

    def _build_skill_level_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.skill_title"), PAGE_TITLE_STYLE))
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.skill_subtitle"), BODY_SUBTITLE_STYLE))
        cards = []
        for level_key in SKILL_LEVEL_KEYS:
            card = SkillLevelCard(level_key, self.i18n)
            card.clicked.connect(self._on_skill_level_clicked)
            self._skill_cards[level_key] = card
            cards.append(card)
        layout.addLayout(self._create_card_row(cards))
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.skill_hint"), HINT_STYLE))
        layout.addStretch()
        return page

    def _build_feature_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.features_title"), PAGE_TITLE_STYLE))
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.features_subtitle"), BODY_SUBTITLE_STYLE))
        for feature_key in FULL_FEATURE_SET:
            checkbox = LockedFeatureCheckBox(self.i18n.t(f"onboarding.feature_{feature_key}_label"))
            self._feature_boxes[feature_key] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch()
        return page

    def _build_runtime_status_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(
            self._create_text_label(self.i18n.t("onboarding.runtime_status_title"), PAGE_TITLE_STYLE)
        )
        self.runtime_status_label = self._create_text_label("", BODY_SUBTITLE_STYLE)
        layout.addWidget(self.runtime_status_label)
        status_layout = QVBoxLayout()
        status_layout.setContentsMargins(0, 10, 0, 0)
        status_layout.setSpacing(8)
        for _ in range(5):
            label = StatusBulletLabel()
            self._runtime_status_labels.append(label)
            status_layout.addWidget(label)
        layout.addLayout(status_layout)
        layout.addStretch()
        self._refresh_runtime_status_page()
        return page

    def _build_initialization_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(self._create_text_label(self.i18n.t("onboarding.initialization_title"), PAGE_TITLE_STYLE))
        self.stage_label = self._create_text_label(self.i18n.t("onboarding.initialization_waiting"), BODY_SUBTITLE_STYLE)
        layout.addWidget(self.stage_label)
        self.progress_bar = RoundedProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)
        return page

    def _runtime_hint_text(self) -> str:
        if self.initialization_manager.check_runtime_health():
            return self.i18n.t("onboarding.runtime_check_passed")
        runtime_selection = self.initialization_manager.detect_runtime_selection(
            self.config.selected_runtime_variant or "auto"
        )
        if runtime_selection.variant == "cuda":
            return self.i18n.t("onboarding.runtime_hint_cuda")
        if runtime_selection.variant == "mac":
            return self.i18n.t("onboarding.runtime_hint_mac")
        return self.i18n.t("onboarding.runtime_hint_cpu")

    def _runtime_status_lines(self) -> list[str]:
        runtime_ready = self.initialization_manager.check_runtime_health()
        runtime_selection = self.initialization_manager.detect_runtime_selection(
            self.config.selected_runtime_variant or "auto"
        )
        resolved_runtime_dir = self.initialization_manager.runtime_display_dir(
            self.selected_runtime_install_location
        )
        install_policy = (
            self.i18n.t("onboarding.runtime_status_policy_windows")
            if sys.platform == "win32"
            else self.i18n.t("onboarding.runtime_status_policy_mac")
        )
        health_line = (
            self.i18n.t("onboarding.runtime_status_item_ready")
            if runtime_ready
            else self.i18n.t("onboarding.runtime_status_item_pending")
        )
        variant_line = self.i18n.t(
            "onboarding.runtime_status_item_variant",
            variant=runtime_selection.variant.upper(),
        )
        source_line = self.i18n.t(
            "onboarding.runtime_status_item_source",
            detail=(
                self.i18n.t("onboarding.runtime_status_result_ready")
                if runtime_ready
                else self.i18n.t("onboarding.runtime_status_result_pending")
            ),
        )
        path_line = self.i18n.t("onboarding.runtime_status_path", path=str(resolved_runtime_dir))
        return [health_line, variant_line, install_policy, source_line, path_line]

    def _refresh_runtime_status_page(self) -> None:
        lines = self._runtime_status_lines()
        self.runtime_status_label.setText(self._runtime_hint_text())
        for label, text in zip(self._runtime_status_labels, lines):
            label.setText(text)

    def _apply_single_selection(self, cards: Mapping[str, _SelectableCardLike], selected_key: str):
        for key, card in cards.items():
            card.set_selected(key == selected_key)

    def _sync_defaults(self):
        self._set_auto_update_enabled(self.auto_update_enabled, force=True)
        self._set_skill_level(self.selected_level, force=True)

    def _set_auto_update_enabled(self, enabled: bool, *, force: bool = False):
        if not force and self.auto_update_enabled == enabled:
            return
        self.auto_update_enabled = enabled
        self._apply_single_selection(self._update_cards, "enabled" if enabled else "disabled")

    def _set_skill_level(self, level_key: str, *, force: bool = False):
        if not force and self.selected_level == level_key:
            return
        self.selected_level = level_key
        self._apply_single_selection(self._skill_cards, level_key)

    def _preparation_can_finish(self) -> bool:
        return not self.initialization_manager.needs_initialization(FULL_FEATURE_SET)

    def _set_current_page(self, page_index: int, *, force: bool = False):
        if not 0 <= page_index < self._page_count():
            return
        if not force and self.current_page == page_index:
            return

        self.current_page = page_index
        self.stack.setCurrentIndex(page_index)
        if self._is_preparation_page(page_index):
            self._refresh_runtime_status_page()
        self._refresh_nav_state()
        for index, dot in enumerate(self._dots):
            dot.setStyleSheet(DOT_ACTIVE_STYLE if index == page_index else DOT_INACTIVE_STYLE)

    def _start_initialization(self):
        self._initialization_complete = False
        self._initialization_running = True
        self._closing_after_interrupt = False
        self._set_current_page(self._page_count() - 1)
        self._progress.reset()
        self.log_view.append(self.i18n.t("onboarding.log_start"))
        self._refresh_nav_state()
        self.initialization_manager.start(self.get_selected_options())

    def _complete_onboarding(self):
        self.onboarding_completed.emit(self.selected_level, self.auto_update_enabled)
        self.accept()

    def _on_update_option_clicked(self, option_key: str):
        self._set_auto_update_enabled(option_key == "enabled")

    def _on_skill_level_clicked(self, level_key: str):
        self._set_skill_level(level_key)

    def _go_previous(self):
        self._set_current_page(self.current_page - 1)

    def _go_next(self):
        if self._is_initialization_page(self.current_page):
            if self._initialization_complete:
                self._complete_onboarding()
            return
        if self._is_preparation_page(self.current_page):
            if self._preparation_can_finish():
                self._complete_onboarding()
                return
            self._start_initialization()
            return
        self._set_current_page(self.current_page + 1)

    def _retry_initialization(self):
        self.retry_btn.hide()
        self.log_view.append(self.i18n.t("onboarding.log_retry"))
        self._progress.reset()
        self._initialization_running = True
        self._closing_after_interrupt = False
        self._refresh_nav_state()
        self.initialization_manager.retry_failed()

    def _on_initialization_succeeded(self, _summary: object) -> None:
        self._initialization_complete = True
        self._initialization_running = False
        self.next_btn.setText(self.i18n.t("onboarding.finish"))
        self.next_btn.setEnabled(True)
        self.retry_btn.hide()
        self._refresh_runtime_status_page()
        QApplication.processEvents()
        QTimer.singleShot(300, self._complete_onboarding)

    def _on_initialization_failed(self, summary: object) -> None:
        self._initialization_complete = False
        self._initialization_running = False
        if isinstance(summary, dict) and summary.get("interrupted"):
            if not self._closing_after_interrupt:
                self.stage_label.setText(self.i18n.t("onboarding.initialization_interrupted"))
                self.log_view.append(self.i18n.t("onboarding.initialization_interrupted"))
                self._refresh_nav_state()
            return
        self.retry_btn.show()
        self.next_btn.setEnabled(False)
        error_text = (
            summary.get("error", self.i18n.t("onboarding.initialization_failed"))
            if isinstance(summary, dict)
            else self.i18n.t("onboarding.initialization_failed")
        )
        self.stage_label.setText(error_text)
        self.log_view.append(f"[failed] {error_text}")
        self._refresh_nav_state()

    def _confirm_interrupt_initialization(self) -> bool:
        reply = StyledMessageBox.question(
            self,
            self.i18n.t("onboarding.close_confirm_title"),
            self.i18n.t("onboarding.close_confirm_message"),
            yes_text=self.i18n.t("onboarding.close_confirm_exit"),
            no_text=self.i18n.t("onboarding.close_confirm_continue"),
        )
        return reply == StyledMessageBox.Yes

    def reject(self) -> None:
        if self._initialization_running and not self._closing_after_interrupt:
            if not self._confirm_interrupt_initialization():
                return
            self._closing_after_interrupt = True
            self.initialization_manager.cancel()
        super().reject()

    def closeEvent(self, event) -> None:
        if self._initialization_running and not self._closing_after_interrupt:
            if not self._confirm_interrupt_initialization():
                event.ignore()
                return
            self._closing_after_interrupt = True
            self.initialization_manager.cancel()
        super().closeEvent(event)
