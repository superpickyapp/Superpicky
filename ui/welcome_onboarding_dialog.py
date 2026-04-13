# -*- coding: utf-8 -*-
"""
SuperPicky - 轻量首次启动欢迎向导
"""

import os
import sys
from typing import Callable, Mapping, Protocol

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# Support direct preview execution / 支持直接运行本文件进行预览
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.skill_level_dialog import SkillLevelCard
from ui.styles import COLORS, FONTS


UPDATE_OPTION_KEYS = ("enabled", "disabled")
SKILL_LEVEL_KEYS = ("beginner", "intermediate", "master")

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
"""

PAGE_TITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['text_primary']};
        font-size: 24px;
        font-weight: 700;
    }}
"""

WELCOME_SUBTITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 13px;
        line-height: 1.4;
    }}
"""

BODY_SUBTITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 13px;
    }}
"""

SKILL_SUBTITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 14px;
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
    """Minimal selection contract for typed card collections."""

    def set_selected(self, selected: bool) -> None:
        ...


class SelectableCard(QFrame):
    """可点击的简单选择卡片。"""

    clicked = Signal(str)

    def __init__(self, option_key: str, title: str, description: str, parent=None):
        super().__init__(parent)
        self.option_key = option_key
        self._selected = False
        self.setObjectName("updateOptionCard")

        self.setCursor(POINTING_HAND_CURSOR)
        self.setFixedHeight(100)
        self.setMinimumWidth(180)

        self._setup_ui(title, description)
        self._apply_style()

    def _setup_ui(self, title: str, description: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        layout.setAlignment(ALIGN_CENTER)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(ALIGN_CENTER)
        # Reset child label border/background inheritance / 重置子标签样式，避免继承卡片边框高亮
        self.title_label.setStyleSheet(SELECTABLE_CARD_TITLE_STYLE)
        layout.addWidget(self.title_label)

        self.desc_label = QLabel(description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setAlignment(ALIGN_CENTER)
        self.desc_label.setStyleSheet(SELECTABLE_CARD_DESC_STYLE)
        layout.addWidget(self.desc_label)

    def set_selected(self, selected: bool):
        if self._selected == selected:
            return
        self._selected = selected
        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet(
            SELECTABLE_CARD_SELECTED_STYLE
            if self._selected
            else SELECTABLE_CARD_UNSELECTED_STYLE
        )

    def mousePressEvent(self, event):
        self.clicked.emit(self.option_key)
        super().mousePressEvent(event)


class WelcomeOnboardingDialog(QDialog):
    """首次启动欢迎向导。"""

    onboarding_completed = Signal(str, bool)

    def __init__(self, i18n, parent=None):
        super().__init__(parent)
        self.i18n = i18n
        self.current_page = 0
        self.selected_level = "intermediate"
        self.auto_update_enabled = True
        self._dots: list[QLabel] = []
        self._skill_cards: dict[str, SkillLevelCard] = {}
        self._update_cards: dict[str, SelectableCard] = {}

        self.setModal(True)
        self.setWindowTitle(self.i18n.t("onboarding.window_title"))
        self.setFixedSize(560, 420)
        self.setStyleSheet(DIALOG_STYLE)

        self._setup_ui()
        self._set_current_page(self.current_page, force=True)

    def _create_page_widget(self) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(12)
        return page, layout

    def _create_text_label(
        self,
        text: str,
        style: str,
        *,
        word_wrap: bool = True,
        alignment: Qt.AlignmentFlag = ALIGN_CENTER,
    ) -> QLabel:
        label = QLabel(text)
        label.setAlignment(alignment)
        label.setWordWrap(word_wrap)
        label.setStyleSheet(style)
        return label

    def _create_nav_button(
        self,
        text: str,
        handler: Callable[[], None],
        *,
        secondary: bool = False,
    ) -> QPushButton:
        button = QPushButton(text)
        if secondary:
            button.setObjectName("secondary")
        button.setFixedSize(110, 38)
        button.clicked.connect(handler)
        return button

    def _create_card_row(
        self,
        cards: list[QWidget],
        *,
        spacing: int = 12,
        margins: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(*margins)
        row.setSpacing(spacing)
        row.setAlignment(ALIGN_CENTER)
        for card in cards:
            row.addWidget(card)
        return row

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        self.stack = QStackedWidget()
        for page_builder in (
            self._build_welcome_page,
            self._build_update_page,
            self._build_skill_level_page,
        ):
            self.stack.addWidget(page_builder())
        root.addWidget(self.stack, 1)

        dots_layout = QHBoxLayout()
        dots_layout.setSpacing(10)
        dots_layout.setAlignment(ALIGN_CENTER)
        for _ in range(self.stack.count()):
            dot = QLabel()
            dot.setFixedSize(10, 10)
            dots_layout.addWidget(dot)
            self._dots.append(dot)
        root.addLayout(dots_layout)

        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(ALIGN_CENTER)
        nav_layout.setSpacing(12)

        self.prev_btn = self._create_nav_button(
            self.i18n.t("onboarding.previous"),
            self._go_previous,
            secondary=True,
        )
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = self._create_nav_button(
            self.i18n.t("onboarding.next"),
            self._go_next,
        )
        nav_layout.addWidget(self.next_btn)

        root.addLayout(nav_layout)

    def _build_welcome_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addStretch()
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.welcome_title"),
                PAGE_TITLE_STYLE,
            )
        )
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.welcome_subtitle"),
                WELCOME_SUBTITLE_STYLE,
            )
        )
        layout.addStretch()
        return page

    def _build_update_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.update_title"),
                PAGE_TITLE_STYLE,
            )
        )
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.update_subtitle"),
                BODY_SUBTITLE_STYLE,
            )
        )
        layout.addSpacing(2)

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

        layout.addLayout(self._create_card_row(cards, margins=(0, 8, 0, 8)))
        layout.addStretch()
        self._set_auto_update_enabled(self.auto_update_enabled, force=True)
        return page

    def _build_skill_level_page(self) -> QWidget:
        page, layout = self._create_page_widget()
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.skill_title"),
                PAGE_TITLE_STYLE,
            )
        )
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.skill_subtitle"),
                SKILL_SUBTITLE_STYLE,
            )
        )
        layout.addSpacing(6)

        cards = []
        for level_key in SKILL_LEVEL_KEYS:
            card = SkillLevelCard(level_key, self.i18n)
            card.clicked.connect(self._on_skill_level_clicked)
            self._skill_cards[level_key] = card
            cards.append(card)

        layout.addLayout(self._create_card_row(cards))
        layout.addWidget(
            self._create_text_label(
                self.i18n.t("onboarding.skill_hint"),
                HINT_STYLE,
            )
        )
        layout.addStretch()
        self._set_skill_level(self.selected_level, force=True)
        return page

    def _apply_single_selection(
        self,
        cards: Mapping[str, _SelectableCardLike],
        selected_key: str,
    ):
        for key, card in cards.items():
            card.set_selected(key == selected_key)

    def _set_auto_update_enabled(self, enabled: bool, *, force: bool = False):
        # Skip redundant UI refreshes / 跳过重复状态刷新，减少不必要的样式重设
        if not force and self.auto_update_enabled == enabled:
            return
        self.auto_update_enabled = enabled
        selected_key = "enabled" if enabled else "disabled"
        self._apply_single_selection(self._update_cards, selected_key)

    def _set_skill_level(self, level_key: str, *, force: bool = False):
        # Centralize selection updates in one place / 统一集中处理选中态，避免多处散落更新
        if not force and self.selected_level == level_key:
            return
        self.selected_level = level_key
        self._apply_single_selection(self._skill_cards, level_key)

    def _set_current_page(self, page_index: int, *, force: bool = False):
        if not 0 <= page_index < self.stack.count():
            return
        if not force and self.current_page == page_index:
            return

        # Keep page state and page UI in sync here / 在统一入口同步页面状态与界面
        self.current_page = page_index
        self.stack.setCurrentIndex(page_index)
        self.prev_btn.setEnabled(page_index > 0)
        self.next_btn.setText(
            self.i18n.t("onboarding.finish")
            if page_index == self.stack.count() - 1
            else self.i18n.t("onboarding.next")
        )

        for index, dot in enumerate(self._dots):
            dot.setStyleSheet(
                DOT_ACTIVE_STYLE if index == page_index else DOT_INACTIVE_STYLE
            )

    def _complete_onboarding(self):
        # Keep payload order stable for main_window.py / 保持信号参数顺序稳定，避免影响 main_window.py
        self.onboarding_completed.emit(
            self.selected_level,
            self.auto_update_enabled,
        )
        self.accept()

    def _on_update_option_clicked(self, option_key: str):
        self._set_auto_update_enabled(option_key == "enabled")

    def _on_skill_level_clicked(self, level_key: str):
        self._set_skill_level(level_key)

    def _go_previous(self):
        self._set_current_page(self.current_page - 1)

    def _go_next(self):
        if self.current_page >= self.stack.count() - 1:
            self._complete_onboarding()
            return
        self._set_current_page(self.current_page + 1)


if __name__ == "__main__":
    from tools.i18n import get_i18n

    app = QApplication(sys.argv)
    dialog = WelcomeOnboardingDialog(get_i18n())
    dialog.onboarding_completed.connect(
        lambda level, auto_update: print(
            f"[preview] onboarding_completed level={level} auto_update={auto_update}"
        )
    )
    dialog.show()
    sys.exit(app.exec())
