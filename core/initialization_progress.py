# -*- coding: utf-8 -*-
"""
Initialization progress models for lightweight onboarding flows.

This module centralizes structured progress events, stage-to-phase mapping,
and the deterministic progress animation policy shared by the welcome dialog
and the repair dialog. The model keeps UI timing logic out of Qt widgets so it
can be verified with small, repeatable tests.

轻量化初始化流程的进度模型。

此模块集中定义结构化进度事件、阶段到动画阶段的映射关系，以及欢迎向导与修复对话框共享的确定性进度动画策略。
这样可以把 UI 的时间推进逻辑从 Qt 控件中抽离出来，便于使用小型、可重复的测试进行验证。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


PROGRESS_KIND_RUNTIME = "runtime_install"
PROGRESS_KIND_DOWNLOAD = "resource_download"

STAGE_PROBING = "probing_sources"
STAGE_CHECKING_UPDATES = "checking_updates"
STAGE_PREPARING_RUNTIME = "preparing_runtime"
STAGE_DOWNLOADING = "downloading_resources"
STAGE_VERIFYING = "verifying"
STAGE_READY = "ready"
STAGE_FAILED = "failed"

PIP_RAW_PROGRESS_PATTERN = re.compile(r"^Progress\s+(?P<current>\d+)\s+of\s+(?P<total>\d+)$")


@dataclass(frozen=True)
class InitializationProgressEvent:
    """
    Structured progress payload emitted by initialization subsystems.

    初始化子系统发出的结构化进度负载。

    Attributes:
        stage: Pipeline stage that owns this event.
        progress_kind: Logical progress stream, such as runtime install or resource download.
        message: Human-readable status text for logs or labels.
        ratio: Normalized completion ratio in [0.0, 1.0] when known.
        bytes_done: Downloaded or processed bytes when known.
        bytes_total: Expected total bytes when known.
        item_index: Zero-based item index in a multi-item batch.
        item_count: Total item count in a multi-item batch.
        resource_id: Resource identifier for per-item tracking.
        source: Active mirror or backend source name.
        is_terminal: Whether the current item or phase has completed.
    """

    stage: str
    progress_kind: str
    message: str
    ratio: float | None = None
    bytes_done: int | None = None
    bytes_total: int | None = None
    item_index: int | None = None
    item_count: int | None = None
    resource_id: str | None = None
    source: str | None = None
    is_terminal: bool = False

    def normalized_ratio(self) -> float | None:
        """
        Return a clamped completion ratio when one can be inferred.

        当可以推导出完成比时，返回一个夹紧后的进度比例。
        """
        if self.ratio is not None:
            return max(0.0, min(1.0, float(self.ratio)))
        if self.bytes_total and self.bytes_total > 0 and self.bytes_done is not None:
            return max(0.0, min(1.0, float(self.bytes_done) / float(self.bytes_total)))
        return None


@dataclass(frozen=True)
class ProgressSnapshot:
    """
    Render-ready progress state returned by the animation model.

    动画模型返回的可直接用于渲染的进度状态。
    """

    display_percent: int
    actual_percent: int
    target_percent: int
    display_value: float
    actual_value: float
    target_value: float
    active_phase: str | None
    is_finishing: bool
    is_settled: bool
    suggested_interval_ms: int


@dataclass(frozen=True)
class ProgressPhaseProfile:
    """
    Visual progress allocation and timing policy for one logical phase.

    单个逻辑阶段的视觉进度分配与时间策略。
    """

    key: str
    start_percent: float
    end_percent: float
    min_duration_seconds: float
    max_duration_seconds: float

    @property
    def span(self) -> float:
        """Return the percentage span owned by this phase."""
        return self.end_percent - self.start_percent


def phase_from_stage(stage: str) -> str | None:
    """
    Map an initialization stage to the owning visual phase.

    将初始化阶段映射到对应的视觉动画阶段。
    """
    if stage == STAGE_PREPARING_RUNTIME:
        return PROGRESS_KIND_RUNTIME
    if stage == STAGE_DOWNLOADING:
        return PROGRESS_KIND_DOWNLOAD
    return None


def parse_pip_raw_progress_line(line: str) -> tuple[int, int] | None:
    """
    Parse `pip --progress-bar raw` lines into byte counters.

    解析 `pip --progress-bar raw` 输出行，提取字节级进度。
    """
    match = PIP_RAW_PROGRESS_PATTERN.match(line.strip())
    if not match:
        return None
    return int(match.group("current")), int(match.group("total"))


class InitializationProgressModel:
    """
    Deterministic progress animation model for long-running initialization tasks.

    长耗时初始化任务的确定性进度动画模型。

    The model combines three inputs:
    1. Real progress as a non-regressing lower bound.
    2. A time-driven target that keeps long tasks within the 400-600 second visual window.
    3. Small bounded oscillations that simulate natural speed fluctuation without
       producing visible backward movement or large real-progress drift.

    该模型组合三类输入：
    1. 真实进度，作为不可回退的下界。
    2. 时间驱动目标，用于让长任务的视觉窗口保持在 400-600 秒区间。
    3. 有界的小幅波动，用于模拟自然的速度抖动，同时避免明显倒退或过度偏离真实进度。
    """

    PHASES: dict[str, ProgressPhaseProfile] = {
        PROGRESS_KIND_RUNTIME: ProgressPhaseProfile(
            key=PROGRESS_KIND_RUNTIME,
            start_percent=0.0,
            end_percent=34.0,
            min_duration_seconds=145.0,
            max_duration_seconds=220.0,
        ),
        PROGRESS_KIND_DOWNLOAD: ProgressPhaseProfile(
            key=PROGRESS_KIND_DOWNLOAD,
            start_percent=34.0,
            end_percent=99.0,
            min_duration_seconds=250.0,
            max_duration_seconds=380.0,
        ),
    }

    def __init__(
        self,
        *,
        seed: int = 17,
        min_visible_progress: int = 4,
        suggested_interval_ms: int = 80,
    ) -> None:
        """
        Initialize the animation model with deterministic timing parameters.

        使用确定性时间参数初始化动画模型。
        """
        self._seed = float(seed)
        self._min_visible_progress = min_visible_progress
        self._suggested_interval_ms = suggested_interval_ms
        self.reset(0.0)

    def reset(self, now: float = 0.0) -> ProgressSnapshot:
        """
        Reset the model to its initial idle state.

        将模型重置为初始空闲状态。
        """
        self._display_percent = 0.0
        self._actual_percent = 0.0
        self._target_percent = 0.0
        self._active_phase: str | None = None
        self._phase_started_at = now
        self._phase_target_seconds = 0.0
        self._phase_peak_target = 0.0
        self._finish_started_at: float | None = None
        self._finish_duration = 0.0
        self._finish_from = 0.0
        self._settled = False
        return self.snapshot()

    def on_stage_changed(self, stage: str, now: float) -> ProgressSnapshot:
        """
        React to a stage transition emitted by the initialization manager.

        响应初始化管理器发出的阶段切换事件。
        """
        phase_key = phase_from_stage(stage)
        if phase_key is not None:
            self._activate_phase(phase_key, now)
        elif stage == STAGE_VERIFYING:
            self._actual_percent = max(self._actual_percent, 99.0)
            self._target_percent = max(self._target_percent, 99.0)
        elif stage in {STAGE_READY, STAGE_FAILED}:
            self._phase_peak_target = max(self._phase_peak_target, self._target_percent)
        return self.advance(now)

    def on_progress_event(
        self,
        event: InitializationProgressEvent,
        now: float,
    ) -> ProgressSnapshot:
        """
        Update actual progress from a structured subsystem event.

        使用结构化子系统事件更新真实进度。
        """
        phase_key = event.progress_kind if event.progress_kind in self.PHASES else phase_from_stage(event.stage)
        if phase_key is not None:
            self._activate_phase(
                phase_key,
                now,
                bytes_total=event.bytes_total,
                item_count=event.item_count,
            )
            phase = self.PHASES[phase_key]
            ratio = event.normalized_ratio()
            if ratio is not None:
                actual = phase.start_percent + (phase.span * ratio)
                self._actual_percent = max(self._actual_percent, actual)
            if event.is_terminal:
                self._actual_percent = max(self._actual_percent, phase.end_percent)
        return self.advance(now)

    def on_finished(self, success: bool, now: float) -> ProgressSnapshot:
        """
        Enter the success settle animation or stop immediately on failure.

        成功时进入收尾动画，失败时立即停止动画推进。
        """
        if not success:
            self._settled = True
            return self.snapshot()

        self._actual_percent = max(self._actual_percent, 100.0)
        self._target_percent = max(self._target_percent, 100.0)
        self._finish_started_at = now
        self._finish_from = max(self._display_percent, min(self._actual_percent, 99.5))
        remaining = max(0.0, 100.0 - self._finish_from)
        self._finish_duration = min(2.0, max(1.0, 1.0 + (remaining / 60.0)))
        self._settled = False
        return self.advance(now)

    def advance(self, now: float) -> ProgressSnapshot:
        """
        Advance the animation to the specified monotonic time.

        将动画推进到指定的单调时间点。
        """
        if self._finish_started_at is not None:
            elapsed = max(0.0, now - self._finish_started_at)
            ratio = min(1.0, elapsed / max(0.001, self._finish_duration))
            eased = 1.0 - (1.0 - ratio) ** 3
            self._display_percent = self._finish_from + ((100.0 - self._finish_from) * eased)
            if ratio >= 1.0:
                self._display_percent = 100.0
                self._settled = True
            return self.snapshot()

        if self._active_phase is None:
            return self.snapshot()

        time_target = self._compute_time_target(now)
        desired = max(self._actual_percent, time_target)
        self._target_percent = max(self._target_percent, desired)
        self._display_percent = max(self._display_percent, desired)
        return self.snapshot()

    def snapshot(self) -> ProgressSnapshot:
        """
        Return an immutable render snapshot for the current model state.

        返回当前模型状态的不可变渲染快照。
        """
        display_percent = int(round(self._display_percent))
        actual_percent = int(round(self._actual_percent))
        target_percent = int(round(self._target_percent))

        if 0 < display_percent < self._min_visible_progress:
            display_percent = self._min_visible_progress

        return ProgressSnapshot(
            display_percent=max(0, min(100, display_percent)),
            actual_percent=max(0, min(100, actual_percent)),
            target_percent=max(0, min(100, target_percent)),
            display_value=max(0.0, min(100.0, self._display_percent)),
            actual_value=max(0.0, min(100.0, self._actual_percent)),
            target_value=max(0.0, min(100.0, self._target_percent)),
            active_phase=self._active_phase,
            is_finishing=self._finish_started_at is not None and not self._settled,
            is_settled=self._settled,
            suggested_interval_ms=self._suggested_interval_ms,
        )

    def _activate_phase(
        self,
        phase_key: str,
        now: float,
        *,
        bytes_total: int | None = None,
        item_count: int | None = None,
    ) -> None:
        """
        Enter or retune a visual phase when new information arrives.

        在新的信息到达时进入或重新调整视觉阶段。
        """
        profile = self.PHASES[phase_key]
        is_new_phase = self._active_phase != phase_key
        if is_new_phase:
            self._active_phase = phase_key
            self._phase_started_at = now
            self._phase_peak_target = max(self._display_percent, profile.start_percent)
            self._display_percent = max(self._display_percent, profile.start_percent)
            self._actual_percent = max(self._actual_percent, profile.start_percent)

        # When size information appears later, only retune the target duration while
        # keeping the current elapsed progress and monotonic target intact.
        # 当后续才拿到大小信息时，只重算目标时长，不回退当前时间进度和单调目标值。
        self._phase_target_seconds = self._choose_target_duration(
            profile,
            bytes_total=bytes_total,
            item_count=item_count,
        )

    def _choose_target_duration(
        self,
        profile: ProgressPhaseProfile,
        *,
        bytes_total: int | None,
        item_count: int | None,
    ) -> float:
        """
        Choose a phase duration inside the configured long-task window.

        在配置好的长任务窗口内选择当前阶段的目标时长。
        """
        if profile.key == PROGRESS_KIND_RUNTIME:
            base = 180.0
            if bytes_total and bytes_total > 0:
                base += min(35.0, bytes_total / float(1024 ** 3) * 20.0)
            return min(profile.max_duration_seconds, max(profile.min_duration_seconds, base))

        base = 300.0
        if bytes_total and bytes_total > 0:
            size_gib = bytes_total / float(1024 ** 3)
            base += min(55.0, size_gib * 45.0)
        if item_count and item_count > 1:
            base += min(35.0, float(item_count - 1) * 8.0)
        return min(profile.max_duration_seconds, max(profile.min_duration_seconds, base))

    def _compute_time_target(self, now: float) -> float:
        """
        Compute the monotonic time-driven target for the active phase.

        计算当前活动阶段的单调时间驱动目标值。
        """
        assert self._active_phase is not None
        profile = self.PHASES[self._active_phase]
        elapsed = max(0.0, now - self._phase_started_at)
        duration = max(1.0, self._phase_target_seconds)
        progress_ratio = min(0.985, elapsed / duration)
        jitter = self._bounded_jitter(profile, elapsed, progress_ratio)
        raw_ratio = min(0.985, max(0.0, progress_ratio + jitter))
        target = profile.start_percent + (profile.span * raw_ratio)
        self._phase_peak_target = max(self._phase_peak_target, target)
        return self._phase_peak_target

    def _bounded_jitter(
        self,
        profile: ProgressPhaseProfile,
        elapsed: float,
        progress_ratio: float,
    ) -> float:
        """
        Return a small deterministic oscillation for natural-looking speed changes.

        返回一个小幅、确定性的振荡量，用于制造更自然的速度变化。
        """
        damping = max(0.18, 1.0 - progress_ratio)
        amplitude = 0.016 * damping
        if profile.key == PROGRESS_KIND_DOWNLOAD:
            amplitude += 0.004

        wave = (
            math.sin((elapsed * 0.10) + self._seed)
            + 0.45 * math.sin((elapsed * 0.27) + (self._seed * 1.7))
            + 0.2 * math.sin((elapsed * 0.51) + (self._seed * 2.3))
        )
        jitter = (wave / 1.65) * amplitude
        return min(0.022, max(-0.018, jitter))
