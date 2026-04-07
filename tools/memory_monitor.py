#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - 内存监视器
后台线程定期采样，统计：
  - 进程 RSS（psutil）
  - Python 堆分配按模块占比（tracemalloc）
  - MPS / CUDA 显存
  - 对象数量 Top N（objgraph，可选）

用法：
    from tools.memory_monitor import MemoryMonitor
    monitor = MemoryMonitor(interval=30)
    monitor.start()
    ...
    monitor.stop()
    monitor.snapshot("处理结束后")   # 手动触发快照并对比基线

日志写入 <SuperPicky 配置目录>/memory_monitor.log
"""

import os
import sys
import time
import threading
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Optional


def _get_config_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SuperPicky"
    elif sys.platform == "win32":
        return Path.home() / "AppData" / "Local" / "SuperPicky"
    else:
        return Path.home() / ".config" / "SuperPicky"


def _get_log_path() -> Path:
    d = _get_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "memory_monitor.log"


def _fmt_bytes(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024**3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024**2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def _get_process_rss() -> Optional[int]:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss
    except Exception:
        return None


def _get_gpu_memory() -> dict:
    result = {}
    try:
        import torch
        if torch.backends.mps.is_available():
            result["MPS current"] = torch.mps.current_allocated_memory()
            result["MPS driver"] = torch.mps.driver_allocated_memory()
        elif torch.cuda.is_available():
            result["CUDA alloc"] = torch.cuda.memory_allocated()
            result["CUDA reserved"] = torch.cuda.memory_reserved()
    except Exception:
        pass
    return result


def _get_top_modules(snapshot, top_n: int = 20):
    """从 tracemalloc 快照按模块汇总内存，返回 [(module_short, size_bytes), ...]"""
    stats = snapshot.statistics("filename")
    module_totals = {}
    for stat in stats:
        filename = stat.traceback[0].filename
        # 缩短路径：只保留 site-packages 之后 或 项目目录之后的部分
        short = filename
        for marker in ("site-packages/", "site-packages\\"):
            idx = filename.find(marker)
            if idx != -1:
                short = filename[idx + len(marker):]
                break
        else:
            # 尝试取项目相对路径
            cwd = os.getcwd()
            if filename.startswith(cwd):
                short = filename[len(cwd):].lstrip("/\\")

        module_totals[short] = module_totals.get(short, 0) + stat.size

    sorted_items = sorted(module_totals.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]


def _get_top_objects(top_n: int = 15):
    """用 objgraph 统计对象数量（可选依赖）"""
    try:
        import objgraph
        return objgraph.most_common_types(limit=top_n)
    except ImportError:
        return None


class MemoryMonitor:
    """
    后台内存监视器。

    参数：
        interval   -- 采样间隔秒数（默认 30）
        top_n      -- tracemalloc 报告的模块数量（默认 20）
        log_fn     -- 可选的额外输出回调 fn(str)，不传则只写文件
    """

    def __init__(self, interval: int = 30, top_n: int = 20, log_fn=None):
        self.interval = interval
        self.top_n = top_n
        self.log_fn = log_fn
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline_snapshot = None
        self._baseline_rss = None
        self._sample_index = 0
        self._log_path = _get_log_path()

    # ─────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────

    def start(self):
        """启动后台采样线程，同时开启 tracemalloc。"""
        if self._thread and self._thread.is_alive():
            return
        tracemalloc.start(10)  # 保存 10 帧调用栈
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="MemoryMonitor", daemon=True
        )
        self._thread.start()
        self._write(f"[MemoryMonitor] 启动，间隔={self.interval}s，日志={self._log_path}")

    def stop(self):
        """停止后台采样线程。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        tracemalloc.stop()
        self._write("[MemoryMonitor] 已停止")

    def snapshot(self, label: str = "手动快照"):
        """
        立即触发一次快照并输出，与基线对比差值。
        可在关键代码节点手动调用：
            monitor.snapshot("处理完 500 张后")
        """
        self._do_snapshot(label=label, compare_baseline=True)

    # ─────────────────────────────────────────────
    # 内部实现
    # ─────────────────────────────────────────────

    def _run(self):
        # 第一次：建立基线
        time.sleep(2)  # 等应用初始化完成
        self._do_snapshot(label="基线（启动后）", set_baseline=True)

        while not self._stop_event.wait(self.interval):
            self._sample_index += 1
            self._do_snapshot(
                label=f"定期采样 #{self._sample_index}",
                compare_baseline=True,
            )

    def _do_snapshot(
        self,
        label: str = "",
        set_baseline: bool = False,
        compare_baseline: bool = False,
    ):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"  {ts}  |  {label}")
        lines.append("=" * 72)

        # ── 进程 RSS ─────────────────────────────────────
        rss = _get_process_rss()
        if rss is not None:
            rss_str = _fmt_bytes(rss)
            delta_str = ""
            if compare_baseline and self._baseline_rss is not None:
                delta = rss - self._baseline_rss
                sign = "+" if delta >= 0 else ""
                delta_str = f"  (基线差: {sign}{_fmt_bytes(delta)})"
            lines.append(f"  进程 RSS : {rss_str}{delta_str}")
        else:
            lines.append("  进程 RSS : 不可用 (psutil 缺失)")

        # ── GPU 显存 ──────────────────────────────────────
        gpu = _get_gpu_memory()
        if gpu:
            lines.append("  GPU 显存 :")
            for k, v in gpu.items():
                lines.append(f"    {k:20s}: {_fmt_bytes(v)}")

        # ── Python 堆（tracemalloc）─────────────────────
        if tracemalloc.is_tracing():
            snap = tracemalloc.take_snapshot()
            total_traced = sum(s.size for s in snap.statistics("filename"))
            lines.append(f"  Python 堆追踪总量: {_fmt_bytes(total_traced)}")

            top = _get_top_modules(snap, self.top_n)
            if top:
                lines.append(f"  {'模块/文件':<52} {'大小':>10}  {'占比':>6}")
                lines.append("  " + "-" * 72)
                for mod, size in top:
                    pct = size / total_traced * 100 if total_traced else 0
                    lines.append(f"  {mod:<52} {_fmt_bytes(size):>10}  {pct:5.1f}%")

            # 与基线对比 top 差值
            if compare_baseline and self._baseline_snapshot is not None:
                diff_stats = snap.compare_to(self._baseline_snapshot, "filename")
                diff_stats = [d for d in diff_stats if d.size_diff != 0]
                diff_stats.sort(key=lambda d: d.size_diff, reverse=True)
                if diff_stats:
                    lines.append("")
                    lines.append(f"  [与基线对比 Top 10 增量]")
                    lines.append(f"  {'模块/文件':<52} {'增量':>12}")
                    lines.append("  " + "-" * 68)
                    for d in diff_stats[:10]:
                        fn = d.traceback[0].filename
                        for marker in ("site-packages/", "site-packages\\"):
                            idx = fn.find(marker)
                            if idx != -1:
                                fn = fn[idx + len(marker):]
                                break
                        else:
                            cwd = os.getcwd()
                            if fn.startswith(cwd):
                                fn = fn[len(cwd):].lstrip("/\\")
                        sign = "+" if d.size_diff >= 0 else ""
                        lines.append(f"  {fn:<52} {sign}{_fmt_bytes(d.size_diff):>12}")

            if set_baseline:
                self._baseline_snapshot = snap
                self._baseline_rss = rss

        # ── 对象数量（objgraph）────────────────────────
        obj_stats = _get_top_objects(15)
        if obj_stats is not None:
            lines.append("")
            lines.append("  [对象数量 Top 15  (objgraph)]")
            lines.append(f"  {'类型':<40} {'数量':>8}")
            lines.append("  " + "-" * 52)
            for cls_name, count in obj_stats:
                lines.append(f"  {cls_name:<40} {count:>8,}")
        else:
            lines.append("  [提示] pip install objgraph 可获得按类对象数量统计")

        text = "\n".join(lines)
        self._write(text)

    def _write(self, text: str):
        # 写文件
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception:
            pass
        # 控制台输出
        try:
            print(text)
        except Exception:
            pass
        # 可选回调（如 GUI log 面板）
        if self.log_fn:
            try:
                self.log_fn(text)
            except Exception:
                pass