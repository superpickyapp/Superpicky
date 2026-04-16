#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SuperPicky - PySide6 版本入口点
Version: 4.0.6 - Country Selection Simplification
"""

import sys
import os
import multiprocessing

if sys.platform == "darwin":
    multiprocessing.set_start_method("spawn", force=True)

multiprocessing.freeze_support()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _inject_patch_path():
    if sys.platform == "darwin":
        _patch_dir = os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "SuperPicky",
            "code_updates",
        )
    elif sys.platform == "win32":
        _patch_dir = os.path.join(
            os.path.expanduser("~"), "AppData", "Local", "SuperPicky", "code_updates"
        )
    else:
        _patch_dir = os.path.join(
            os.path.expanduser("~"), ".config", "SuperPicky", "code_updates"
        )
    if os.path.isdir(_patch_dir) and _patch_dir not in sys.path:
        sys.path.insert(0, _patch_dir)
    if not hasattr(sys, "_SUPERPICKY_APP_ROOT"):
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            sys._SUPERPICKY_APP_ROOT = os.path.dirname(os.path.abspath(sys.executable))
        elif hasattr(sys, "_MEIPASS"):
            sys._SUPERPICKY_APP_ROOT = sys._MEIPASS
        else:
            sys._SUPERPICKY_APP_ROOT = os.path.dirname(os.path.abspath(__file__))


_inject_patch_path()


def _run_runtime_bootstrap_if_requested():
    if "--runtime-bootstrap" not in sys.argv[1:]:
        return
    from core.runtime_bootstrap import run_runtime_bootstrap

    raise SystemExit(run_runtime_bootstrap(sys.argv[1:]))


_run_runtime_bootstrap_if_requested()

if sys.platform == "win32":
    import io

    def _ensure_utf8_stream(stream):
        if stream is None:
            return open(os.devnull, "w", encoding="utf-8", errors="replace")

        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
            return stream
        except Exception:
            pass

        buffer = getattr(stream, "buffer", None)
        if buffer is not None:
            try:
                return io.TextIOWrapper(buffer, encoding="utf-8", errors="replace")
            except Exception:
                pass

        return stream

    sys.stdout = _ensure_utf8_stream(sys.stdout)
    sys.stderr = _ensure_utf8_stream(sys.stderr)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from app_user_stat.telemetry import bootstrap_telemetry
from ui.main_window import SuperPickyMainWindow
from ui.styles import APP_TOOLTIP_STYLE
from tools.system_logger import setup_error_logging
from config import migrate_old_data, migrate_legacy_ioc_settings

setup_error_logging()
migrate_old_data()
migrate_legacy_ioc_settings()

_memory_monitor = None
if os.environ.get("SP_MEMORY_MONITOR") == "1":
    from tools.memory_monitor import MemoryMonitor

    _memory_monitor = MemoryMonitor(interval=30)

_main_window = None


def main():
    """主函数"""
    global _main_window

    if sys.platform == "darwin":
        safe_cwd = os.path.expanduser("~")
        os.chdir(safe_cwd)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    from constants import APP_VERSION
    from core.build_info import COMMIT_HASH

    commit_hash = COMMIT_HASH
    if commit_hash == "154984fd":
        try:
            import subprocess

            hash_short = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .strip()
                .decode("utf-8")
            )
            commit_hash = hash_short
        except:
            pass

    app.setApplicationName("SuperPicky")
    app.setApplicationDisplayName(f"慧眼选鸟v{APP_VERSION} ({commit_hash})")
    app.setOrganizationName("JamesPhotography")
    app.setOrganizationDomain("jamesphotography.com.au")

    app.setQuitOnLastWindowClosed(False)

    icon_path = os.path.join(os.path.dirname(__file__), "img", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    if sys.platform == "win32":
        from PySide6.QtCore import Qt

        app.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    app.setStyleSheet(APP_TOOLTIP_STYLE)

    if _main_window is None:
        _main_window = SuperPickyMainWindow()
        _main_window.show()
        bootstrap_telemetry(_main_window, on_ready=_main_window.run_startup_prompts)
        app.aboutToQuit.connect(_main_window._cleanup_on_quit)
        if _memory_monitor is not None:
            _memory_monitor.start()
            app.aboutToQuit.connect(_memory_monitor.stop)
    else:
        _main_window.raise_()
        _main_window.activateWindow()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
