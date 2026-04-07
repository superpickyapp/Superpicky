import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

SPEC_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASE_PATH = SPEC_DIR.resolve()
sys.path.insert(0, str(BASE_PATH))

from constants import APP_VERSION


def _existing_tree(src: Path, dest: str):
    if src.exists():
        return [(str(src), dest)]
    return []


def _safe_collect_data_files(package_name: str):
    try:
        return collect_data_files(package_name)
    except Exception:
        return []


def _safe_copy_metadata(dist_name: str):
    try:
        return copy_metadata(dist_name)
    except Exception:
        return []


def _safe_collect_onnxruntime_cpu_data_files():
    datas = []
    seen = set()
    excluded_suffixes = (
        "onnxruntime/capi/onnxruntime_providers_cuda.dll",
        "onnxruntime/capi/onnxruntime_providers_tensorrt.dll",
    )

    try:
        for src, dest in collect_data_files("onnxruntime"):
            normalized = f"{dest}/{os.path.basename(src)}".replace("\\", "/")
            if normalized.endswith(excluded_suffixes):
                continue
            key = (src, dest)
            if key in seen:
                continue
            seen.add(key)
            datas.append((src, dest))
    except Exception:
        return []

    return datas


def _collect_onnxruntime_cpu_binaries():
    binaries = []
    seen = set()
    excluded_prefixes = ("cublas", "cufft", "cudnn", "cudart", "nv")
    excluded_names = {
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_tensorrt.dll",
    }

    def add_entries(entries):
        for entry in entries:
            key = tuple(entry[:2])
            if key in seen:
                continue
            name = os.path.basename(entry[0]).lower()
            if name in excluded_names or name.startswith(excluded_prefixes):
                continue
            seen.add(key)
            binaries.append(entry)

    add_entries(collect_dynamic_libs("onnxruntime"))

    try:
        import onnxruntime as ort

        capi_dir = Path(ort.__file__).resolve().parent / "capi"
        for pattern in ("*.dll", "*.pyd"):
            for path in sorted(capi_dir.glob(pattern)):
                add_entries([(str(path), "onnxruntime/capi")])
    except Exception:
        pass

    return binaries


PACKAGE_DATA_NAMES = [
    "imageio",
    "rawpy",
    "pillow_heif",
]

all_datas = []
all_datas.extend(_existing_tree(BASE_PATH / "models", "models"))
all_datas.extend(_existing_tree(BASE_PATH / "exiftools_win", "exiftools_win"))
all_datas.extend(_existing_tree(BASE_PATH / "img", "img"))
all_datas.extend(_existing_tree(BASE_PATH / "locales", "locales"))
all_datas.extend(_existing_tree(BASE_PATH / "locales" / "en.lproj", "en.lproj"))
all_datas.extend(_existing_tree(BASE_PATH / "locales" / "zh-Hans.lproj", "zh-Hans.lproj"))
all_datas.extend(_existing_tree(BASE_PATH / "birdid" / "data", "birdid/data"))
all_datas.extend(_existing_tree(BASE_PATH / "SuperBirdIDPlugin.lrplugin", "SuperBirdIDPlugin.lrplugin"))

for package_name in PACKAGE_DATA_NAMES:
    all_datas.extend(_safe_collect_data_files(package_name))

all_datas.extend(_safe_collect_onnxruntime_cpu_data_files())

for dist_name in [
    "imageio",
    "rawpy",
    "pillow_heif",
    "onnxruntime",
]:
    all_datas.extend(_safe_copy_metadata(dist_name))

hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._ld_preload",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "onnxruntime.capi.onnxruntime_validation",
    "PIL",
    "cv2",
    "numpy",
    "yaml",
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "imageio",
    "rawpy",
    "imagehash",
    "pywt",
    "pillow_heif",
    "core",
    "core.burst_detector",
    "core.config_manager",
    "core.exposure_detector",
    "core.file_manager",
    "core.flight_detector_onnx",
    "core.focus_point_detector",
    "core.keypoint_detector_onnx",
    "core.photo_processor",
    "core.rating_engine",
    "core.stats_formatter",
    "multiprocessing",
    "multiprocessing.spawn",
    "tools.update_checker",
    "packaging",
    "packaging.version",
    "birdid",
    "birdid.bird_identifier_onnx",
    "birdid.osea_classifier_onnx",
    "birdid.ebird_country_filter",
    "iqa_scorer_onnx",
    "birdid_server",
    "server_manager",
    "flask",
    "flask.json",
    "cryptography",
    "cryptography.fernet",
]

a = Analysis(
    ["main.py"],
    pathex=[str(BASE_PATH)],
    binaries=_collect_onnxruntime_cpu_binaries(),
    datas=all_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        hook_name
        for hook_name, hook_path in [
            ("pyi_rth_cv2.py", BASE_PATH / "pyi_rth_cv2.py"),
        ]
        if hook_path.exists()
    ],
    excludes=[
        "PyQt5",
        "PyQt6",
        "tkinter",
        "ultralytics",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

icon_ico = BASE_PATH / "img" / "icon.ico"
icon_icns = BASE_PATH / "img" / "SuperPicky-V0.02.icns"
exe_icon = None
if sys.platform == "win32" and icon_ico.exists():
    exe_icon = str(icon_ico)
elif icon_icns.exists():
    exe_icon = str(icon_icns)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SuperPicky",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SuperPicky",
)

app = BUNDLE(
    coll,
    name="SuperPicky.app",
    icon=str(icon_icns) if icon_icns.exists() else None,
    bundle_identifier="com.jamesphotography.superpicky",
    info_plist={
        "CFBundleName": "SuperPicky",
        "CFBundleDisplayName": "SuperPicky",
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSAppleEventsUsageDescription": "SuperPicky needs AppleEvents permission to communicate with other apps.",
        "NSAppleScriptEnabled": False,
    },
)
