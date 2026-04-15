import os
import site
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
import sys
sys.path.append(os.path.abspath('.'))
from constants import APP_VERSION

base_path = os.path.abspath('.')
sp = site.getsitepackages()
site_packages = sp[1] if len(sp) > 1 else sp[0]

ultralytics_base = site_packages
if not os.path.exists(os.path.join(ultralytics_base, 'ultralytics')):
    try:
        import ultralytics
        ultralytics_base = os.path.dirname(os.path.dirname(ultralytics.__file__))
    except ImportError:
        pass

ultralytics_datas = collect_data_files('ultralytics')
imageio_datas = collect_data_files('imageio')
rawpy_datas = collect_data_files('rawpy')
pillow_heif_datas = collect_data_files('pillow_heif')

all_datas = [
    # Legacy full-build datas remain in SuperPicky.spec.
    # The lite build intentionally excludes bundled models and birdid/data so
    # first-run initialization can fetch them on demand.
    (os.path.join(base_path, 'exiftools_mac'), 'exiftools_mac'),
    (os.path.join(base_path, 'img'), 'img'),
    (os.path.join(base_path, 'locales'), 'locales'),
    (os.path.join(base_path, 'locales', 'en.lproj'), 'en.lproj'),
    (os.path.join(base_path, 'locales', 'zh-Hans.lproj'), 'zh-Hans.lproj'),
    (os.path.join(ultralytics_base, 'ultralytics/cfg'), 'ultralytics/cfg'),
    (os.path.join(base_path, 'SuperBirdIDPlugin.lrplugin'), 'SuperBirdIDPlugin.lrplugin'),
    (os.path.join(base_path, 'ioc'), 'ioc'),
]

all_datas.extend(ultralytics_datas)
all_datas.extend(imageio_datas)
all_datas.extend(rawpy_datas)
all_datas.extend(pillow_heif_datas)
all_datas.extend(copy_metadata('imageio'))
all_datas.extend(copy_metadata('rawpy'))
all_datas.extend(copy_metadata('ultralytics'))
all_datas.extend(copy_metadata('pillow_heif'))

a = Analysis(
    ['main.py'],
    pathex=[base_path],
    binaries=[],
    datas=all_datas,
    hiddenimports=[
        'ultralytics',
        'PIL',
        'cv2',
        'numpy',
        'yaml',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'imageio',
        'rawpy',
        'imagehash',
        'pywt',
        'pillow_heif',
        'core',
        'core.burst_detector',
        'core.config_manager',
        'core.exposure_detector',
        'core.file_manager',
        'core.flight_detector',
        'core.focus_point_detector',
        'core.keypoint_detector',
        'core.photo_processor',
        'core.rating_engine',
        'core.source_probe',
        'core.initialization_manager',
        'core.stats_formatter',
        'multiprocessing',
        'multiprocessing.spawn',
        'tools.update_checker',
        'packaging',
        'packaging.version',
        'birdid',
        'birdid.bird_identifier',
        'birdid.ebird_country_filter',
        'birdid_server',
        'server_manager',
        'flask',
        'flask.json',
        'cryptography',
        'cryptography.fernet',
        '_telemetry_build',
        'app_user_stat._telemetry_build',
        'app_user_stat',
        'app_user_stat.telemetry',
        'app_user_stat.consent_texts',
        'app_user_stat.consent_texts.en_US',
        'app_user_stat.consent_texts.zh_CN',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_cv2.py'] if os.path.exists('pyi_rth_cv2.py') else [],
    excludes=[
        'torch', 'torchvision', 'torchaudio', 'timm',
        'PyQt5', 'PyQt6', 'tkinter',
        'polars', 'numba', 'llvmlite', 'pyarrow', 'facexlib', 'datasets',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SuperPickyLite',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(base_path, 'img', 'SuperPicky-V0.02.icns') if os.path.exists(os.path.join(base_path, 'img', 'SuperPicky-V0.02.icns')) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SuperPickyLite',
)

app = BUNDLE(
    coll,
    name='SuperPickyLite.app',
    icon=os.path.join(base_path, 'img', 'SuperPicky-V0.02.icns') if os.path.exists(os.path.join(base_path, 'img', 'SuperPicky-V0.02.icns')) else None,
    bundle_identifier='com.jamesphotography.superpicky.lite',
    info_plist={
        'CFBundleName': 'SuperPickyLite',
        'CFBundleDisplayName': 'SuperPickyLite',
        'CFBundleVersion': APP_VERSION,
        'CFBundleShortVersionString': APP_VERSION,
        'NSHighResolutionCapable': True,
        'NSAppleEventsUsageDescription': '慧眼选鸟需要发送 AppleEvents 与其他应用通信。',
        'NSAppleScriptEnabled': False,
    },
)
