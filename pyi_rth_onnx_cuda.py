import os
import sys
from pathlib import Path

SYSTEM_CUDA_OPT_IN = "SUPERPICKY_USE_SYSTEM_CUDA"


def _add_runtime_dir(path: Path) -> None:
    if not path.exists():
        return

    resolved = str(path.resolve())
    try:
        os.add_dll_directory(resolved)
    except (AttributeError, FileNotFoundError, OSError):
        pass

    current_path = os.environ.get("PATH", "")
    parts = current_path.split(os.pathsep) if current_path else []
    if resolved not in parts:
        os.environ["PATH"] = resolved + (os.pathsep + current_path if current_path else "")


def _setup_onnx_cuda_runtime() -> None:
    if sys.platform != "win32":
        return

    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    runtime_dirs = [
        base_dir / "gpu_runtime",
        base_dir / "onnxruntime" / "capi",
    ]

    use_system_cuda = os.environ.get(SYSTEM_CUDA_OPT_IN, "").strip().lower() in {"1", "true", "yes"}
    if use_system_cuda:
        cuda_path = os.environ.get("CUDA_PATH", "").strip()
        if cuda_path:
            runtime_dirs.append(Path(cuda_path) / "bin")

    for runtime_dir in runtime_dirs:
        _add_runtime_dir(runtime_dir)


_setup_onnx_cuda_runtime()
