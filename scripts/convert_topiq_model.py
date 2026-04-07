import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from topiq_model_onnx import CFANet  # noqa: E402


IMAGE_SIZE = 384


class TopiqOnnxWrapper(nn.Module):
    """Wrap CFANet to force tuple outputs for ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mos, dist = self.model(x, return_mos=True, return_dist=True)
        return mos, dist


def build_topiq_model() -> torch.nn.Module:
    return CFANet()


def _torch_load_compat(path: str, *, map_location: str, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _should_retry_without_weights_only(error: Exception) -> bool:
    message = str(error)
    return (
        "weights_only" in message
        or "Weights only load failed" in message
        or "WeightsUnpickler" in message
    )


def _is_git_lfs_pointer_file(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            header = f.read(256)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/")


def _extract_state_dict(loaded_obj) -> Dict[str, torch.Tensor]:
    if isinstance(loaded_obj, dict):
        if "params" in loaded_obj and isinstance(loaded_obj["params"], dict):
            state_dict = loaded_obj["params"]
        elif "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            state_dict = loaded_obj["state_dict"]
        elif "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
            state_dict = loaded_obj["model_state_dict"]
        else:
            state_dict = loaded_obj
    elif isinstance(loaded_obj, torch.nn.Module):
        state_dict = loaded_obj.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(loaded_obj)}")

    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def load_topiq_checkpoint(model_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch model not found: {model_path}")
    if _is_git_lfs_pointer_file(model_path):
        raise RuntimeError(f"Detected Git LFS pointer file, real weights not downloaded: {model_path}")

    try:
        loaded = _torch_load_compat(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if _should_retry_without_weights_only(e):
            print("[TOPIQ] weights_only=True load failed, retrying with weights_only=False")
            loaded = _torch_load_compat(model_path, map_location="cpu", weights_only=False)
        else:
            raise

    return _extract_state_dict(loaded)


def convert_topiq_model(
    pt_path: str = "models/cfanet_iaa_ava_res50-3cd62bb3.pth",
    onnx_path: str = "models/cfanet_iaa_ava_res50.onnx",
    opset: int = 17,
    image_size: int = IMAGE_SIZE,
) -> str:
    model = build_topiq_model()
    state_dict = load_topiq_checkpoint(pt_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[TOPIQ] Missing keys while loading: {len(missing)}")
    if unexpected:
        print(f"[TOPIQ] Unexpected keys while loading: {len(unexpected)}")

    model = model.to("cpu")
    model.eval()
    wrapper = TopiqOnnxWrapper(model).to("cpu").eval()

    dummy_input = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "mos": {0: "batch_size"},
        "dist": {0: "batch_size"},
    }

    out_dir = os.path.dirname(os.path.abspath(onnx_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["mos", "dist"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_desc = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    output_desc = [(o.name, o.shape, o.type) for o in session.get_outputs()]
    print(f"[TOPIQ] Exported ONNX: {onnx_path}")
    print(f"[TOPIQ] Inputs: {input_desc}")
    print(f"[TOPIQ] Outputs: {output_desc}")
    return onnx_path


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def check_topiq_consistency(
    pt_path: str = "models/cfanet_iaa_ava_res50-3cd62bb3.pth",
    onnx_path: str = "models/cfanet_iaa_ava_res50.onnx",
    num_samples: int = 20,
    batch_size: int = 1,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, object]:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model not found: {pt_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = build_topiq_model()
    state_dict = load_topiq_checkpoint(pt_path)
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cpu")
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    if len(output_names) != 2:
        raise RuntimeError(f"Expected 2 ONNX outputs (mos, dist), got {len(output_names)}: {output_names}")
    mos_name, dist_name = output_names

    rng = np.random.default_rng(seed)
    failed_indices: List[int] = []
    mos_abs_max: List[float] = []
    mos_abs_mean: List[float] = []
    dist_abs_max: List[float] = []
    dist_abs_mean: List[float] = []

    processed = 0
    while processed < num_samples:
        current_batch = min(batch_size, num_samples - processed)
        batch_input = rng.standard_normal(
            (current_batch, 3, image_size, image_size), dtype=np.float32
        )

        with torch.no_grad():
            torch_mos, torch_dist = model(
                torch.from_numpy(batch_input), return_mos=True, return_dist=True
            )
            torch_mos = torch_mos.detach().cpu().numpy()
            torch_dist = torch_dist.detach().cpu().numpy()

        onnx_mos, onnx_dist = session.run([mos_name, dist_name], {input_name: batch_input})

        diff_mos = np.abs(torch_mos - onnx_mos)
        diff_dist = np.abs(torch_dist - onnx_dist)

        for i in range(current_batch):
            idx = processed + i
            mos_abs_max.append(float(np.max(diff_mos[i])))
            mos_abs_mean.append(float(np.mean(diff_mos[i])))
            dist_abs_max.append(float(np.max(diff_dist[i])))
            dist_abs_mean.append(float(np.mean(diff_dist[i])))
            try:
                np.testing.assert_allclose(torch_mos[i], onnx_mos[i], rtol=rtol, atol=atol)
                np.testing.assert_allclose(torch_dist[i], onnx_dist[i], rtol=rtol, atol=atol)
            except AssertionError:
                failed_indices.append(idx)

        processed += current_batch

    mos_max_stats = _summary_stats(mos_abs_max)
    mos_mean_stats = _summary_stats(mos_abs_mean)
    dist_max_stats = _summary_stats(dist_abs_max)
    dist_mean_stats = _summary_stats(dist_abs_mean)
    passed = len(failed_indices) == 0

    print(f"[TOPIQ][CHECK] Samples: {num_samples}")
    print(
        f"[TOPIQ][CHECK] mos_abs_max: max={mos_max_stats['max']:.8f}, "
        f"mean={mos_max_stats['mean']:.8f}, p95={mos_max_stats['p95']:.8f}"
    )
    print(
        f"[TOPIQ][CHECK] mos_abs_mean: max={mos_mean_stats['max']:.8f}, "
        f"mean={mos_mean_stats['mean']:.8f}, p95={mos_mean_stats['p95']:.8f}"
    )
    print(
        f"[TOPIQ][CHECK] dist_abs_max: max={dist_max_stats['max']:.8f}, "
        f"mean={dist_max_stats['mean']:.8f}, p95={dist_max_stats['p95']:.8f}"
    )
    print(
        f"[TOPIQ][CHECK] dist_abs_mean: max={dist_mean_stats['max']:.8f}, "
        f"mean={dist_mean_stats['mean']:.8f}, p95={dist_mean_stats['p95']:.8f}"
    )
    print(f"[TOPIQ][CHECK] Result: {'PASS' if passed else 'FAIL'}")
    if failed_indices:
        print(f"[TOPIQ][CHECK] Failed sample indices: {failed_indices}")

    return {
        "passed": passed,
        "samples": num_samples,
        "failed_indices": failed_indices,
        "mos_abs_max": mos_max_stats,
        "mos_abs_mean": mos_mean_stats,
        "dist_abs_max": dist_max_stats,
        "dist_abs_mean": dist_mean_stats,
        "rtol": rtol,
        "atol": atol,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TOPIQ CFANet PTH to ONNX and optionally run consistency checks."
    )
    parser.add_argument("--pt", default="models/cfanet_iaa_ava_res50-3cd62bb3.pth", help="Path to .pth model")
    parser.add_argument("--onnx", default="models/cfanet_iaa_ava_res50.onnx", help="Path to output .onnx model")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Model input H/W (square)")
    parser.add_argument("--convert", dest="convert", action="store_true", default=True, help="Run ONNX export (default: on)")
    parser.add_argument("--no-convert", dest="convert", action="store_false", help="Skip ONNX export")
    parser.add_argument("--check", action="store_true", help="Run PT vs ONNX consistency check")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of random samples for consistency check")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for consistency check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generated inputs")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for allclose")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for allclose")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.convert:
        convert_topiq_model(
            pt_path=args.pt,
            onnx_path=args.onnx,
            opset=args.opset,
            image_size=args.image_size,
        )

    if args.check:
        check_topiq_consistency(
            pt_path=args.pt,
            onnx_path=args.onnx,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            image_size=args.image_size,
            rtol=args.rtol,
            atol=args.atol,
        )


if __name__ == "__main__":
    main()
