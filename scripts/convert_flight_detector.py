import argparse
import os
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models


IMAGE_SIZE = 384


def build_flight_model() -> torch.nn.Module:
    """Build the EfficientNet-B3 classifier used by FlightDetector."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 1),
        nn.Sigmoid(),
    )
    return model


def _torch_load_compat(path: str, *, map_location: str, weights_only: bool):
    """torch.load wrapper that works across PyTorch versions."""
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
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            state_dict = loaded_obj["state_dict"]
        elif "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
            state_dict = loaded_obj["model_state_dict"]
        else:
            state_dict = loaded_obj
    elif isinstance(loaded_obj, torch.nn.Module):
        state_dict = loaded_obj.state_dict()
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(loaded_obj)}")

    # Handle DataParallel checkpoints.
    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def load_efficientnet_checkpoint(model_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch model not found: {model_path}")
    if _is_git_lfs_pointer_file(model_path):
        raise RuntimeError(f"Detected Git LFS pointer file, real weights not downloaded: {model_path}")

    try:
        loaded = _torch_load_compat(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if _should_retry_without_weights_only(e):
            print("[EfficientNet] weights_only=True load failed, retrying with weights_only=False")
            loaded = _torch_load_compat(model_path, map_location="cpu", weights_only=False)
        else:
            raise

    return _extract_state_dict(loaded)


def convert_efficientnet(
    pt_path: str = "models/superFlier_efficientnet.pth",
    onnx_path: str = "models/superFlier_efficientnet.onnx",
    opset: int = 13,
    dynamic: bool = True,
) -> str:
    model = build_flight_model()
    state_dict = load_efficientnet_checkpoint(pt_path)
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    out_dir = os.path.dirname(os.path.abspath(onnx_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_desc = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    output_desc = [(o.name, o.shape, o.type) for o in session.get_outputs()]
    print(f"[EfficientNet] Exported ONNX: {onnx_path}")
    print(f"[EfficientNet] Inputs: {input_desc}")
    print(f"[EfficientNet] Outputs: {output_desc}")
    return onnx_path


def check_efficientnet_consistency(
    pt_path: str = "models/superFlier_efficientnet.pth",
    onnx_path: str = "models/superFlier_efficientnet.onnx",
    num_samples: int = 20,
    batch_size: int = 1,
    seed: int = 42,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, object]:
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model not found: {pt_path}")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = build_flight_model()
    model.load_state_dict(load_efficientnet_checkpoint(pt_path))
    model = model.to("cpu")
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    rng = np.random.default_rng(seed)
    sample_abs_max: List[float] = []
    sample_abs_mean: List[float] = []
    failed_indices: List[int] = []

    processed = 0
    while processed < num_samples:
        current_batch = min(batch_size, num_samples - processed)
        batch_input = rng.standard_normal(
            (current_batch, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )

        with torch.no_grad():
            torch_out = model(torch.from_numpy(batch_input)).detach().cpu().numpy()
        onnx_out = session.run([output_name], {input_name: batch_input})[0]

        diff = np.abs(torch_out - onnx_out)

        for i in range(current_batch):
            idx = processed + i
            cur_diff = diff[i]
            sample_abs_max.append(float(np.max(cur_diff)))
            sample_abs_mean.append(float(np.mean(cur_diff)))
            try:
                np.testing.assert_allclose(torch_out[i], onnx_out[i], rtol=rtol, atol=atol)
            except AssertionError:
                failed_indices.append(idx)

        processed += current_batch

    max_abs_diff = float(np.max(sample_abs_max)) if sample_abs_max else 0.0
    mean_abs_diff = float(np.mean(sample_abs_mean)) if sample_abs_mean else 0.0
    p95_abs_diff = float(np.percentile(np.asarray(sample_abs_max, dtype=np.float32), 95)) if sample_abs_max else 0.0
    passed = len(failed_indices) == 0

    print(f"[EfficientNet][CHECK] Samples: {num_samples}")
    print(f"[EfficientNet][CHECK] max_abs_diff={max_abs_diff:.8f}")
    print(f"[EfficientNet][CHECK] mean_abs_diff={mean_abs_diff:.8f}")
    print(f"[EfficientNet][CHECK] p95_abs_diff={p95_abs_diff:.8f}")
    print(f"[EfficientNet][CHECK] Result: {'PASS' if passed else 'FAIL'}")
    if failed_indices:
        print(f"[EfficientNet][CHECK] Failed sample indices: {failed_indices}")

    return {
        "passed": passed,
        "samples": num_samples,
        "failed_indices": failed_indices,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "p95_abs_diff": p95_abs_diff,
        "rtol": rtol,
        "atol": atol,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert superFlier EfficientNet PTH to ONNX and optionally run consistency checks."
    )
    parser.add_argument("--pt", default="models/superFlier_efficientnet.pth", help="Path to .pth model")
    parser.add_argument("--onnx", default="models/superFlier_efficientnet.onnx", help="Path to output .onnx model")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--convert", dest="convert", action="store_true", default=True, help="Run ONNX export (default: on)")
    parser.add_argument("--no-convert", dest="convert", action="store_false", help="Skip ONNX export")
    parser.add_argument("--check", action="store_true", help="Run PT vs ONNX consistency check")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of random samples for consistency check")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for consistency check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generated inputs")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for allclose")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for allclose")
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", default=True, help="Export with dynamic batch axis (default: on)")
    parser.add_argument("--static", dest="dynamic", action="store_false", help="Export static batch axis")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.convert:
        convert_efficientnet(
            pt_path=args.pt,
            onnx_path=args.onnx,
            opset=args.opset,
            dynamic=args.dynamic,
        )

    if args.check:
        check_efficientnet_consistency(
            pt_path=args.pt,
            onnx_path=args.onnx,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
        )


if __name__ == "__main__":
    main()
