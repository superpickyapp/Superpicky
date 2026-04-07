import argparse
import os
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models


IMAGE_SIZE = 416


class PartLocalizer(nn.Module):
    """Keypoint localization model used by keypoint_detector."""

    def __init__(self, num_parts: int = 3, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        self.num_parts = num_parts
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.coord_head = nn.Linear(hidden_dim // 2, num_parts * 2)
        self.vis_head = nn.Linear(hidden_dim // 2, num_parts)

    def forward(self, x):
        features = self.head(self.backbone(x))
        coords = torch.sigmoid(self.coord_head(features)).view(-1, self.num_parts, 2)
        vis = torch.sigmoid(self.vis_head(features))
        return coords, vis


def build_keypoint_model() -> torch.nn.Module:
    return PartLocalizer(num_parts=3, hidden_dim=512, dropout=0.2)


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

    normalized = {}
    for key, value in state_dict.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def load_keypoint_checkpoint(model_path: str) -> Dict[str, torch.Tensor]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PyTorch model not found: {model_path}")
    if _is_git_lfs_pointer_file(model_path):
        raise RuntimeError(f"Detected Git LFS pointer file, real weights not downloaded: {model_path}")

    try:
        loaded = _torch_load_compat(model_path, map_location="cpu", weights_only=True)
    except Exception as e:
        if _should_retry_without_weights_only(e):
            print("[Keypoint] weights_only=True load failed, retrying with weights_only=False")
            loaded = _torch_load_compat(model_path, map_location="cpu", weights_only=False)
        else:
            raise
    return _extract_state_dict(loaded)


def convert_keypoint_detector(
    pt_path: str = "models/cub200_keypoint_resnet50.pth",
    onnx_path: str = "models/cub200_keypoint_resnet50.onnx",
    opset: int = 13,
    dynamic: bool = True,
) -> str:
    model = build_keypoint_model()
    state_dict = load_keypoint_checkpoint(pt_path)
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "coords": {0: "batch_size"},
            "vis": {0: "batch_size"},
        }

    out_dir = os.path.dirname(os.path.abspath(onnx_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["coords", "vis"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_desc = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    output_desc = [(o.name, o.shape, o.type) for o in session.get_outputs()]
    print(f"[Keypoint] Exported ONNX: {onnx_path}")
    print(f"[Keypoint] Inputs: {input_desc}")
    print(f"[Keypoint] Outputs: {output_desc}")
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


def check_keypoint_consistency(
    pt_path: str = "models/cub200_keypoint_resnet50.pth",
    onnx_path: str = "models/cub200_keypoint_resnet50.onnx",
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

    model = build_keypoint_model()
    model.load_state_dict(load_keypoint_checkpoint(pt_path))
    model = model.to("cpu")
    model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    if len(output_names) != 2:
        raise RuntimeError(f"Expected 2 ONNX outputs (coords, vis), got {len(output_names)}: {output_names}")
    coords_name, vis_name = output_names

    rng = np.random.default_rng(seed)
    failed_indices: List[int] = []
    coords_abs_max: List[float] = []
    coords_abs_mean: List[float] = []
    vis_abs_max: List[float] = []
    vis_abs_mean: List[float] = []

    processed = 0
    while processed < num_samples:
        current_batch = min(batch_size, num_samples - processed)
        batch_input = rng.standard_normal(
            (current_batch, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32
        )

        with torch.no_grad():
            torch_coords, torch_vis = model(torch.from_numpy(batch_input))
            torch_coords = torch_coords.detach().cpu().numpy()
            torch_vis = torch_vis.detach().cpu().numpy()

        onnx_coords, onnx_vis = session.run([coords_name, vis_name], {input_name: batch_input})

        diff_coords = np.abs(torch_coords - onnx_coords)
        diff_vis = np.abs(torch_vis - onnx_vis)

        for i in range(current_batch):
            idx = processed + i
            coords_abs_max.append(float(np.max(diff_coords[i])))
            coords_abs_mean.append(float(np.mean(diff_coords[i])))
            vis_abs_max.append(float(np.max(diff_vis[i])))
            vis_abs_mean.append(float(np.mean(diff_vis[i])))
            try:
                np.testing.assert_allclose(torch_coords[i], onnx_coords[i], rtol=rtol, atol=atol)
                np.testing.assert_allclose(torch_vis[i], onnx_vis[i], rtol=rtol, atol=atol)
            except AssertionError:
                failed_indices.append(idx)

        processed += current_batch

    coords_max_stats = _summary_stats(coords_abs_max)
    coords_mean_stats = _summary_stats(coords_abs_mean)
    vis_max_stats = _summary_stats(vis_abs_max)
    vis_mean_stats = _summary_stats(vis_abs_mean)
    passed = len(failed_indices) == 0

    print(f"[Keypoint][CHECK] Samples: {num_samples}")
    print(
        f"[Keypoint][CHECK] coords_abs_max: max={coords_max_stats['max']:.8f}, "
        f"mean={coords_max_stats['mean']:.8f}, p95={coords_max_stats['p95']:.8f}"
    )
    print(
        f"[Keypoint][CHECK] coords_abs_mean: max={coords_mean_stats['max']:.8f}, "
        f"mean={coords_mean_stats['mean']:.8f}, p95={coords_mean_stats['p95']:.8f}"
    )
    print(
        f"[Keypoint][CHECK] vis_abs_max: max={vis_max_stats['max']:.8f}, "
        f"mean={vis_max_stats['mean']:.8f}, p95={vis_max_stats['p95']:.8f}"
    )
    print(
        f"[Keypoint][CHECK] vis_abs_mean: max={vis_mean_stats['max']:.8f}, "
        f"mean={vis_mean_stats['mean']:.8f}, p95={vis_mean_stats['p95']:.8f}"
    )
    print(f"[Keypoint][CHECK] Result: {'PASS' if passed else 'FAIL'}")
    if failed_indices:
        print(f"[Keypoint][CHECK] Failed sample indices: {failed_indices}")

    return {
        "passed": passed,
        "samples": num_samples,
        "failed_indices": failed_indices,
        "coords_abs_max": coords_max_stats,
        "coords_abs_mean": coords_mean_stats,
        "vis_abs_max": vis_max_stats,
        "vis_abs_mean": vis_mean_stats,
        "rtol": rtol,
        "atol": atol,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert cub200 keypoint detector PTH to ONNX and optionally run consistency checks."
    )
    parser.add_argument("--pt", default="models/cub200_keypoint_resnet50.pth", help="Path to .pth model")
    parser.add_argument("--onnx", default="models/cub200_keypoint_resnet50.onnx", help="Path to output .onnx model")
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
        convert_keypoint_detector(
            pt_path=args.pt,
            onnx_path=args.onnx,
            opset=args.opset,
            dynamic=args.dynamic,
        )

    if args.check:
        check_keypoint_consistency(
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
