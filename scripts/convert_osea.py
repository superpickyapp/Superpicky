import torch
import onnxruntime as ort
import numpy as np
from torchvision import models


def _torch_load_compat(path: str, *, map_location: str, weights_only: bool):
	"""torch.load wrapper that works across PyTorch versions."""
	try:
		return torch.load(path, map_location=map_location, weights_only=weights_only)
	except TypeError:
		# Older PyTorch does not support weights_only
		return torch.load(path, map_location=map_location)


def _should_retry_without_weights_only(error: Exception) -> bool:
	message = str(error)
	return (
		"weights_only" in message
		or "Weights only load failed" in message
		or "WeightsUnpickler" in message
	)


def _extract_state_dict(loaded_obj):
	if isinstance(loaded_obj, dict):
		if "state_dict" in loaded_obj:
			return loaded_obj["state_dict"]
		if "model_state_dict" in loaded_obj:
			return loaded_obj["model_state_dict"]
		return loaded_obj
	if isinstance(loaded_obj, torch.nn.Module):
		return loaded_obj.state_dict()
	raise TypeError(f"不支持的模型格式: {type(loaded_obj)}")


def _is_git_lfs_pointer_file(file_path: str) -> bool:
	"""Detect Git LFS pointer files to avoid cryptic torch pickle errors."""
	try:
		with open(file_path, "rb") as f:
			header = f.read(256)
	except OSError:
		return False
	return header.startswith(b"version https://git-lfs.github.com/spec/")


def _load_osea_checkpoint(model_path: str):
	if _is_git_lfs_pointer_file(model_path):
		raise RuntimeError(f"检测到 Git LFS 指针文件（未下载实际模型权重）: {model_path}")
	try:
		return _torch_load_compat(model_path, map_location="cpu", weights_only=True)
	except Exception as e:
		if _should_retry_without_weights_only(e):
			print("[OSEA] weights_only=True 加载失败，回退 weights_only=False（仅限可信模型）")
			return _torch_load_compat(model_path, map_location="cpu", weights_only=False)
		raise

def convert_osea(pytorch_model_path, onnx_model_path):

	"""
		导出成onnx
	"""

	loaded = _load_osea_checkpoint(pytorch_model_path)
	model = models.resnet34(num_classes=11000)
	state_dict = _extract_state_dict(loaded)
	model.load_state_dict(state_dict)
	model = model.to('cpu')
	model.eval()
	
	torch.onnx.export(
		model,
		torch.randn(1, 3, 224, 224),
		onnx_model_path,
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={"input": {0: 'batch_size'}, "output": {0: 'batch_size'}},
		opset_version=12,
		do_constant_folding=True
	)


def check_osea(pytorch_model_path, onnx_model_path):
    """
    验证两者是否一致
    """

    # 1. 准备相同的一组随机输入数据
    # 注意：维度必须与导出时的 dummy_input 一致
    input_shape = (1, 3, 224, 224) 
    dummy_input = torch.randn(*input_shape)
    # 转为 numpy 格式供 ONNX 使用
    dummy_input_numpy = dummy_input.detach().cpu().numpy()

    # 2. 运行 PyTorch 模型获取输出
    model = models.resnet34(num_classes=11000)
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        torch_output = model(dummy_input)

    # 3. 运行 ONNX 模型获取输出
    # 创建推理会话
    session = ort.InferenceSession(onnx_model_path)
    # 获取输入节点名称 (通常是 'input')
    input_name = session.get_inputs()[0].name
    # 进行推理
    onnx_output = session.run(None, {input_name: dummy_input_numpy})

    # 4. 比较差异
    # onnx_output 是个列表，通常取第一个元素
    torch_out_numpy = torch_output.detach().cpu().numpy()
    diff = np.abs(torch_out_numpy - onnx_output[0])

    print(f"最大绝对误差 (Max Abs Diff): {np.max(diff)}")
    print(f"平均绝对误差 (Mean Abs Diff): {np.mean(diff)}")

    # 5. 设置阈值验证
    try:
        np.testing.assert_allclose(torch_out_numpy, onnx_output[0], rtol=1e-03, atol=1e-05)
        print("✅ 验证通过！两个模型的输出高度一致。")
    except AssertionError as e:
        print("❌ 验证失败！误差过大，请检查算子或 opset 版本。")


if __name__ == "__main__":
	convert_osea("models/model20240824.pth", "models/model20240824.onnx")
	check_osea("models/model20240824.pth", "models/model20240824.onnx")
