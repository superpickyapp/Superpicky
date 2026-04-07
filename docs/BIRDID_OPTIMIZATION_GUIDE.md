# 🐦 BirdID 识别模型优化指南

> 本文档总结了 SuperPicky/SuperBirdID 项目中鸟类识别模型的优化参数和最佳实践，供 CoreML 转换团队参考。

---

## 📋 模型概述

| 属性 | 值 |
|------|-----|
| **模型名称** | birdid2024.pt |
| **模型格式** | PyTorch (.pt) |
| **输入尺寸** | 224×224 RGB |
| **输出** | 11,000+ 类别的 logits |
| **架构** | 自定义 CNN 分类器 |

---

## 🎯 核心优化参数

### 1. 温度缩放 (Temperature Scaling)

```python
TEMPERATURE = 0.5
probs = softmax(logits / TEMPERATURE)
```

| 参数 | 原值 | 优化值 | 说明 |
|-----|------|--------|-----|
| Temperature | 0.6 | **0.5** | 降低温度使概率分布更"尖锐"，提高 top-1 置信度 |

**效果**: 置信度更集中，减少模糊预测

---

### 2. YOLO 检测裁剪边距 (Padding)

```python
padding = 150  # 像素
```

| 参数 | 原值 | 优化值 | 说明 |
|-----|------|--------|-----|
| Padding | 20 | **150** | 裁剪时在鸟周围保留更多环境上下文 |

**效果**: 减少截断翅膀/尾巴，提供更完整的鸟体特征

---

### 3. 多增强融合 (Multi-Enhancement Fusion)

```python
ENHANCEMENT_METHODS = [
    "none",              # 原图
    "edge_enhance_more", # PIL EDGE_ENHANCE_MORE 滤波
    "unsharp_mask",      # PIL UnsharpMask 锐化
    "contrast_edge",     # 亮度1.2 + 对比度1.3 + 边缘增强
    "desaturate"         # 饱和度降至50%
]
```

**融合策略**:
```python
# 对每种增强方法运行推理，收集原始 logits
all_logits = []
for method in ENHANCEMENT_METHODS:
    enhanced = apply_enhancement(image, method)
    logits = model(preprocess(enhanced))
    all_logits.append(logits)

# 取所有 logits 的平均值
fused_logits = torch.stack(all_logits).mean(dim=0)

# 对融合后的 logits 应用温度缩放和 softmax
probs = softmax(fused_logits / TEMPERATURE)
```

**效果**: 多视角融合提高鲁棒性，减少单一增强方法的偏差

---

## 🖼️ 图像预处理流程

### 步骤 1: 加载与裁剪
```python
# RAW 文件处理
if is_raw_file(path):
    # 优先提取内嵌 JPEG 预览 (快速)
    preview = extract_embedded_preview(path)
    if preview is None:
        # 回退到 rawpy 半尺寸渲染
        raw = rawpy.imread(path)
        image = raw.postprocess(half_size=True)
```

### 步骤 2: YOLO 鸟类检测
```python
# 使用 YOLO11l-seg 模型检测鸟类
yolo_model = YOLO("yolo11l-seg.pt")
results = yolo_model(image, conf=0.25)

# 选择最大置信度的鸟类检测框
best_box = max(bird_detections, key=lambda x: x.confidence)

# 带 padding 裁剪
x1, y1, x2, y2 = best_box
x1 = max(0, x1 - padding)
y1 = max(0, y1 - padding)
x2 = min(width, x2 + padding)
y2 = min(height, y2 + padding)
cropped = image.crop((x1, y1, x2, y2))
```

### 步骤 3: 智能缩放为 224×224
```python
def smart_resize(image, target_size=224):
    # 保持宽高比缩放到 256
    w, h = image.size
    if w < h:
        new_w = 256
        new_h = int(256 * h / w)
    else:
        new_h = 256
        new_w = int(256 * w / h)
    
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    
    # 中心裁剪到 224×224
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    return resized.crop((left, top, left + target_size, top + target_size))
```

### 步骤 4: 归一化
```python
# 注意: 使用 BGR 通道顺序 (非 RGB)
bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

# ImageNet 归一化参数 (BGR 顺序)
mean = [0.406, 0.456, 0.485]  # B, G, R
std = [0.225, 0.224, 0.229]   # B, G, R

normalized = (bgr_array / 255.0 - mean) / std
```

⚠️ **重要**: 归一化使用 **BGR** 通道顺序，不是 RGB！

---

## 📊 优化效果对比

| 测试案例 | 优化前 | 优化后 |
|---------|--------|--------|
| 灰头丛鹟 | 排名8, 0.98% | **Top-1, 99.92%** |
| 利氏吸蜜鸟 | 排名80, 0.05% | **Top-1, 92.37%** |
| 小掩鼻风鸟 | 排名6, 1.64% | **Top-1, 70.00%** |
| 黄斑吸蜜鸟 | 排名2, 19.53% | **Top-1, 79.96%** |

---

## 🌍 eBird 地理过滤

### 过滤策略
```python
# 1. 获取地理区域代码
region_code = get_region_from_gps(latitude, longitude)
# 例如: "AU-QLD" (澳大利亚昆士兰)

# 2. 加载该区域的物种列表
species_set = load_ebird_species(region_code)

# 3. 过滤识别结果
filtered_results = [
    r for r in results
    if r['ebird_code'] in species_set
]
```

### 离线地区检测
```python
# 预定义边界用于快速离线检测
REGION_BOUNDARIES = {
    "AU": {"lat": (-44, -10), "lon": (113, 154)},
    "CN": {"lat": (18, 54), "lon": (73, 135)},
    "US": {"lat": (24, 50), "lon": (-125, -66)},
    # ...
}
```

---

## 💡 CoreML 转换建议

### 1. 输入规格
- **输入名称**: `input`
- **输入形状**: `[1, 3, 224, 224]` (NCHW)
- **数据类型**: Float32
- **归一化**: 在模型内部或预处理中实现

### 2. 预处理注意事项
```swift
// Swift 预处理示例
let resizedImage = smartResize(inputImage, to: 224)

// 转换为 BGR 并归一化
var pixelBuffer = createPixelBuffer(resizedImage)
applyBGRNormalization(pixelBuffer,
    mean: [0.406, 0.456, 0.485],
    std: [0.225, 0.224, 0.229])
```

### 3. 后处理
```swift
// 温度缩放
let temperature: Float = 0.5
let scaledLogits = logits.map { $0 / temperature }

// Softmax
let probs = softmax(scaledLogits)

// 获取 top-k
let topK = probs.enumerated()
    .sorted { $0.element > $1.element }
    .prefix(5)
```

### 4. 多增强融合 (可选)
如果设备性能允许，可以实现简化版的多增强融合：
- 只使用 2-3 种增强方法
- 或者在服务端进行完整融合

---

## 📁 相关文件

| 文件 | 用途 |
|-----|------|
| `birdid/bird_identifier.py` | 核心识别逻辑 |
| `birdid/data/birdinfo.json` | 鸟种信息映射 (class_id → 名称) |
| `birdid/data/bird_reference.sqlite` | eBird 物种数据库 |
| `birdid/ebird_country_filter.py` | 地理过滤逻辑 |

---

## 🔧 调试建议

1. **对比测试**: 使用相同图片在 PyTorch 和 CoreML 上对比 top-10 结果
2. **检查归一化**: 确认 BGR 顺序和参数正确
3. **验证尺寸**: 确保输入为 224×224，缩放使用 LANCZOS
4. **温度参数**: 0.5 是关键，不要遗漏

---

*文档生成时间: 2026-01-18*
*SuperPicky V4.0.0*
