# 🌐 SuperBirdID API 服务使用说明

## 概述

SuperBirdID API 提供了HTTP REST API接口，允许外部程序（如Adobe Lightroom插件、批处理脚本等）调用鸟类识别功能。

---

## 快速开始

### 1. 启动API服务器

```bash
# 基本启动（监听 127.0.0.1:5156）
python SuperBirdID_API.py

# 自定义端口
python SuperBirdID_API.py --port 8000

# 允许外部访问
python SuperBirdID_API.py --host 0.0.0.0 --port 5156

# 调试模式
python SuperBirdID_API.py --debug
```

### 2. 验证服务器状态

```bash
curl http://127.0.0.1:5156/health
```

返回示例:
```json
{
  "status": "ok",
  "service": "SuperBirdID API",
  "version": "1.0.0",
  "yolo_available": true,
  "ebird_available": true
}
```

---

## API接口文档

### 1. 健康检查

**端点**: `GET /health`

**返回**:
```json
{
  "status": "ok",
  "service": "SuperBirdID API",
  "version": "1.0.0",
  "yolo_available": true,
  "ebird_available": true
}
```

---

### 2. 识别鸟类

**端点**: `POST /recognize`

**请求体**:
```json
{
  "image_path": "/path/to/image.jpg",    // 选项1: 图片路径
  "image_base64": "base64_encoded...",   // 选项2: Base64编码图片
  "use_yolo": true,                      // 可选，默认true
  "use_gps": true,                       // 可选，默认true
  "top_k": 3                             // 可选，默认3
}
```

**返回**:
```json
{
  "success": true,
  "results": [
    {
      "rank": 1,
      "cn_name": "白头鹎",
      "en_name": "Light-vented Bulbul",
      "scientific_name": "Pycnonotus sinensis",
      "confidence": 95.5,
      "ebird_match": true
    },
    {
      "rank": 2,
      "cn_name": "白喉红臀鹎",
      "en_name": "Sooty-headed Bulbul",
      "scientific_name": "Pycnonotus aurigaster",
      "confidence": 3.2,
      "ebird_match": false
    }
  ],
  "yolo_info": "YOLO检测: 1个目标, 置信度0.95",
  "gps_info": {
    "latitude": 39.123,
    "longitude": 116.456,
    "region": "中国",
    "info": "GPS: 39.123000, 116.456000 (ExifTool)"
  }
}
```

**示例**:

```bash
# 使用curl
curl -X POST http://127.0.0.1:5156/recognize \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/bird.jpg", "top_k": 3}'

# 使用Python requests
import requests
response = requests.post(
    "http://127.0.0.1:5156/recognize",
    json={"image_path": "/path/to/bird.jpg", "top_k": 3}
)
print(response.json())
```

---

### 3. 获取鸟种详细信息

**端点**: `GET /bird/info`

**参数**:
- `cn_name`: 中文名称（必需）

**返回**:
```json
{
  "success": true,
  "info": {
    "cn_name": "白头鹎",
    "en_name": "Light-vented Bulbul",
    "scientific_name": "Pycnonotus sinensis",
    "short_description": "白头鹎，是雀形目鹎科鹎属的鸟类...",
    "full_description": "外形特征：成鸟具有鲜明的头部图案...",
    "ebird_code": "livbul1"
  }
}
```

**示例**:
```bash
curl "http://127.0.0.1:5156/bird/info?cn_name=白头鹎"
```

---

### 4. 写入EXIF Title

**端点**: `POST /exif/write-title`

**请求体**:
```json
{
  "image_path": "/path/to/image.jpg",
  "bird_name": "白头鹎"
}
```

**返回**:
```json
{
  "success": true,
  "message": "✓ 已写入EXIF Title: 白头鹎"
}
```

**示例**:
```bash
curl -X POST http://127.0.0.1:5156/exif/write-title \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/bird.jpg", "bird_name": "白头鹎"}'
```

---

### 5. 写入EXIF Caption

**端点**: `POST /exif/write-caption`

**请求体**:
```json
{
  "image_path": "/path/to/image.jpg",
  "caption": "白头鹎，是雀形目鹎科鹎属的鸟类..."
}
```

**返回**:
```json
{
  "success": true,
  "message": "✓ 已写入EXIF Caption"
}
```

---

## 使用场景

### 1. Adobe Lightroom 插件

Lightroom可以通过Lua脚本调用外部HTTP API。示例伪代码：

```lua
-- Lightroom插件示例 (Lua)
local http = require("LrHttp")
local json = require("json")

function recognizeBird(imagePath)
  local url = "http://127.0.0.1:5156/recognize"
  local body = json.encode({
    image_path = imagePath,
    top_k = 3
  })

  local result, headers = http.post(url, body)
  local data = json.decode(result)

  if data.success then
    local topResult = data.results[1]
    -- 写入EXIF Title
    http.post("http://127.0.0.1:5156/exif/write-title",
      json.encode({
        image_path = imagePath,
        bird_name = topResult.cn_name
      })
    )
    return topResult.cn_name
  end
end
```

### 2. Python批处理脚本

```python
import requests
import glob

# 批量识别目录下所有图片
for image_path in glob.glob("/path/to/photos/*.jpg"):
    response = requests.post(
        "http://127.0.0.1:5156/recognize",
        json={"image_path": image_path, "top_k": 1}
    )

    result = response.json()
    if result['success']:
        bird_name = result['results'][0]['cn_name']
        print(f"{image_path} -> {bird_name}")

        # 自动写入EXIF
        requests.post(
            "http://127.0.0.1:5156/exif/write-title",
            json={"image_path": image_path, "bird_name": bird_name}
        )
```

### 3. JavaScript/Node.js

```javascript
const axios = require('axios');

async function recognizeBird(imagePath) {
  const response = await axios.post('http://127.0.0.1:5156/recognize', {
    image_path: imagePath,
    top_k: 3
  });

  if (response.data.success) {
    const results = response.data.results;
    console.log('识别结果:', results);
    return results;
  }
}

recognizeBird('/path/to/bird.jpg');
```

### 4. Shell脚本

```bash
#!/bin/bash

# 识别鸟类
IMAGE_PATH="/path/to/bird.jpg"

RESULT=$(curl -s -X POST http://127.0.0.1:5156/recognize \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$IMAGE_PATH\", \"top_k\": 1}")

# 提取鸟种名称（使用jq解析JSON）
BIRD_NAME=$(echo $RESULT | jq -r '.results[0].cn_name')

echo "识别结果: $BIRD_NAME"

# 写入EXIF
curl -X POST http://127.0.0.1:5156/exif/write-title \
  -H "Content-Type: application/json" \
  -d "{\"image_path\": \"$IMAGE_PATH\", \"bird_name\": \"$BIRD_NAME\"}"
```

---

## 客户端示例

我们提供了完整的Python客户端示例：`api_client_example.py`

### 使用方法

```bash
# 识别单张图片并写入EXIF
python api_client_example.py /path/to/bird.jpg
```

### 主要功能

- ✅ 健康检查
- ✅ 识别鸟类（支持路径和Base64）
- ✅ 获取鸟种详细信息
- ✅ 写入EXIF Title和Caption
- ✅ 完整工作流程演示

---

## 部署选项

### 1. 本地运行（默认）

```bash
# 仅本机访问
python SuperBirdID_API.py --host 127.0.0.1 --port 5156
```

### 2. 局域网访问

```bash
# 允许局域网内其他设备访问
python SuperBirdID_API.py --host 0.0.0.0 --port 5156
```

其他设备访问地址：`http://<你的IP>:5156`

### 3. 后台运行（macOS/Linux）

```bash
# 使用nohup在后台运行
nohup python SuperBirdID_API.py --host 127.0.0.1 --port 5156 > api.log 2>&1 &

# 查看日志
tail -f api.log

# 停止服务
pkill -f SuperBirdID_API.py
```

### 4. 使用systemd（Linux生产环境）

创建服务文件 `/etc/systemd/system/superbird-api.service`:

```ini
[Unit]
Description=SuperBirdID API Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/SuperBirdID
ExecStart=/usr/bin/python3 SuperBirdID_API.py --host 127.0.0.1 --port 5156
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务:
```bash
sudo systemctl enable superbird-api
sudo systemctl start superbird-api
sudo systemctl status superbird-api
```

---

## 性能优化

### 1. 模型预加载

API服务器启动时会自动预加载所有模型，避免首次请求时的加载延迟。

### 2. 多线程支持

Flask服务器默认启用多线程模式（`threaded=True`），支持并发请求。

### 3. 缓存建议

对于频繁访问的鸟种信息，建议在客户端实现缓存机制。

---

## 错误处理

API返回的错误格式：

```json
{
  "success": false,
  "error": "错误描述",
  "traceback": "详细错误堆栈（仅调试模式）"
}
```

常见错误码：
- `400 Bad Request`: 请求参数错误
- `404 Not Found`: 资源未找到（如图片文件、鸟种信息）
- `500 Internal Server Error`: 服务器内部错误

---

## 安全建议

### 1. 本地开发

默认配置（`127.0.0.1:5000`）仅允许本机访问，安全性最高。

### 2. 局域网部署

如需局域网访问，建议：
- 使用防火墙限制访问IP
- 添加API密钥认证（需自行实现）
- 使用HTTPS（需配置SSL证书）

### 3. 生产环境

不建议直接将Flask开发服务器暴露到公网。如需公网访问，请：
- 使用Gunicorn/uWSGI作为WSGI服务器
- 使用Nginx作为反向代理
- 配置HTTPS和速率限制
- 实现用户认证和授权

---

## 依赖要求

```bash
pip install flask flask-cors requests
```

已安装版本：
- Flask 3.1.2
- flask-cors 6.0.1
- requests (已随conda安装)

---

## 故障排除

### 问题1: 端口被占用

```
OSError: [Errno 48] Address already in use
```

**解决方案**: 更换端口
```bash
python SuperBirdID_API.py --port 5001
```

### 问题2: 模型加载失败

**检查**: 确保模型文件存在于正确路径
```bash
ls *.pt  # 检查模型文件
```

### 问题3: 连接被拒绝

**检查**:
1. 服务器是否正在运行
2. 防火墙设置
3. 客户端使用的IP和端口是否正确

---

## 完整工作流程示例

```python
import requests

API_URL = "http://127.0.0.1:5156"

# 1. 识别鸟类
response = requests.post(f"{API_URL}/recognize", json={
    "image_path": "/path/to/bird.jpg",
    "top_k": 3
})

result = response.json()
if result['success']:
    # 2. 获取第一名结果
    top_bird = result['results'][0]
    bird_name = top_bird['cn_name']

    # 3. 获取详细信息
    info_response = requests.get(f"{API_URL}/bird/info",
                                 params={'cn_name': bird_name})
    bird_info = info_response.json()['info']

    # 4. 写入EXIF
    requests.post(f"{API_URL}/exif/write-title", json={
        "image_path": "/path/to/bird.jpg",
        "bird_name": bird_name
    })

    requests.post(f"{API_URL}/exif/write-caption", json={
        "image_path": "/path/to/bird.jpg",
        "caption": bird_info['short_description']
    })

    print(f"✓ 完成！识别为: {bird_name}")
```

---

## 总结

SuperBirdID API提供了强大的RESTful接口，使得：

- ✅ **Lightroom插件**可以调用识别功能
- ✅ **批处理脚本**可以自动化处理大量图片
- ✅ **Web应用**可以集成鸟类识别
- ✅ **移动应用**可以远程调用服务
- ✅ **第三方工具**可以无缝集成

服务器启动后，就像一个专业的鸟类识别引擎，随时待命！🐦🚀
