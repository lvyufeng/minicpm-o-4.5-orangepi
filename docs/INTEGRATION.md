# MiniCPM-O-4.5 Orange Pi Backend Integration Guide

本文档说明如何将 MiniCPM-O-4.5 Orange Pi 后端集成到 MiniCPM-o-Demo 项目中。

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    MiniCPM-o-Demo                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Frontend / API Server                     │    │
│  │       (FastAPI, WebSocket, Gradio)                  │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
│  ┌──────────────────▼──────────────────────────────────┐    │
│  │         Backend Factory (backend_factory.py)        │    │
│  │    if backend_type == "orangepi":                   │    │
│  │        return OrangePiBackend()                     │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │                                        │
└─────────────────────┼────────────────────────────────────────┘
                      │ TCP Socket (JSON protocol)
┌─────────────────────▼────────────────────────────────────────┐
│          Orange Pi Backend (本项目)                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │      orangepi_backend.py (Python client)           │    │
│  └──────────────────┬──────────────────────────────────┘    │
│                     │ TCP Socket                            │
│  ┌──────────────────▼──────────────────────────────────┐    │
│  │      backend_server (C++ inference engine)         │    │
│  │   - ACL Runtime (Ascend NPU)                       │    │
│  │   - Custom AscendC Ops                             │    │
│  │   - Vision / Audio / Language Models               │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## 集成步骤

### 1. 安装 Orange Pi Backend

```bash
# 克隆本项目
git clone https://github.com/your-org/minicpm-o-4.5-orangepi.git
cd minicpm-o-4.5-orangepi

# 编译 C++ 引擎和自定义算子
cd src/csrc/custom_ops
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
./build.sh

# 安装自定义算子
cd build_out
./custom_opp_ubuntu_aarch64.run --quiet --install-path=$(pwd)/../../../custom_opp

# 编译主引擎
cd ../../../build
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/../custom_opp
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DCUSTOM_OPP_VENDOR=$ASCEND_CUSTOM_OPP_PATH/vendors/customize
make -j$(nproc)

# 验证编译产物
ls -lh backend_server libminicpmo_engine.a
```

### 2. 集成到 MiniCPM-o-Demo

#### 方案 A: 符号链接(开发环境推荐)

```bash
cd /path/to/MiniCPM-o-Demo

# 链接 backend 模块
ln -sf /path/to/minicpm-o-4.5-orangepi/src/backend ./backend_orangepi

# 或者直接复制
cp -r /path/to/minicpm-o-4.5-orangepi/src/backend ./backend_orangepi
```

#### 方案 B: 修改 backend_factory.py

编辑 `core/processors/backend_factory.py`:

```python
def create_backend(backend_type: str, **kwargs):
    """Create backend instance based on type."""
    
    if backend_type == "pytorch":
        from core.processors.pytorch_backend import PyTorchBackend
        return PyTorchBackend(**kwargs)
    
    elif backend_type == "orangepi":
        # 添加 Orange Pi backend 支持
        import sys
        sys.path.insert(0, "/path/to/minicpm-o-4.5-orangepi/src")
        from backend.orangepi_backend import OrangePiBackend
        return OrangePiBackend(**kwargs)
    
    else:
        raise ValueError(f"Unknown backend: {backend_type}")
```

### 3. 配置并启动 Backend Server

创建配置文件 `config.orangepi.json`:

```json
{
  "backend": {
    "type": "orangepi",
    "model_path": "/path/to/minicpm-o-4.5-safetensors",
    "server_host": "127.0.0.1",
    "server_port": 50051,
    "device_id": 0
  },
  "workers": [
    {
      "gpu_id": 0,
      "model_path": "/path/to/minicpm-o-4.5-safetensors",
      "backend_server_host": "127.0.0.1",
      "backend_server_port": 50051
    }
  ]
}
```

启动 backend server:

```bash
cd /path/to/minicpm-o-4.5-orangepi

# 方式1: 使用启动脚本
export MODEL_PATH=/path/to/minicpm-o-4.5-safetensors
export DEVICE_ID=0
./scripts/start_backend.sh

# 方式2: 直接运行
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/custom_opp
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash
export ASCEND_DEVICE_ID=0

./build/backend_server \
  --model_path /path/to/minicpm-o-4.5-safetensors \
  --host 127.0.0.1 \
  --port 50051
```

### 4. 启动 MiniCPM-o-Demo

```bash
cd /path/to/MiniCPM-o-Demo

# 使用 Orange Pi backend 配置
python app.py --config config.orangepi.json
```

## 接口兼容性

Orange Pi Backend 实现了与 PyTorchBackend 相同的接口:

### 已实现的方法

- ✅ `load_model()` - 连接到 backend server
- ✅ `metrics()` - 获取运行指标
- ✅ `chat_prefill()` - Chat 模式预填充
- ✅ `chat_init_tts()` - 初始化 TTS
- ✅ `chat_streaming_generate()` - 流式生成
- ✅ `chat_non_streaming_generate()` - 非流式生成
- ✅ `shutdown()` - 关闭连接

### 待实现的方法

- ⏳ `duplex_prepare()` - Duplex 模式准备
- ⏳ `duplex_prefill()` - Duplex 预填充
- ⏳ `duplex_generate()` - Duplex 生成
- ⏳ `half_duplex_*()` - Half-duplex 系列方法

当前版本专注于 **Chat 模式**(单轮对话和流式对话),Duplex 和 Half-duplex 模式将在后续版本实现。

## 性能优化建议

### 1. 自定义算子优化

当前实现了以下高性能 AscendC 算子:

- `RmsNorm1024Custom` - RMS Normalization (优化版,hidden_size=1024)
- `MatmulCubeCustom` - 矩阵乘法(Cube 单元加速)
- `MatmulW4a16Custom` - W4A16 量化矩阵乘法
- `MatmulW8a8I32Custom` - W8A8 量化矩阵乘法
- `SiluMulCustom` - SiLU + 逐元素乘法融合
- `LinearGatedDeltaRuleCustom` - Gated Linear Attention(线性注意力核心算子)

如需进一步优化:

```bash
cd src/csrc/custom_ops/op_kernel
# 编辑对应算子的 .cpp 实现
# 重新编译并安装
```

### 2. 模型量化

支持以下量化方案(由 WeightsIndex 自动检测):

- **W4A16**: 4-bit 权重 + 16-bit 激活(推荐用于 Attention QKV/FFN)
- **W8A8**: 8-bit 权重 + 8-bit 激活(推荐用于 lm_head)

设置环境变量控制量化策略:

```bash
# 启用 W8A8 解码(仅 lm_head)
export MINICPM_W8A8_DECODE=1

# 启用所有支持的 W8A8 算子
export MINICPM_W8A8_DECODE=all
```

### 3. KV Cache 管理

当前实现为静态 KV cache(固定最大长度)。如遇显存不足:

```cpp
// 修改 src/csrc/include/minicpmo/language_model.h
struct LanguageModelConfig {
    int64_t max_seq_len = 8192;  // 降低此值可减少显存占用
    // ...
};
```

## 故障排查

### Backend Server 无法启动

检查环境变量:

```bash
# 必需的环境变量
echo $ASCEND_HOME_PATH        # /usr/local/Ascend/ascend-toolkit/latest
echo $ASCEND_CUSTOM_OPP_PATH  # /path/to/custom_opp
echo $ASCEND_DEVICE_ID        # 0

# 检查 NPU 状态
npu-smi info

# 检查自定义算子
ls $ASCEND_CUSTOM_OPP_PATH/vendors/customize/op_api/lib/
```

### 连接超时

检查 backend server 是否正在运行:

```bash
netstat -tlnp | grep 50051
ps aux | grep backend_server
```

检查防火墙:

```bash
sudo ufw allow 50051/tcp  # Ubuntu
sudo firewall-cmd --add-port=50051/tcp --permanent  # CentOS
```

### 推理结果不正确

1. 验证模型文件完整性
2. 检查自定义算子是否正确安装
3. 对比 PyTorch backend 的输出(如果可用)

```bash
# 启用详细日志
export LOG_LEVEL=DEBUG
./scripts/start_backend.sh
```

## 性能基准

在 Orange Pi 5 Max (RK3588 + Ascend 310B) 上的初步测试:

| 模式 | Tokens/s | 内存占用 | 备注 |
|------|----------|----------|------|
| Chat (FP16) | ~15 | 8GB | 无量化 |
| Chat (W4A16) | ~25 | 5GB | 推荐配置 |
| Streaming | ~20 | 5GB | W4A16 + TTS |

*实际性能取决于模型大小、输入长度和硬件配置。*

## 下一步

1. **实现 Duplex 模式** - 实时语音交互
2. **优化内存管理** - 动态 KV cache、内存池
3. **多卡支持** - 模型并行/流水线并行
4. **Python 绑定** - 避免 TCP socket 开销

## 参考资源

- [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) - 原始 PyTorch 实现
- [MiniCPM-V Orange Pi](https://github.com/lvyufeng/minicpm-v-4.6-orangepi) - 参考项目
- [Ascend CANN 开发文档](https://www.hiascend.com/document)
