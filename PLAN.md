# MiniCPM-O-4.5 项目开发计划

## 项目目标
将 MiniCPM-O-4.5 全模态模型移植到 Orange Pi AIPro 20T (Ascend 310B NPU) 上运行。

## 参考项目

### 1. minicpm-v-4.6-orangepi
**用途**: C++ 推理引擎架构和 NPU 优化

- 核心组件:
  - ACL 上下文管理 (`acl_context.cpp/h`)
  - Tensor 抽象层 (`tensor.cpp/h`)
  - 权重加载 (`weights.cpp/h`)
  - Ops 封装 (`ops.cpp/h`)
  - Decoder 层 (`decoder_layer.cpp/h`)
  - 语言模型 (`language_model.cpp/h`)
  - 视觉塔 (`vision.cpp/h`)
  - 量化权重支持 (`quantized_weight.cpp/h`)

- 自定义算子 (src/csrc/custom_ops/):
  - `matmul_cube_custom` - M=1 优化的 cube matmul
  - `attention_step_custom` - 单 token attention
  - `rms_norm1024_custom` - 融合 RMSNorm
  - `silu_mul_custom` - 融合 SwiGLU
  - `linear_causal_conv_step_custom` - 向量化 conv1d
  - `linear_gated_delta_rule_*` - gated-delta-rule 递推
  - 量化相关: `matmul_w4a16`, `matmul_w8a8_i32`

### 2. MiniCPM-o-Demo
**用途**: Backend 协议和实时交互框架

- 协议文档 (docs/backend-protocol/):
  - `network.md` - 四原语 (init/push/pull/unary)
  - `schema.md` - 消息格式和编码
  - `sequences.md` - 时序图示例

- 参考实现:
  - `py_backend/server.py` - WebSocket backend 服务器
  - `py_backend/chat_util.py` - 请求解析
  - `py_backend/media.py` - 音频/图像编解码
  - `runtime/backend_client.py` - 客户端实现
  - `worker.py` - 会话管理和转发
  - `gateway.py` - 负载均衡和入口

- Docker 参考:
  - `docker/Dockerfile.cpp-worker-backend` - C++ backend 容器
  - `docker/entrypoint-cpp-worker-backend.sh` - 启动脚本

## 开发阶段

### 阶段 1: 基础结构 ✓
- [x] 项目目录结构
- [x] CMakeLists.txt
- [x] 构建脚本 (build.sh, install_custom_ops.sh, set_env.sh)
- [x] 基础文档 (README.md, LICENSE)
- [x] Stub 文件占位

### 阶段 2: 自定义算子移植
从 minicpm-v-4.6-orangepi 复制并适配:

```bash
src/csrc/custom_ops/
├── CMakeLists.txt
├── cmake/                    # AscendC 构建系统
├── framework/                # 算子框架
├── op_host/                  # Host 端 tiling
├── op_kernel/                # AscendC kernel 实现
├── *.json                    # 算子定义
└── build.sh
```

关键算子:
1. `matmul_cube_custom` - 核心 matmul 优化
2. `rms_norm1024_custom` - normalization
3. `silu_mul_custom` - activation
4. `attention_step_custom` - decode attention

### 阶段 3: 推理引擎核心
从 minicpm-v-4.6-orangepi 移植并适配 MiniCPM-O 架构:

#### 3.1 基础设施
- `acl_context.cpp/h` - ACL 初始化和设备管理
- `tensor.cpp/h` - Tensor 抽象(shape, stride, buffer)
- `ops.cpp/h` - ACL ops 封装 (matmul, add, norm, etc.)

#### 3.2 权重系统
- `weights.cpp/h` - 从 safetensors 加载权重
- `quantized_weight.cpp/h` - 量化权重支持

#### 3.3 模型组件
- `decoder_layer.cpp/h` - Transformer decoder 层
- `language_model.cpp/h` - Qwen2-based 语言模型
- `vision.cpp/h` - SigLIP 视觉塔
- `audio_encoder.cpp` - 音频编码器 (新增)
- `audio_decoder.cpp` - 音频解码器/TTS (新增)

### 阶段 4: Backend 协议层
实现 C++ WebSocket backend server:

#### 4.1 协议实现
参考 `py_backend/server.py` 实现:
- WebSocket `/backend` endpoint
- init 原语 - 初始化会话
- push 原语 - 接收输入事件
- pull 原语 - 下发输出事件
- unary 原语 - 单次请求-响应

#### 4.2 消息处理
参考 `py_backend/chat_util.py`:
- 解析 messages/content 字段
- 处理 generation config
- TTS 参数解析

#### 4.3 媒体编解码
参考 `py_backend/media.py`:
- 音频: float PCM 编解码
- 图像: JPEG 帧解码
- 视频: 帧序列处理

### 阶段 5: 集成和测试

#### 5.1 Python 绑定
- `src/python/minicpmo/session.py` - Python 接口
- ctypes/pybind11 绑定

#### 5.2 Worker/Gateway
复用 MiniCPM-o-Demo 的 Python 实现:
- `worker.py` - 会话转发
- `gateway.py` - 负载均衡

#### 5.3 端到端测试
参考 `tests/e2e_realtime.py`:
- 文本对话测试
- 多模态输入测试
- 流式输出测试

### 阶段 6: 优化

#### 6.1 性能优化
- lm_head 分块 (参考 minicpm-v)
- conv1d step 向量化
- 多切片视觉处理

#### 6.2 权重量化
- INT8 量化 (`matmul_w8a8_i32`)
- INT4 量化 (`matmul_w4a16`)
- 目标: 3 tps → 10+ tps

## 模型架构差异

### MiniCPM-V-4.6 vs MiniCPM-O-4.5

| 组件 | MiniCPM-V-4.6 | MiniCPM-O-4.5 |
|------|---------------|---------------|
| 语言模型 | Qwen2 24层 | Qwen2 24层 (相同) |
| 视觉 | SigLIP | SigLIP (相同) |
| 音频输入 | ❌ | ✅ Audio Encoder |
| 音频输出 | ❌ | ✅ TTS Decoder |
| Vocab | 248094 | 需确认 |
| Hidden | 1024 | 需确认 |

需要新增:
1. 音频编码器 (Whisper-like?)
2. 音频解码器 + Token2Wav TTS
3. 多模态融合逻辑

## 文件映射

### 从 minicpm-v 复制 (直接复用或轻度修改)
```
minicpm-v-4.6-orangepi/              → minicpm-o-4.5-orangepi/
├── CMakeLists.txt                    → CMakeLists.txt (已改)
├── scripts/                          → scripts/ (已复制)
├── src/csrc/custom_ops/              → src/csrc/custom_ops/ (待复制)
├── src/csrc/include/minicpmv/        → src/csrc/include/minicpmo/
│   ├── acl_context.h                 → acl_context.h ✓
│   ├── tensor.h                      → tensor.h ✓
│   ├── ops.h                         → ops.h ✓
│   ├── weights.h                     → weights.h ✓
│   ├── quantized_weight.h            → quantized_weight.h ✓
│   ├── decoder_layer.h               → decoder_layer.h ✓
│   ├── language_model.h              → language_model.h ✓
│   └── vision.h                      → vision.h ✓
└── src/csrc/lib/                     → src/csrc/lib/ (对应文件 ✓)
```

### 新增 (参考 MiniCPM-o-Demo)
```
├── src/csrc/backend/
│   └── backend_server.cpp            → WebSocket 协议服务器 ✓
├── src/csrc/lib/
│   ├── audio_encoder.cpp             → 音频编码 ✓
│   └── audio_decoder.cpp             → TTS 解码 ✓
├── worker.py                         → 会话转发 ✓
└── gateway.py                        → 负载均衡 ✓
```

## 下一步行动

1. **复制自定义算子**: `cp -r /tmp/minicpm-v-ref/src/csrc/custom_ops/* src/csrc/custom_ops/`
2. **测试算子编译**: `./scripts/install_custom_ops.sh`
3. **移植基础组件**: acl_context, tensor, ops
4. **移植语言模型**: decoder_layer, language_model
5. **移植视觉塔**: vision
6. **实现 backend 协议**: backend_server.cpp
7. **集成测试**: 端到端跑通
8. **性能优化**: 量化、kernel 优化
