# MiniCPM-O-4.5 on Ascend 310B (Orange Pi AIPro 20T)

[English](README.md) · [中文](README_ZH.md)

一个 C++/AscendC 推理引擎,将 [MiniCPM-O-4.5](https://huggingface.co/openbmb/MiniCPM-o-2_6) 全模态模型运行在 Orange Pi AIPro 20T 板载的 Ascend 310B NPU 上。

**支持文本、图像、音频多模态输入输出,推理热路径完全在 NPU 上运行,Python 端仅用于预处理。**

> ⚠️ **项目状态**: 初始开发中
> - 参考 [minicpm-v-4.6-orangepi](https://github.com/lvyufeng/minicpm-v-4.6-orangepi) 的推理引擎架构
> - 集成 [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 的实时交互协议

## 架构设计

### 三层架构

```
┌─────────────────────────────────────────────────┐
│  Gateway (gateway.py)                           │
│  - WebSocket /v1/realtime endpoint             │
│  - Worker pool management                       │
│  - Load balancing                              │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  Worker (worker.py)                             │
│  - Session management                           │
│  - Protocol forwarding                          │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│  C++ Backend (backend_server)                   │
│  ┌───────────────────────────────────────────┐ │
│  │ Protocol Layer                            │ │
│  │ - WebSocket /backend endpoint            │ │
│  │ - init/push/pull/unary primitives       │ │
│  └───────────────┬───────────────────────────┘ │
│  ┌───────────────▼───────────────────────────┐ │
│  │ Inference Engine (NPU)                    │ │
│  │ - Language Model (Qwen2 24-layer)        │ │
│  │ - Vision Tower (SigLIP)                  │ │
│  │ - Audio Encoder/Decoder                  │ │
│  │ - Custom AscendC kernels                 │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### 与参考项目的关系

| 组件 | 来源 | 说明 |
|------|------|------|
| C++ 推理引擎核心 | minicpm-v-4.6-orangepi | Ascend NPU 优化、自定义算子、内存管理 |
| Backend 协议 | MiniCPM-o-Demo | WebSocket 协议、init/push/pull 原语 |
| Worker/Gateway | MiniCPM-o-Demo | 会话管理、负载均衡 |
| 模型结构 | MiniCPM-O-4.5 | 全模态架构(text+image+audio) |

## 快速开始

### 硬件 & 软件要求

- **板子**: Orange Pi AIPro 20T (Ascend 310B)
- **系统**: Ubuntu 22.04 aarch64
- **CANN toolkit**: 8.3.RC2, 安装在 `/usr/local/Ascend/ascend-toolkit/latest/`
- **CMake**: ≥ 3.20
- **Python**: 3.8+

### 安装步骤

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 编译自定义 AscendC 算子
./scripts/install_custom_ops.sh

# 3. 编译 C++ backend
./scripts/build.sh

# 4. 下载模型权重
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-o-2_6 models/MiniCPM-o-4.5
```

### 启动服务

```bash
# 设置环境变量
source scripts/set_env.sh

# 启动 C++ backend (端口 22500)
./build/backend_server --model-path ./models/MiniCPM-o-4.5 --port 22500

# 启动 worker (端口 22400)
python worker.py --backend-server-url http://127.0.0.1:22500 --port 22400

# 启动 gateway (端口 8006)
python gateway.py --host 0.0.0.0 --port 8006

# 注册 worker 到 gateway
curl -X PUT http://127.0.0.1:8007/internal/workers/local-worker \
    -H 'content-type: application/json' \
    --data '{"endpoint":"127.0.0.1:22400","gpu_group":"gpu-0"}'
```

### 测试

```bash
# 健康检查
curl http://127.0.0.1:22500/health  # backend
curl http://127.0.0.1:22400/health  # worker
curl http://127.0.0.1:8006/health   # gateway

# 端到端测试
python tests/e2e_realtime.py
```

## 开发路线

- [x] 项目结构搭建
- [x] C++ 推理引擎核心代码移植
  - [x] ACL 上下文管理
  - [x] Tensor 抽象层
  - [x] Weights 索引和加载
  - [x] 算子封装 (ops.cpp)
  - [x] Vision Encoder (SigLIP)
  - [x] Audio Encoder/Decoder (Whisper + TTS)
  - [x] Language Model (Qwen2 + Linear Attention)
  - [x] Decoder Layer (Full/Linear Attention)
- [x] 自定义 AscendC 算子实现
  - [x] RmsNorm1024Custom
  - [x] MatmulCubeCustom (M=1 快速路径)
  - [x] MatmulW4a16Custom (4-bit 量化)
  - [x] MatmulW8a8I32Custom (8-bit 量化)
  - [x] SiluMulCustom (算子融合)
  - [x] GatedRmsNormZCustom
  - [x] LinearGatedDeltaRuleCustom (线性注意力)
  - [x] AttentionStepCustom (单步解码注意力)
- [x] Backend 协议适配层 (Python)
  - [x] OrangePiBackend 实现
  - [x] Backend Factory
  - [x] 兼容 MiniCPM-o-Demo 接口
- [ ] C++ backend_server 协议层完善
  - [ ] TCP/WebSocket 协议处理
  - [ ] init/push/pull 原语实现
  - [ ] Session 管理
- [ ] Worker/Gateway 集成测试
- [ ] 权重量化优化 (W4A16/W8A8 策略调优)
- [ ] 性能 benchmark 和优化

## 性能目标

基于 minicpm-v-4.6 的经验:

| 阶段 | 目标 |
|------|------|
| 基础实现 | ~3 tokens/s (aclnnMm baseline) |
| Cube 优化 | ~5 tokens/s (自定义 matmul) |
| 权重量化 | ~10+ tokens/s (INT8/INT4) |

## License

Apache License 2.0, 见 [LICENSE](LICENSE)

## 致谢

- [OpenBMB / MiniCPM-O 团队](https://github.com/OpenBMB/MiniCPM-o)
- [lvyufeng / minicpm-v-4.6-orangepi](https://github.com/lvyufeng/minicpm-v-4.6-orangepi)
- [OpenBMB / MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)
