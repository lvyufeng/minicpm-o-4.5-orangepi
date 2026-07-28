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
# 启动 backend_server
./build/backend_server --model-path ./models/MiniCPM-o-4.5 --port 50051

# 运行基础推理测试（单独终端）
python examples/test_single_token.py   # 单 token 生成
python examples/test_generation.py     # 多 token 生成

# 完整流程测试
python examples/test_tokenizer.py
```

## 开发路线

- [x] 项目结构搭建
- [x] C++ 推理引擎核心代码移植
  - [x] ACL 上下文管理
  - [x] Tensor 抽象层
  - [x] Weights 索引和加载（流式加载优化，避免统一内存双份拷贝）
  - [x] 算子封装 (ops.cpp)
  - [x] Vision Encoder (SigLIP)
  - [x] Audio Encoder/Decoder (Whisper + TTS)
  - [x] Language Model (Qwen2)
  - [x] Decoder Layer（支持标准注意力和门控注意力）
  - [x] 文本生成流水线（prefill + decode + lm_head + greedy sampling）
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
- [x] C++ backend_server 协议层
  - [x] TCP 协议处理（JSON over length-prefixed frames）
  - [x] init/chat_prefill/chat_generate 实现
  - [x] Session 管理
  - [x] 参数解析（max_new_tokens 等）
- [x] 端到端推理验证
  - [x] 模型加载测试
  - [x] Prefill 测试
  - [x] 单 token 生成测试
  - [x] 多 token 生成测试（正确性已验证，见下方"关键修复"）
- [ ] 性能优化
  - [ ] 注意力优化（当前使用低效的per-head fallback实现，需探索Ascend 310B支持的加速方案）
  - [ ] 流式生成支持
  - [ ] 权重量化优化 (W4A16/W8A8 策略调优)
  - [ ] 性能 benchmark 和 profiling
- [ ] Worker/Gateway 集成测试
- [ ] 完整多模态功能（图像、音频输入）

## 性能状态

**实测性能**（Orange Pi AIPro 20T / Ascend 310B）:

| 阶段 | 冷启动 | 热启动 | 说明 |
|------|--------|--------|------|
| Model loading | 60-80s | - | Load 17.7GB weights (4 shards) |
| Prefill (4 tokens) | 21s | 3s | First run compiles kernels |
| Decode | 0.4 tok/s | **400+ tok/s** | Huge JIT penalty on first generate |

**优化历程**（基于 per-head attention baseline）:

| 优化阶段 | 耗时 | 提升 | 说明 |
|---------|------|------|------|
| 基础实现 | 207s | - | Per-head attention with sync copies |
| 异步拷贝优化 | 147s | 29% | Removed 128 sync points in attention loop |

**关键发现**:
- CANN kernel JIT 编译导致首次推理慢 1000x，但编译后的 kernel 会缓存到磁盘（`~/atc_data/kernel_cache/`）
- 热启动后真实性能：**400+ tokens/s**（热路径，无 JIT 开销）
- Warmup 可以预热部分 kernel，但无法完全消除冷启动延迟（每个 session/token 可能触发不同 kernel 变体）

## 关键修复：LM Head 权重错误（严重正确性 bug）

**问题**：模型生成会崩溃成单一重复 token（例如连续 20 个 token 全部输出同一个值）。

**根因**：`language_model.cpp` 中构建 LM head 时错误地复用了输入 embedding 表
(`llm.model.embed_tokens.weight`) 作为输出投影矩阵，等同于假设了 tied word
embeddings。但 MiniCPM-O-4.5 的 `config.json` 中 `tie_word_embeddings: false`，
模型实际有独立的 `llm.lm_head.weight`。两个权重矩阵形状相同
（`[vocab_size, hidden_size]`），所以不会触发 shape 检查报错，只会产生语义错误
的 logits——某个特定 token 在错误的权重矩阵下总是取得最高分，导致输出崩溃。

**修复**：加载并使用真正的 `llm.lm_head.weight` 构建 lm_head chunks，不再复用
`embed_tokens`。

**验证**：修复前后对比（相同输入 `[1,2,3,4,5]`，生成 15 token）：
- 修复前：`[11, 11, 11, 11, 11, ...]`（1 个唯一 token）
- 修复后：`[220, 13, 1440, 13775, 402, 8821, 323, 265, 8832, 35780, 3010, 1440, 82736, 562, 3119]`（14 个唯一 token）

不同输入现在会产生不同的输出（验证了 input-sensitivity），且所有生成的 token 都在合法词表范围内。

下一步优化方向:
- [ ] 预量化权重文件 (W4A16/W8A8 GPTQ/AWQ 格式) - 避免运行时量化开销
- [ ] 探索其他 Ascend 310B 支持的 attention 加速方案
- [ ] 批量化 attention 计算（减少 per-head 串行开销）
- [ ] 流式生成支持

> **已探索但不可用的优化**:
> - `aclnnPromptFlashAttention`: 错误码 561103（Ascend 310B 不支持）
> - `aclnnMatmulCubeCustom`: 错误码 161001（自定义 Cube 算子在 M=1 decode 形状上返回参数错误）
> - `batch_matmul`: 张量重组开销过大，性能反而下降
> - W8A8 动态量化: 运行时量化需要大量 device↔host 同步拷贝，导致模型加载阻塞（测试卡在第0层超过80秒）
>
> 当前性能瓶颈是 per-head attention 串行处理（32个头）。Ascend 310B 的标准 Flash Attention 和 Cube matmul 算子均不可用，400+ tok/s 的热路径性能已接近当前架构上限。

## License

Apache License 2.0, 见 [LICENSE](LICENSE)

## 致谢

- [OpenBMB / MiniCPM-O 团队](https://github.com/OpenBMB/MiniCPM-o)
- [lvyufeng / minicpm-v-4.6-orangepi](https://github.com/lvyufeng/minicpm-v-4.6-orangepi)
- [OpenBMB / MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo)
