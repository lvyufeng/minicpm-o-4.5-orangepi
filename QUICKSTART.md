# 快速开始指南

## 当前实现状态

✅ **已完成核心功能** (2026-07-27)
- C++ 推理引擎完整实现 (NPU 上运行)
- Text-only 推理流程 (Prefill + Generate)
- Python Backend 适配层与 tokenizer 集成
- TCP 协议通信 (length-prefixed binary + JSON)

## 编译和运行

### 1. 编译项目

```bash
cd /mnt/data/minicpm-o-4.5-orangepi

# 编译自定义算子
cd src/csrc/custom_ops
./build.sh
cd ../../..

# 编译推理引擎
./scripts/build.sh
```

编译成功后应该看到：
```
build/backend_server        (383KB)
build/libminicpmo_engine.a  (614KB)
```

### 2. 下载模型权重

```bash
# 安装 huggingface-cli
pip install huggingface_hub[cli] transformers

# 下载 MiniCPM-O-4.5 模型 (~10GB)
huggingface-cli download openbmb/MiniCPM-o-2_6 \
    --local-dir models/MiniCPM-o-4.5 \
    --local-dir-use-symlinks False
```

### 3. 启动 Backend Server

```bash
# 设置环境变量
export ASCEND_CUSTOM_OPP_PATH=/mnt/data/minicpm-o-4.5-orangepi/custom_opp
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash
export ASCEND_DEVICE_ID=0

# 启动服务器
./build/backend_server \
    --model_path models/MiniCPM-o-4.5 \
    --port 50051 \
    --host 127.0.0.1
```

成功启动后会看到：
```
MiniCPM-O Backend Server
Device: NPU 0
[Server] Listening on 127.0.0.1:50051
```

### 4. 测试推理

在另一个终端运行：

```bash
# 基础连接测试
python examples/test_backend_connection.py

# 完整推理测试 (tokenizer + prefill + generate)
python examples/test_tokenizer.py
```

## 推理流程说明

```
用户输入 "Hello, who are you?"
    ↓
[Python] tokenizer.encode() → [1234, 5678, 91011]
    ↓
[Python→C++] TCP: {"type":"chat_prefill", "input_ids":[1234,5678,91011]}
    ↓
[C++] embedding_lookup(input_ids) → embeddings [seq_len, 1024]
    ↓
[C++] prefill_from_embeddings() → last_hidden [1, 1024]
    ↓
[C++] lm_head_greedy(last_hidden) → first_token_id = 42
    ↓
[C++] Loop: decode_step_greedy() → token_id (每步生成一个)
    ↓
[C++→Python] TCP: {"chunk":{"token_id":42, "is_eos":false}}
    ↓
[Python] tokenizer.decode([42]) → "I"
    ↓
输出给用户: "I am..."
```

## 性能预期

基于 minicpm-v-4.6 的经验：

| 阶段 | 预期性能 |
|------|---------|
| Prefill | ~100 tokens/s (取决于序列长度) |
| Decode (baseline) | ~3 tokens/s (未优化) |
| Decode (优化后) | ~10+ tokens/s (W8A8 量化) |

## 与 MiniCPM-o-Demo 集成

### 方式 1: 独立测试 (推荐先做)

直接使用 `examples/test_tokenizer.py` 测试 backend_server。

### 方式 2: 集成到 MiniCPM-o-Demo

```bash
# 1. 克隆 MiniCPM-o-Demo
git clone https://github.com/OpenBMB/MiniCPM-o-Demo
cd MiniCPM-o-Demo

# 2. 链接 backend 模块
ln -sf /mnt/data/minicpm-o-4.5-orangepi/src/backend ./backend_orangepi

# 3. 修改 backend_factory.py
# 添加 orangepi backend 检测逻辑 (参考 INTEGRATION.md)

# 4. 启动完整服务栈
# Terminal 1: backend_server
cd /mnt/data/minicpm-o-4.5-orangepi
./build/backend_server --model_path models/MiniCPM-o-4.5 --port 50051

# Terminal 2: worker
cd MiniCPM-o-Demo
python worker.py --backend-type orangepi --backend-server-url http://127.0.0.1:50051

# Terminal 3: gateway
python gateway.py --port 8006
```

## 故障排查

### 问题 1: backend_server 启动失败

```bash
# 检查 NPU 状态
npu-smi info

# 检查环境变量
echo $ASCEND_CUSTOM_OPP_PATH
ls $ASCEND_CUSTOM_OPP_PATH/vendors/customize/op_api/lib/

# 检查自定义算子是否安装
ls custom_opp/vendors/customize/op_api/lib/libaclnn_*.so
```

### 问题 2: 连接失败

```bash
# 检查服务器是否运行
ps aux | grep backend_server
netstat -tlnp | grep 50051

# 测试端口连通性
telnet 127.0.0.1 50051
```

### 问题 3: Tokenizer 加载失败

```bash
# 安装依赖
pip install transformers torch

# 检查 tokenizer 文件
ls models/MiniCPM-o-4.5/tokenizer.json
```

### 问题 4: 生成结果异常

- 检查模型权重是否完整下载
- 检查 config.json 中的模型配置是否匹配代码
- 启用 DEBUG 日志查看详细信息

## 下一步计划

1. **性能优化**
   - 启用 W8A8 量化
   - 优化 KV Cache 管理
   - Benchmark 和性能调优

2. **多模态支持**
   - Vision Pipeline (图像输入)
   - Audio Pipeline (语音输入/输出)

3. **生产部署**
   - 多卡支持
   - 负载均衡
   - 监控和日志

## 参考文档

- [DEVELOPMENT.md](DEVELOPMENT.md) - 架构设计和开发指南
- [DEPLOYMENT.md](DEPLOYMENT.md) - 详细部署指南
- [INTEGRATION.md](INTEGRATION.md) - MiniCPM-o-Demo 集成指南
- [STATUS.md](STATUS.md) - 项目当前状态
