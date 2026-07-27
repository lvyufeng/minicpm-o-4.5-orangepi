# 部署指南

本文档提供 MiniCPM-O-4.5 Orange Pi 后端的完整部署流程。

## 环境准备

### 硬件要求

- **设备**: Orange Pi 5 Max / AIPro 20T (搭载 Ascend 310B NPU)
- **内存**: 建议 16GB+ (模型加载需要 ~8GB)
- **存储**: 20GB+ 可用空间 (模型权重 ~10GB)

### 软件依赖

1. **操作系统**: Ubuntu 22.04 aarch64

2. **CANN Toolkit**: Ascend Computing Architecture and Network

```bash
# 检查 CANN 是否已安装
ls /usr/local/Ascend/ascend-toolkit/latest/

# 如果未安装,从华为官网下载对应版本
# https://www.hiascend.com/software/cann/community
```

3. **编译工具链**:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    git-lfs \
    python3-dev \
    python3-pip
```

4. **Python 依赖**:

```bash
pip3 install -r requirements.txt
```

## 编译步骤

### 1. 克隆项目

```bash
git clone https://github.com/your-org/minicpm-o-4.5-orangepi.git
cd minicpm-o-4.5-orangepi
```

### 2. 编译自定义算子

```bash
cd src/csrc/custom_ops

# 设置 CANN 环境
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

# 编译算子包
./build.sh

# 检查编译产物
ls build_out/custom_opp_ubuntu_aarch64.run

# 安装算子到项目目录
cd build_out
./custom_opp_ubuntu_aarch64.run --quiet \
    --install-path=$(pwd)/../../../custom_opp

# 验证安装
ls ../../../custom_opp/vendors/customize/op_api/lib/
```

预期输出应包含:
- `libaclnn_rms_norm1024_custom.so`
- `libaclnn_matmul_cube_custom.so`
- `libaclnn_silu_mul_custom.so`
- 等其他自定义算子库

### 3. 编译推理引擎

```bash
cd /path/to/minicpm-o-4.5-orangepi
mkdir -p build && cd build

# 设置自定义算子路径
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/../custom_opp
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash

# 配置 CMake
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCUSTOM_OPP_VENDOR=$ASCEND_CUSTOM_OPP_PATH/vendors/customize

# 编译
make -j$(nproc)

# 验证编译产物
ls -lh backend_server libminicpmo_engine.a minicpmo_hybrid_decode
```

预期输出:
```
-rwxr-xr-x backend_server          (~14KB)
-rw-r--r-- libminicpmo_engine.a    (~600KB)
-rwxr-xr-x minicpmo_hybrid_decode  (~14KB)
```

## 模型准备

### 下载模型权重

```bash
# 安装 git-lfs
git lfs install

# 下载 MiniCPM-O-4.5 safetensors 权重
cd /path/to/models
git clone https://huggingface.co/openbmb/MiniCPM-o-2_6 MiniCPM-o-4.5

# 或使用 huggingface-cli
pip install huggingface_hub[cli]
huggingface-cli download openbmb/MiniCPM-o-2_6 \
    --local-dir MiniCPM-o-4.5 \
    --local-dir-use-symlinks False
```

### 验证模型文件

```bash
ls MiniCPM-o-4.5/

# 必需文件:
# - model-*.safetensors (多个分片文件)
# - model.safetensors.index.json
# - config.json
# - tokenizer.json
# - preprocessor_config.json
```

## 运行服务

### 方式 1: 使用启动脚本(推荐)

```bash
cd /path/to/minicpm-o-4.5-orangepi

# 设置环境变量
export MODEL_PATH=/path/to/models/MiniCPM-o-4.5
export DEVICE_ID=0
export PORT=50051

# 启动 backend server
./scripts/start_backend.sh
```

### 方式 2: 直接运行

```bash
# 设置环境
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CUSTOM_OPP_PATH=/path/to/minicpm-o-4.5-orangepi/custom_opp
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash
export ASCEND_DEVICE_ID=0

# 启动 backend server
cd /path/to/minicpm-o-4.5-orangepi/build
./backend_server \
    --model_path /path/to/models/MiniCPM-o-4.5 \
    --host 127.0.0.1 \
    --port 50051 \
    --device_id 0 \
    --log_level INFO
```

### 验证服务运行

```bash
# 检查进程
ps aux | grep backend_server

# 检查端口监听
netstat -tlnp | grep 50051

# 测试连接 (需要先实现 backend_server 的协议层)
python examples/test_backend.py --test connection
```

## 集成到 MiniCPM-o-Demo

### 1. 安装 Python Backend 模块

```bash
cd /path/to/MiniCPM-o-Demo

# 方式 A: 符号链接(开发推荐)
ln -sf /path/to/minicpm-o-4.5-orangepi/src/backend ./backend_orangepi

# 方式 B: 直接复制
cp -r /path/to/minicpm-o-4.5-orangepi/src/backend ./backend_orangepi
```

### 2. 修改 backend_factory.py

编辑 `core/processors/backend_factory.py`,添加:

```python
def create_backend(backend_type: str, **kwargs):
    if backend_type == "orangepi":
        import sys
        sys.path.insert(0, "./backend_orangepi")
        from orangepi_backend import OrangePiBackend
        return OrangePiBackend(**kwargs)
    # ... 原有的 pytorch backend
```

### 3. 配置文件

创建 `config.orangepi.json`:

```json
{
  "backend": "orangepi",
  "model_path": "/path/to/models/MiniCPM-o-4.5",
  "workers": [
    {
      "gpu_id": 0,
      "backend_server_host": "127.0.0.1",
      "backend_server_port": 50051
    }
  ]
}
```

### 4. 启动完整服务栈

```bash
# Terminal 1: 启动 backend_server
cd /path/to/minicpm-o-4.5-orangepi
./scripts/start_backend.sh

# Terminal 2: 启动 MiniCPM-o-Demo
cd /path/to/MiniCPM-o-Demo
python app.py --config config.orangepi.json
```

## 故障排查

### 问题 1: backend_server 无法启动

**症状**: 进程启动后立即退出

**排查**:

```bash
# 检查环境变量
echo $ASCEND_HOME_PATH
echo $ASCEND_CUSTOM_OPP_PATH
echo $ASCEND_DEVICE_ID

# 检查 NPU 状态
npu-smi info

# 查看详细日志
export LOG_LEVEL=DEBUG
./scripts/start_backend.sh
```

**常见原因**:
- CANN 环境未正确初始化
- 自定义算子未安装或路径不对
- NPU 被其他进程占用
- 模型路径不存在或权重文件不完整

### 问题 2: 编译时找不到自定义算子头文件

**症状**: `fatal error: aclnn_rms_norm1024_custom.h: No such file or directory`

**解决**:

```bash
# 确认自定义算子已安装
ls $ASCEND_CUSTOM_OPP_PATH/vendors/customize/op_api/include/

# 重新配置 CMake,指定正确路径
cd build
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DCUSTOM_OPP_VENDOR=$ASCEND_CUSTOM_OPP_PATH/vendors/customize
```

### 问题 3: 运行时找不到自定义算子库

**症状**: `Error: cannot open shared object file: libaclnn_xxx_custom.so`

**解决**:

```bash
# 设置库搜索路径
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/vendors/customize/op_api/lib:$LD_LIBRARY_PATH

# 或者在启动脚本中 source set_env.bash
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash
```

### 问题 4: Python 无法连接到 backend_server

**症状**: `ConnectionRefusedError: [Errno 111] Connection refused`

**排查**:

```bash
# 检查 backend_server 是否运行
ps aux | grep backend_server

# 检查端口监听
netstat -tlnp | grep 50051

# 检查防火墙
sudo ufw status
sudo ufw allow 50051/tcp
```

### 问题 5: 推理结果异常

**排查步骤**:

1. 验证模型权重完整性:
```bash
# 检查所有 safetensors 文件大小是否正常
ls -lh MiniCPM-o-4.5/*.safetensors
```

2. 对比 PyTorch backend 输出(如果可用):
```bash
# 使用相同输入在两个 backend 上运行,对比输出
```

3. 检查量化配置:
```bash
# 临时禁用量化测试
unset MINICPM_W8A8_DECODE
./scripts/start_backend.sh
```

## 性能调优

### 1. 量化策略

```bash
# 默认: 不使用 W8A8
./scripts/start_backend.sh

# 启用 W8A8 (仅 lm_head)
export MINICPM_W8A8_DECODE=1
./scripts/start_backend.sh

# 启用所有 W8A8 算子
export MINICPM_W8A8_DECODE=all
./scripts/start_backend.sh
```

### 2. KV Cache 大小

编辑 `src/csrc/include/minicpmo/language_model.h`:

```cpp
struct LanguageModelConfig {
    int64_t max_seq_len = 4096;  // 降低可减少显存,但限制上下文长度
    // ...
};
```

重新编译后生效。

### 3. 批处理大小

当前版本 batch_size=1。如需支持批处理,需要修改:
- `backend_server.cpp` 的请求队列
- 各模块的 `forward` 接口增加 batch 维度

## 监控与日志

### 查看 NPU 使用率

```bash
# 实时监控
watch -n 1 npu-smi info

# 查看设备利用率
npu-smi info -t usages -i 0
```

### Backend Server 日志

日志级别设置:

```bash
export LOG_LEVEL=DEBUG  # DEBUG | INFO | WARNING | ERROR
./scripts/start_backend.sh
```

### 性能分析

```bash
# 使用 CANN profiler
export PROFILING_MODE=true
export PROFILING_DIR=./profiling_data
./build/backend_server ...

# 分析结果
msprof --output=profiling_data --application=backend_server
```

## 生产部署建议

1. **进程管理**: 使用 systemd 或 supervisor 管理 backend_server 进程
2. **负载均衡**: 多实例部署,使用 nginx 或 haproxy 做负载均衡
3. **监控告警**: 接入 Prometheus + Grafana 监控 NPU 状态
4. **日志收集**: 使用 ELK 或 Loki 集中收集日志
5. **容错机制**: 实现请求重试和 graceful degradation

## 参考资料

- [CANN 开发者文档](https://www.hiascend.com/document)
- [AscendC 算子开发指南](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/operatordev/ascendc-dev/ascendc-dev_00001.html)
- [MiniCPM-o-Demo 集成指南](docs/INTEGRATION.md)
