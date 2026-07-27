# 开发指南

本文档面向希望在此项目基础上进行二次开发或贡献代码的开发者。

## 项目架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (Python)                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  MiniCPM-o-Demo / 自定义应用                       │    │
│  │  - FastAPI Web Server                              │    │
│  │  - WebSocket /v1/realtime                          │    │
│  │  - Gradio UI                                       │    │
│  └──────────────────┬──────────────────────────────────┘    │
└────────────────────┼─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              Backend 适配层 (Python)                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │  backend/orangepi_backend.py                       │    │
│  │  - 统一接口封装                                     │    │
│  │  - TCP Socket 通信                                 │    │
│  │  - JSON 协议序列化                                 │    │
│  └──────────────────┬──────────────────────────────────┘    │
└────────────────────┼─────────────────────────────────────────┘
                     │ TCP Socket (JSON protocol)
┌────────────────────▼─────────────────────────────────────────┐
│            C++ 推理引擎 (NPU)                                │
│  ┌────────────────────────────────────────────────────┐    │
│  │  backend/backend_server.cpp (待完善)               │    │
│  │  - TCP 服务器                                      │    │
│  │  - 请求分发                                        │    │
│  │  - Session 管理                                    │    │
│  └──────────────────┬──────────────────────────────────┘    │
│  ┌──────────────────▼──────────────────────────────────┐    │
│  │  推理引擎核心 (lib/*.cpp)                          │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ language_model.cpp                           │ │    │
│  │  │ - Qwen2 Language Model                       │ │    │
│  │  │ - KV Cache 管理                              │ │    │
│  │  │ - Prefill + Decode                           │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ vision.cpp                                   │ │    │
│  │  │ - SigLIP Vision Encoder                      │ │    │
│  │  │ - Patch Embedding                            │ │    │
│  │  │ - VitMerger                                  │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ audio_encoder.cpp / audio_decoder.cpp        │ │    │
│  │  │ - Whisper Audio Encoder                      │ │    │
│  │  │ - SSML TTS Decoder                           │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ decoder_layer.cpp                            │ │    │
│  │  │ - Full Attention (Softmax)                   │ │    │
│  │  │ - Linear Attention (Gated Delta Rule)       │ │    │
│  │  │ - SwiGLU MLP                                 │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ ops.cpp                                      │ │    │
│  │  │ - ACL 算子调用封装                           │ │    │
│  │  │ - 自动 fallback 到内置算子                   │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ weights.cpp / tensor.cpp                     │ │    │
│  │  │ - Safetensors 加载                           │ │    │
│  │  │ - 设备内存管理                               │ │    │
│  │  │ - 量化权重支持                               │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  自定义 AscendC 算子 (custom_ops/)                  │    │
│  │  - RmsNorm1024Custom                               │    │
│  │  - MatmulCubeCustom                                │    │
│  │  - MatmulW4a16Custom / MatmulW8a8I32Custom         │    │
│  │  - SiluMulCustom                                   │    │
│  │  - LinearGatedDeltaRuleCustom                      │    │
│  │  - AttentionStepCustom                             │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 核心模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| **Backend 适配** | `src/backend/orangepi_backend.py` | 提供统一的 Python 接口,与 C++ backend server 通信 |
| **Backend Server** | `src/csrc/backend/backend_server.cpp` | TCP 服务器,协议处理,请求分发 (待完善) |
| **Language Model** | `src/csrc/lib/language_model.cpp` | Qwen2 LM 推理,KV cache,Prefill/Decode |
| **Vision Encoder** | `src/csrc/lib/vision.cpp` | SigLIP 视觉编码,Patch embedding,VitMerger |
| **Audio Encoder** | `src/csrc/lib/audio_encoder.cpp` | Whisper 音频编码 (基础实现) |
| **Audio Decoder** | `src/csrc/lib/audio_decoder.cpp` | TTS 音频解码 (基础实现) |
| **Decoder Layer** | `src/csrc/lib/decoder_layer.cpp` | Full/Linear Attention,MLP,LayerNorm |
| **算子封装** | `src/csrc/lib/ops.cpp` | ACL 算子调用,自定义算子集成 |
| **权重管理** | `src/csrc/lib/weights.cpp` | Safetensors 加载,内存管理,量化 |
| **Tensor 抽象** | `src/csrc/lib/tensor.cpp` | 设备内存封装,shape 管理,数据传输 |
| **ACL 上下文** | `src/csrc/lib/acl_context.cpp` | NPU 设备管理,Stream 管理 |
| **自定义算子** | `src/csrc/custom_ops/op_kernel/` | AscendC 高性能算子实现 |

## 开发环境设置

### 1. 安装依赖

```bash
# 系统依赖
sudo apt install -y build-essential cmake git python3-dev

# Python 依赖
pip install -r requirements.txt

# 开发工具(可选)
pip install black flake8 mypy pytest
```

### 2. 配置 IDE

**VS Code 推荐设置** (`.vscode/settings.json`):

```json
{
  "C_Cpp.default.includePath": [
    "${workspaceFolder}/src/csrc/include",
    "/usr/local/Ascend/ascend-toolkit/latest/include"
  ],
  "C_Cpp.default.defines": [
    "ASCEND_NPU"
  ],
  "python.analysis.extraPaths": [
    "${workspaceFolder}/src"
  ]
}
```

### 3. 编译 Debug 版本

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug \
    -DCUSTOM_OPP_VENDOR=$ASCEND_CUSTOM_OPP_PATH/vendors/customize
make -j$(nproc)
```

## 添加新算子

### 场景: 需要添加一个新的融合算子

#### 步骤 1: 编写 AscendC 核函数

创建 `src/csrc/custom_ops/op_kernel/my_new_op.cpp`:

```cpp
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename T>
class MyNewOpKernel {
public:
    __aicore__ inline MyNewOpKernel() {}
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR out,
                                 uint32_t totalLength) {
        // 初始化: 定义输入输出队列
        xGm.SetGlobalBuffer((__gm__ T*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ T*)y, totalLength);
        outGm.SetGlobalBuffer((__gm__ T*)out, totalLength);
        
        pipe.InitBuffer(xLocal, BUFFER_NUM, totalLength * sizeof(T));
        pipe.InitBuffer(yLocal, BUFFER_NUM, totalLength * sizeof(T));
        pipe.InitBuffer(outLocal, BUFFER_NUM, totalLength * sizeof(T));
        
        this->totalLength = totalLength;
    }
    
    __aicore__ inline void Process() {
        // 主处理循环
        uint32_t loopCount = totalLength / BLOCK_SIZE;
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t idx) {
        LocalTensor<T> xLocalTensor = xLocal.AllocTensor<T>();
        LocalTensor<T> yLocalTensor = yLocal.AllocTensor<T>();
        DataCopy(xLocalTensor, xGm[idx * BLOCK_SIZE], BLOCK_SIZE);
        DataCopy(yLocalTensor, yGm[idx * BLOCK_SIZE], BLOCK_SIZE);
        xLocal.EnQue(xLocalTensor);
        yLocal.EnQue(yLocalTensor);
    }
    
    __aicore__ inline void Compute(uint32_t idx) {
        LocalTensor<T> xLocalTensor = xLocal.DeQue<T>();
        LocalTensor<T> yLocalTensor = yLocal.DeQue<T>();
        LocalTensor<T> outLocalTensor = outLocal.AllocTensor<T>();
        
        // 实现计算逻辑,例如: out = x + y * 2
        Add(outLocalTensor, xLocalTensor, yLocalTensor, BLOCK_SIZE);
        Muls(outLocalTensor, outLocalTensor, (T)2, BLOCK_SIZE);
        
        outLocal.EnQue<T>(outLocalTensor);
        xLocal.FreeTensor(xLocalTensor);
        yLocal.FreeTensor(yLocalTensor);
    }
    
    __aicore__ inline void CopyOut(uint32_t idx) {
        LocalTensor<T> outLocalTensor = outLocal.DeQue<T>();
        DataCopy(outGm[idx * BLOCK_SIZE], outLocalTensor, BLOCK_SIZE);
        outLocal.FreeTensor(outLocalTensor);
    }

    TPipe pipe;
    GlobalTensor<T> xGm, yGm, outGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> xLocal, yLocal;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outLocal;
    uint32_t totalLength;
    static constexpr uint32_t BLOCK_SIZE = 256;
};

// 核函数入口
extern "C" __global__ __aicore__ void my_new_op_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR out, uint32_t totalLength) {
    
    MyNewOpKernel<half> op;
    op.Init(x, y, out, totalLength);
    op.Process();
}
```

#### 步骤 2: 编写主机侧接口

创建 `src/csrc/custom_ops/op_host/my_new_op_tiling.h`:

```cpp
#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
struct MyNewOpTilingData : public TilingData {
    uint32_t totalLength;
};
}  // namespace optiling
```

创建 `src/csrc/custom_ops/op_host/my_new_op.cpp`:

```cpp
#include "my_new_op_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    MyNewOpTilingData tiling;
    // 从context获取shape信息并填充tiling
    // ...
    context->SetBlockDim(8);  // 设置block数
    context->GetAttrs()->SetAttr("tiling_key", tiling);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ops {
class MyNewOpCustom : public OpDef {
public:
    explicit MyNewOpCustom(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16});
        
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
    }
};

OP_ADD(MyNewOpCustom);
}  // namespace ops
```

#### 步骤 3: 注册算子

创建 `my_new_op_custom.json`:

```json
{
  "op_interface": {
    "op_name": "MyNewOpCustom",
    "inputs": [
      {"name": "x", "dtype": "float16"},
      {"name": "y", "dtype": "float16"}
    ],
    "outputs": [
      {"name": "out", "dtype": "float16"}
    ]
  }
}
```

#### 步骤 4: 集成到引擎

编辑 `src/csrc/lib/ops.cpp`,添加调用接口:

```cpp
#include "aclnn_my_new_op_custom.h"

void my_new_op(const Tensor& x, const Tensor& y, Tensor& out, aclrtStream stream) {
    uint64_t ws_size = 0;
    void* ws = nullptr;
    aclOpExecutor* executor = nullptr;
    
    // 尝试调用自定义算子
    int ret = aclnnMyNewOpCustomGetWorkspaceSize(
        x.acl_tensor(), y.acl_tensor(), out.acl_tensor(), &ws_size, &executor);
    
    if (ret == ACL_SUCCESS && ws_size > 0) {
        check_acl(aclrtMalloc(&ws, ws_size, ACL_MEM_MALLOC_HUGE_FIRST), "alloc ws");
    }
    
    if (ret == ACL_SUCCESS) {
        check_acl(aclnnMyNewOpCustom(ws, ws_size, executor, stream), "my_new_op_custom");
    } else {
        // Fallback: 使用内置算子组合实现
        Tensor temp({/* shape */}, DType::Float16); temp.allocate();
        add(x, y, temp, stream);
        muls(temp, 2.0f, out, stream);
    }
    
    if (ws) aclrtFree(ws);
}
```

#### 步骤 5: 重新编译

```bash
cd src/csrc/custom_ops
./build.sh

cd ../../../build
export ASCEND_CUSTOM_OPP_PATH=$(pwd)/../custom_opp
source $ASCEND_CUSTOM_OPP_PATH/vendors/customize/bin/set_env.bash
make -j$(nproc)
```

## 调试技巧

### 1. NPU 内核调试

启用 ACL 日志:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1  # 0=debug, 1=info, 2=warning, 3=error
export ASCEND_SLOG_PRINT_TO_STDOUT=1
./build/backend_server ...
```

### 2. 算子性能分析

```bash
# 启用 profiling
export PROFILING_MODE=true
export PROFILING_OPTIONS="task_trace,op_trace"
./build/backend_server ...

# 分析结果
msprof --output=./profiling_data
```

### 3. 内存泄漏检测

```bash
# 使用 valgrind (仅能检测主机侧)
valgrind --leak-check=full ./build/backend_server

# NPU 内存泄漏: 在退出前打印剩余分配
aclrtGetMemInfo(ACL_HBM_MEM, &free, &total);
```

### 4. GDB 调试

```bash
gdb ./build/backend_server
(gdb) set environment ASCEND_DEVICE_ID 0
(gdb) break language_model.cpp:123
(gdb) run --model_path /path/to/model
```

## 测试

### 单元测试

```bash
# 编译测试
cd build
cmake .. -DBUILD_TESTING=ON
make -j$(nproc)

# 运行测试
ctest --output-on-failure
```

### 集成测试

```python
# tests/test_integration.py
import pytest
from backend import create_backend

def test_backend_lifecycle():
    backend = create_backend("orangepi", model_path="/path/to/model", gpu_id=0)
    backend.load_model()
    metrics = backend.metrics()
    assert metrics["backend"] == "orangepi"
    backend.shutdown()
```

## 性能优化清单

- [ ] 使用 Cube 单元加速矩阵乘法
- [ ] 算子融合 (LayerNorm+Linear, SiLU+Mul)
- [ ] 权重量化 (W4A16, W8A8)
- [ ] KV Cache 优化 (PagedAttention)
- [ ] 多 Stream 并行
- [ ] 权重预加载到 HBM
- [ ] 批处理推理

## 代码风格

- **C++**: Google C++ Style Guide,使用 `clang-format`
- **Python**: PEP 8,使用 `black` 格式化
- **命名**: 驼峰 (类名),下划线 (函数/变量)
- **注释**: 公开接口必须有文档注释

## 贡献流程

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/my-feature`)
3. 提交修改 (`git commit -am 'Add my feature'`)
4. 推送分支 (`git push origin feature/my-feature`)
5. 创建 Pull Request

## 常见问题

**Q: 如何添加对新模型的支持?**

A: 参考 `language_model.cpp`,实现新的 `XXXConfig` 和 `XXXWeights`,在 `load_language_model_weights` 中添加加载逻辑。

**Q: 如何优化特定 shape 的算子?**

A: 在自定义算子中添加 shape 特化路径,或在 `ops.cpp` 中根据 shape 选择不同的算子实现。

**Q: 如何支持新的量化格式?**

A: 在 `quantized_weight.h` 中定义新的量化结构体,在 `quantized_weight.cpp` 中实现加载和反量化逻辑。

## 参考资源

- [AscendC 编程指南](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/operatordev/ascendc-dev/ascendc-dev_00001.html)
- [ACL API 参考](https://www.hiascend.com/document/detail/zh/canncommercial/63RC2/infacldevg/aclcppdevg/aclcppdevg_000001.html)
- [MiniCPM-O 模型文档](https://github.com/OpenBMB/MiniCPM-o)
