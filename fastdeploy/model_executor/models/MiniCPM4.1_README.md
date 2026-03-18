# MiniCPM4.1-8B 模型 FastDeploy 部署指南

本文档介绍如何在 FastDeploy 框架中部署和运行 `openbmb/MiniCPM4.1-8B` 模型。我们支持了标准的生成式推理，并无缝对接了 FastDeploy 的高性能算子（如 Flash Attention / Paged Attention 和 SwiGLU）、底层图优化以及现有的各种低 bit 量化推理能力（如 W8A8、AWQ、GPTQ 等）。

## 1. 简介

MiniCPM4.1-8B 是一款高性能混合推理模型。在 FastDeploy 中，该模型采用了标准的自回归 Decoder-only 架构实现，包含：
- Grouped-Query Attention (GQA)
- SwiGLU 激活函数
- Rotary Position Embedding (RoPE)
- RMSNorm 归一化

由于其结构与 LLaMA/Qwen 高度相似，FastDeploy 能够以最优的显存和计算效率来执行推理。如果您需要原生的 128k 超长上下文支持（依赖于 InfLLM v2 sparse attention），可以自行通过注册 Custom Op 进行加载扩展，FastDeploy 的默认实现兼容了标准的长上下文 dense attention 推理。

## 2. 准备工作

确保您的运行环境符合 FastDeploy 要求，安装好 GPU 版本的 FastDeploy 及其依赖：
```bash
pip install -r requirements.txt
```

获取模型权重：
从 HuggingFace 模型库下载 `openbmb/MiniCPM4.1-8B` 权重并保存至本地，如 `/path/to/MiniCPM4.1-8B/`。

## 3. 推理使用

在 FastDeploy 中运行 MiniCPM4.1-8B：

### 3.1 基础单卡推理 (FP16/BF16)
您可以直接使用 FastDeploy 的内置加载器加载模型：

```python
from fastdeploy import LLM, SamplingParams

model_path = "/path/to/MiniCPM4.1-8B"
llm = LLM(model=model_path, tensor_parallel_size=1, max_model_len=8192)

prompts = ["介绍一下量子力学？", "如何使用Python编写爬虫？"]
sampling_params = SamplingParams(max_tokens=512)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}\nResponse: {output.outputs.text}\n")
```

### 3.2 低 bit 量化推理 (W8A8 / W4A16 / GPTQ / AWQ)

我们为 MiniCPM4.1-8B 适配了 FastDeploy 现有的低 bit 量化能力。得益于 `RowParallelLinear` 等抽象算子的实现，量化过程会自动在加载时读取配置文件并使用 `process_weights_after_loading` 触发。

假设您已有 MiniCPM4.1-8B 的 GPTQ/AWQ 权重，或者使用 W8A8 PTQ (Post-Training Quantization) 推理，只需在配置中或启动参数中指定即可：

**W8A8 PTQ 启动示例:**
```python
from fastdeploy import LLM, SamplingParams

# 假设您加载的是经过 W8A8 量化的权重目录，或者是需要在加载时配置量化参数
# FastDeploy 的 LLM 接口会自动读取模型目录中的 config.json（如 quant_method="w8a8"）
llm = LLM(
    model="/path/to/MiniCPM4.1-8B",
    tensor_parallel_size=1,
    max_model_len=8192,
    quantization="w8a8" # 或者 "gptq", "awq"
)

sampling_params = SamplingParams(max_tokens=256)
outputs = llm.generate(["What is FastDeploy?"], sampling_params)

for output in outputs:
    print(output.outputs.text)
```

对于 AWQ 和 GPTQ，同样直接将 `model_dir` 指向您经过 AWQ/GPTQ 转换的本地目录，FastDeploy 的 `minicpm.py` 实现会自动识别并解析量化层。

## 4. 组网代码说明

- 组网代码位于 `FastDeploy/fastdeploy/model_executor/models/minicpm.py`。
- 我们实现了 `MiniCPMAttention` 和 `MiniCPMMLP` 模块。
- 自动适配权重映射：通过内部 `stacked_params_mapping`，将 HuggingFace 上的 `q_proj`, `k_proj`, `v_proj` 自动融合并转换为 FastDeploy 中的 `qkv_proj` 并行线性层；`gate_proj` 和 `up_proj` 融合为 `up_gate_proj`。
- 支持张量并行 (Tensor Parallelism): 您可以在多卡环境下运行此模型，模型内的并行算子均已进行通信配置切分。

## 5. 自定义算子支持 (可选)

如需使用 InfLLM v2 稀疏注意力 (Sparse Attention) 针对极端长文本进行推理加速，您可在 `FastDeploy/custom_ops/gpu_ops/` 目录下开发/接入 `infllm_v2.cu` 算子并编译链接。但在不影响默认运行逻辑的前提下，内置的高性能 Dense Attention 已完全支持标准的 MiniCPM4.1-8B 推理。