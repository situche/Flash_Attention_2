# Attention 机制对比：标准注意力与闪电注意力

本项目实现了两种自注意力机制的核心计算方式：标准的 `Softmax` 注意力和改进版的 `Flash Attention`。该代码实现了自注意力层的计算，并对比了标准自注意力和 `Flash Attention` 在计算上的差异。具体地，我们通过实现一个标准的 `Softmax` 注意力计算和一个基于分块计算的 `Flash Attention`，并使用一个简单的测试来验证两者结果的相等性。

### 主要内容
- `calculate_local_fx`：计算 Softmax 中的局部函数。
- `decomposed_softmax`：将标准的 Softmax 计算过程分解为多个步骤进行计算。
- `StdAttentionCord`：标准的自注意力机制。
- `FlashAttentionCore`：改进版的 `Flash Attention`，通过分块计算来提升内存效率。

## 文件结构

```
- README.md           # 本文件
- attention.py         # 包含自注意力计算的核心实现
```

## 功能概述

### `calculate_local_fx(x: torch.Tensor)`
此函数计算给定输入张量 `x` 的局部 Softmax 分解。在这里，我们通过对每个输入的最大值进行减法操作来防止溢出，然后进行 Softmax 操作并返回必要的中间变量（`m`, `l`, `fx`）。

### `decomposed_softmax(fx1, l1, m1, fx2, l2, m2)`
此函数实现了对两个张量的 Softmax 计算的分解，并将其结果合并。它通过计算每个输入的局部 Softmax，然后合并它们，最终得到一个类似于标准 Softmax 的结果。

### `StdAttentionCord`
这是一个标准的自注意力模块实现。通过计算查询 `q`、键 `k` 和值 `v` 的点积来计算注意力权重，并最终输出加权后的值。

### `FlashAttentionCore`
这是改进版的自注意力实现。与标准的注意力不同，`Flash Attention` 通过将输入张量分块处理来提高内存使用效率，适用于大规模的序列数据处理。

## 示例

```python
import math
import torch
import torch.nn as nn

# 初始化张量
x1 = torch.randn(16, 1024)
x2 = torch.randn(16, 1024)

# 标准 Softmax 注意力计算
standard_softmax = nn.Softmax(dim=-1)
standard_softmax_result = standard_softmax(torch.cat((x1,x2), dim=-1))

# 分解 Softmax 计算
m1, l1, fx1 = calculate_local_fx(x1)
m2, l2, fx2 = calculate_local_fx(x2)
decomposed_softmax_result = decomposed_softmax(fx1, l1, m1, fx2, l2, m2)

# 比较两者结果是否相等
print(torch.allclose(standard_softmax_result, decomposed_softmax_result, atol=1e-3))

# 标准自注意力
std_attn = StdAttentionCord(x1.shape[-1])
std_out = std_attn(x1, x2, x2)

# Flash Attention
flash_attn = FlashAttentionCore(x1.shape[-1], 8)
flash_out = flash_attn(x1, x2, x2)

# 比较两者输出是否相等
print(torch.allclose(std_out, flash_out, atol=1e-3))
```

## 环境要求

- Python >= 3.6
- PyTorch >= 1.7.0
- NumPy >= 1.19.0

## 依赖安装

可以通过以下命令安装所需依赖：

```bash
pip install torch numpy
```

## 项目目的

这个项目主要是展示如何通过分解 Softmax 计算来优化自注意力机制中的内存使用，尤其是在处理长序列时。`Flash Attention` 作为一种内存优化技巧，能够有效提高计算效率，在需要处理大规模数据时尤为重要。

## 版权与许可证

该代码基于 MIT 许可证发布，您可以自由使用、修改和分发该代码。

---
