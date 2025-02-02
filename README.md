# Attention 机制对比：标准注意力与闪电注意力

本项目展示了两种注意力机制：标准注意力和闪电注意力。代码中还对比了分解 Softmax 与标准 Softmax 的效果，主要用于多头注意力计算。

## 环境要求

确保安装了以下 Python 包：

```bash
pip install torch
```

## 代码介绍

### 1. **分解 Softmax**

此部分实现了 Softmax 的分解方式，通过分解的方式减少内存使用：

```python
def calculate_local_fx(x: torch.Tensor):
    m = torch.max(x, dim=-1, keepdim=True)[0]
    fx = torch.exp(x - m)
    l = torch.sum(fx, dim=-1, keepdim=True)
    return m, l, fx
```

### 2. **标准注意力（StdAttentionCord）**

标准注意力通过计算查询（Q）、键（K）和值（V）的点积相似度，使用 Softmax 计算加权平均值：

```python
class StdAttentionCord(nn.Module):
    def __init__(self, attention_head_size):
        super().__init__()
        self.softmax_scaling_constant = 1 / math.sqrt(attention_head_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        x = torch.matmul(q, k.transpose(-1, -2))
        x = x * self.softmax_scaling_constant
        x = self.softmax(x)
        x = torch.matmul(x, v)
        return x
```

### 3. **闪电注意力（FlashAttentionCore）**

闪电注意力通过将 Q、K、V 划分成小块来减少内存占用并提高计算效率：

```python
class FlashAttentionCore(nn.Module):
    def __init__(self, attention_head_size, block_size):
        super().__init__()
        self.softmax_scaling_constant = 1 / math.sqrt(attention_head_size)
        self.block_size = block_size

    def forward(self, q, k, v):
        num_blocks = q.shape[1] // self.block_size
        q_list, k_list, v_list = [torch.chunk(tensor, num_blocks, dim=1) for tensor in [q, k, v]]
        # 执行块计算并返回结果
        return O
```

### 4. **验证与测试**

我们验证了标准注意力和闪电注意力的输出是否相同：

```python
torch.manual_seed(42)
q, k, v = [torch.rand(8, 512, 768) for _ in range(3)]
std_attn = StdAttentionCord(q.shape[-1])
flash_attn = FlashAttentionCore(q.shape[-1], 8)

std_out = std_attn(q, k, v)
flash_out = flash_attn(q, k, v)
print(torch.allclose(std_out, flash_out, atol=1e-3))
```

## 总结

- **标准注意力**：实现简单，但在计算大规模数据时可能会占用大量内存。
- **闪电注意力**：通过分块计算，提高了内存效率和计算速度，适合处理大规模数据。
