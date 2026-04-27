# Minimal Pair DTW Method - Forward Sliding Window

## 概述 (Overview)

`get_pair_via_dtw_minimal` 是 KIANA 工具包中新增的高效 DTW 配对方法。它使用前向滑动窗口方法，避免了分块处理的重叠问题。

`get_pair_via_dtw_minimal` is a new efficient DTW pairing method in the KIANA toolkit. It uses a forward sliding window approach, avoiding the overlap problems of block processing.

## 核心特性 (Key Features)

1. **前向滑动窗口 (Forward Sliding Window)**: 处理第一个块后验证所有后续数据
2. **自动验证 (Automatic Validation)**: 检查配对关系在剩余数据上是否成立
3. **失败点重启 (Restart on Failure)**: 当验证失败时，从失败点重新开始
4. **无重叠问题 (No Overlap Issues)**: 纯前向过程，避免块之间的重叠

## 算法原理 (Algorithm)

```
1. 处理第一个块 (chunk_size 个元素)
   ↓
2. 建立配对关系 (时间映射关系)
   ↓
3. 验证这个关系在剩余所有数据上是否成立
   ↓
4. 如果全部验证通过 → 完成
   ↓
5. 如果在位置 K 验证失败 → 从位置 K 重新开始处理
   ↓
6. 重复直到处理完所有数据
```

这是一个前向的类似滑动窗口的过程，而不是分块处理。

## 使用示例 (Usage Examples)

### 基础用法 (Basic Usage)

```python
from kiana import get_pair_via_dtw_minimal
import numpy as np

# 准备两个长时间序列
template = np.cumsum(np.random.uniform(0.9, 1.1, 1000))
query = template + np.random.normal(0, 0.1, 1000)

# 使用前向滑动窗口 DTW 配对
pairs = get_pair_via_dtw_minimal(
    template, 
    query, 
    chunk_size=50,      # 初始块的大小
    tolerance=0.01,     # 验证容差
    verbose=True        # 显示详细信息
)

print(f"配对数量: {len(pairs)}")
print(f"示例配对: {pairs[:5]}")
```

### 处理时钟漂移 (Handling Clock Drift)

```python
# 模拟时钟漂移：前半部分匹配良好，后半部分有漂移
template = np.cumsum(np.random.uniform(0.9, 1.1, 200))
query = np.concatenate([
    template[:100] + np.random.normal(0, 0.02, 100),  # 前半部分
    template[100:] * 1.05 + np.random.normal(0, 0.05, 100)  # 后半部分有5%漂移
])

# 算法会自动检测漂移点并重新配对
pairs = get_pair_via_dtw_minimal(
    template, 
    query, 
    chunk_size=40,
    tolerance=0.05,  # 较大的容差适应噪声
    verbose=True
)

# 输出会显示在哪里检测到失败并重新开始
```

## 参数说明 (Parameters)

- **template** (array-like): 参考序列（模板）
- **query** (array-like): 待对齐序列
- **chunk_size** (int, 默认=50): 初始块的大小
- **tolerance** (float, 默认=0.01): 验证的容差
- **step_pattern** (str, 默认="symmetric2"): DTW 步进模式
- **verbose** (bool, 默认=False): 是否显示详细处理信息
- **max_attempts** (int, 默认=3): 每个块的最大尝试次数

## 返回值 (Return Value)

返回配对列表，格式为 `[(template_idx, query_idx), ...]`，表示模板序列和查询序列中对应点的索引。

## 与分块处理的区别 (Difference from Block Processing)

| 特性 | 前向滑动窗口 | 分块处理 |
|------|------------|---------|
| 处理方式 | 处理块 → 验证全部剩余 | 处理块1 → 处理块2 → ... |
| 重叠问题 | 无重叠 | 需要处理块之间的重叠 |
| 效率 | 验证快，失败才重新处理 | 每个块都要完整处理 |
| 适用场景 | 数据质量较好，失败点少 | 数据噪声大，需要全程处理 |

## 工作流程示例 (Workflow Example)

```
序列长度: 150
chunk_size: 50

步骤 1: 处理 [0:50]
       建立配对关系
       验证 [50:150]
       → 在位置 75 验证失败

步骤 2: 处理 [75:125]  
       建立新的配对关系
       验证 [125:150]
       → 全部验证通过，完成！

最终结果: 位置 0-74 和 75-150 的配对
```

## 性能特点 (Performance Characteristics)

1. **最佳情况**: 数据质量好，验证全部通过 → 只需处理一个块
2. **一般情况**: 少量失败点，需要重新处理 → 处理 2-3 个块
3. **最坏情况**: 频繁失败 → 接近传统分块方法

## 适用场景 (Use Cases)

✅ **适合使用**:
- 长时间序列 (> 100 个点)
- 数据质量较好，噪声小
- 时钟漂移不频繁
- 需要避免重叠问题

❌ **不适合使用**:
- 短序列 (< 50 个点) - 自动回退到常规 DTW
- 数据质量差，频繁出现异常
- 需要精确对齐每个点

## 注意事项 (Notes)

1. 该方法假设配对关系在一定范围内是稳定的
2. 当检测到关系失效时会自动重新建立
3. `tolerance` 参数需要根据数据质量调整
4. `chunk_size` 应该足够大以建立可靠的关系

## 相关函数 (Related Functions)

- `get_pair_via_dtw()`: 常规 DTW 配对方法
- `get_paired_ephys_event_index()`: 将配对转换为索引映射
- `_purify_pairs()`: 清理配对结果

## 贡献 (Contributing)

欢迎提交问题报告和改进建议！如果您发现任何 bug 或有新功能建议，请在 GitHub 上开启 issue。
