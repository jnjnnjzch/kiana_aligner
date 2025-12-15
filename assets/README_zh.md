<h1 align="center">KIANA</h1>

<p align="center">
  <strong>K</strong>iana <strong>I</strong>s <strong>A</strong> <strong>N</strong>eural <strong>A</strong>ligner.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/jnjnnjzch/kiana_aligner/main/assets/Kiana_logo.png" alt="Kiana Logo" width="200">
</p>

<p align="center">
  <a href="https://github.com/jnjnnjzch/kiana_aligner/blob/main/README.md">⬅️ 返回英文版 (Back to English README)</a>
</p>


### 项目简介 (Project Summary)
**KIANA** 是一个专为神经科学研究设计的 Python 数据对齐工具包。它提供了一套强大的工具，用于同步、整合与分析来自不同来源的异质性时间序列数据——例如电生理信号（Spike trains, LFP）、行为学数据及影像数据（如 fMRI）——并将它们统一到一个可靠的时间轴上。

`kiana` 旨在将神经科学实验中繁琐、易错的数据对齐工作流程变得简单、可靠和可复现。

---

## 💡 解决了什么问题？ (Why Kiana?)

在神经科学研究中，我们常常需要处理来自不同设备、具有不同时间基准的数据流：
* **行为控制系统**（如 MonkeyLogic）：毫秒级事件标记。
* **高速摄像机**或**动捕系统**：基于视频帧的时间戳。
* **电生理记录系统**（如 Plexon, Blackrock）：微秒级神经脉冲和 LFP 信号。

将这些“异质”的时间轴精确地对齐，是进行后续分析的**前提**，但这个过程通常非常痛苦、耗时且容易出错（例如时钟漂移）。`kiana` 就是为了解决这个痛点而生，它提供了一个**“配方驱动”**的框架，让您能优雅地定义、执行和验证复杂的数据同步任务。

## ✨ 核心功能

* **配方驱动 (Recipe-Driven)**: 通过 `.add_segment()` 链式调用，像写“配方”一样清晰地定义数据处理的每一个步骤。
* **多源加载**: 内置灵活的加载器 (`MatLoader`, `DataFrameLoader`)，并支持轻松扩展。
* **鲁棒对齐**: 核心采用 **动态时间规整 (DTW)** 算法，能有效处理事件序列中常见的“时间漂移”、“事件丢失”或“额外干扰事件”等问题。
* **多上下文同步**: 轻松将一个行为时间轴与多个独立的电生理记录上下文（例如，来自不同探针或不同设备的时间系统）对齐。
* **一站式分析与可视化**: 内置强大的 `SpikeTrainAnalyzer`，从数据对齐到最终发表级的 PSTH/Raster 图表绘制，一步到位。

## 🚀 安装

您可以通过 `pip` 直接从 PyPI 安装：
```bash
pip install kiana
```

另一种方式是进行开发者安装：
1.  **首先，克隆 (Clone) 代码仓库到您的本地电脑**:
    ```bash
    git clone git+https://github.com/jnjnnjzch/kiana_aligner.git
    ```

2.  **进入项目根目录**:
    ```bash
    cd kiana_aligner
    ```

3.  **执行可编辑模式安装**:
    ```bash
    pip install -e .
    ```

## 🚀 快速上手 (Quick Start)

让我们通过一个完整的、可运行的示例，在 5 分钟内体验 `kiana` 的核心魅力：整合两份异质行为数据（一份模拟的实验日志和一份模拟的动捕事件），并与一份电生理数据对齐。

```python
import numpy as np
import pandas as pd

# 假设 kiana 已经通过 pip install 安装好了
from kiana import BehavioralProcessor, DataFrameLoader 

# --- 1. 准备模拟数据 (在真实场景中，这些数据来自您的文件) ---

# a) 模拟来自 .mat 文件的行为日志 (已加载为 DataFrame)
#    包含试次 (TrialID) 和行为码 (BehavioralCode) 信息
mock_mat_events = pd.DataFrame({
    'EventTime': [10.1, 15.2, 19.8, 30.5, 35.8, 39.9],
    'BehavioralCode': [19, 45, 9, 19, 45, 9],
    'TrialID': [1, 1, 1, 2, 2, 2]
})

# b) 模拟来自摄像头动捕的事件 (只有时间戳)
mock_motion_events = pd.DataFrame({
    'EventTime': [12.5, 33.1]
})

# c) 模拟来自 EphysProcessor 处理后的单个电生理控制器的结果
#    包含以秒为单位的时间戳 (times) 和以采样点为单位的索引 (indices)
mock_ephys_data = {
    'times': np.array([10.0, 20.0, 30.0, 40.0]),
    'indices': np.array([300000, 600000, 900000, 1200000])
}


# --- 2. 初始化处理器，并用“配方”模式添加数据段 ---

# 实例化处理器
bhv_proc = BehavioralProcessor()

# 添加第一个数据段：来自行为日志，并指定锚点
bhv_proc.add_segment(
    segment_name='TrialLog',
    loader=DataFrameLoader(trial_id_col='TrialID'), # 告知加载器哪一列是试次ID
    source=mock_mat_events
).with_anchors("BehavioralCode == 19") # 使用行为码19作为与电生理对齐的“锚点”

# 添加第二个数据段：来自运动捕捉，所有事件都作为锚点
bhv_proc.add_segment(
    segment_name='MotionCapture',
    loader=DataFrameLoader(),
    source=mock_motion_events
) # 不使用 with_anchors()，默认此段内所有事件都为锚点

# 执行构建，整合所有行为数据段
bhv_proc.build()


# --- 3. 添加同步上下文，将行为数据与电生理数据对齐 ---

bhv_proc.add_sync_context(
    context_name='A1', # 为这个电生理通道命名
    ephys_times=mock_ephys_data['times'],
    ephys_indices=mock_ephys_data['indices'],
    sampling_rate=30000
)

# --- 4. 获取并展示最终对齐的事件数据框 ---

final_df = bhv_proc.get_final_dataframe()

print("🎉 kiana 对齐完成！最终事件时间轴:")
# 为了显示更美观，我们只选择部分关键列
display_cols = ['segment_name', 'EventTime', 'BehavioralCode', 
                'TrialID', 'is_anchor', 'EphysTime_A1', 'EphysIndice_A1']
print(final_df[display_cols])
```

### 📖 理解输出结果

当您运行上面的代码后，您会得到一个整合了所有信息的 `pandas DataFrame`。请特别关注最后几列：

* **`EphysTime_[controller_name]`** (例如 `EphysTime_A1`):
    这是**最重要**的列之一。它表示每个行为事件，在**对齐之后**，对应到该电生理控制器时间轴上的精确时间（单位：秒）。所有后续需要与神经信号进行时间比较的分析，都应使用这一列。

* **`EphysIndice_[controller_name]`** (例如 `EphysIndice_A1`):
    这是**最精确、最可靠**的列。它表示对齐后的事件时间，对应到电生理记录文件中的**采样点索引**。如果您需要从原始波形数据中提取事件相关的神经信号片段（Spike 或 LFP），请务必使用这一列作为您的“金标准”。

* **`AbsoluteDateTime`** (自动推算):
    `kiana` 还会为您推算出每个事件在现实世界中的大概日历时间。**请注意**: 由于不同设备系统时钟存在误差，这个时间**仅供参考**，帮助您快速定位实验的大概时间段。**切勿**将它用于任何精确的科学分析。

## 🤝 参与贡献 (Contributing)

我们非常欢迎任何形式的贡献！无论是提交 Bug 报告、提出新功能建议，还是直接贡献代码。请随时在 GitHub Issues 页面提出您的想法或提交 Pull Request。