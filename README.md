<h1 align="center">KIANA</h1>

<p align="center">
  <strong>K</strong>iana <strong>I</strong>s <strong>A</strong> <strong>N</strong>eural <strong>A</strong>ligner.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/jnjnnjzch/kiana_aligner/main/assets/Kiana_logo.png" alt="Kiana Logo" width="200">
</p>

<p align="center">
  <a href="https://github.com/jnjnnjzch/kiana_aligner/blob/main/assets/README_zh.md">🇨🇳 中文说明 (Chinese Version)</a>
</p>



### Project Summary
KIANA is a Python toolkit designed for neural data alignment in neuroscience research. It provides tools for processing and aligning time-series data from various sources—such as electrophysiology (e.g., spike trains) and imaging (e.g., fMRI)—to experimental events or stimuli.

It serves as a comprehensive toolkit for synchronizing, integrating, and analyzing heterogeneous neuroscience data (behavioral, electrophysiological, etc.). `kiana` aims to make the tedious, error-prone workflow of data alignment simple, reliable, and reproducible.

---

## 💡 Why Kiana?

In neuroscience research, we often need to process data streams from different devices with distinct time bases:
* **Behavioral Control Systems** (e.g., MonkeyLogic): Millisecond-level event markers.
* **High-Speed Cameras** or **Motion Capture Systems**: Video frames.
* **Electrophysiology Systems** (e.g., Plexon, Blackrock): Microsecond-level neural spikes and LFP signals.

Precise alignment of these "heterogeneous" timelines is a **prerequisite** for subsequent analysis, but the process is often painful, time-consuming, and prone to errors. `kiana` was created to solve this pain point. It provides a **"Recipe-Driven"** framework that allows you to elegantly define, execute, and verify complex data synchronization tasks.

## ✨ Core Features

* **Recipe-Driven**: Define every step of data processing clearly using `.add_segment()` method chaining, just like writing a "recipe".
* **Multi-Source Loading**: Built-in flexible loaders (`MatLoader`, `DataFrameLoader`) with support for easy extension.
* **Robust Alignment**: The core uses **Dynamic Time Warping (DTW)** algorithms to effectively handle common issues in event sequences such as "clock drift", "missing events", or "extra artifacts".
* **Multi-Context Synchronization**: Easily align a single behavioral timeline with multiple independent electrophysiological recording contexts (e.g., times from different probes or devices).
* **One-Stop Analysis & Visualization**: Includes a powerful `SpikeTrainAnalyzer`, allowing you to go from data alignment to publication-quality PSTH/Raster plots in one step.

## 🚀 Installation

You can install directly from PyPI via `pip`:
```bash
pip install kiana
```

Alternatively, for development:
1.  **Clone the repository to your local machine**:
    ```bash
    git clone git+https://github.com/jnjnnjzch/kiana_aligner.git
    ```

2.  **Enter the project root directory**:
    ```bash
    cd kiana_aligner
    ```

3.  **Install in editable mode**:
    ```bash
    pip install -e .
    ```

## 🚀 Quick Start

Let's experience the core power of `kiana` in 5 minutes with a complete, runnable example: integrating two heterogeneous behavioral datasets (mock experiment logs and mock motion capture events) and aligning them with electrophysiological data.

```python
import numpy as np
import pandas as pd

# Assuming kiana is installed via pip
from kiana import BehavioralProcessor, DataFrameLoader 

# --- 1. Prepare Mock Data (In reality, these come from your files) ---

# a) Mock behavioral logs from a .mat file (Loaded as DataFrame)
#    Contains TrialID and BehavioralCode info
mock_mat_events = pd.DataFrame({
    'EventTime': [10.1, 15.2, 19.8, 30.5, 35.8, 39.9],
    'BehavioralCode': [19, 45, 9, 19, 45, 9],
    'TrialID': [1, 1, 1, 2, 2, 2]
})

# b) Mock motion capture events from a camera (Timestamps only)
mock_motion_events = pd.DataFrame({
    'EventTime': [12.5, 33.1]
})

# c) Mock results from a single Ephys Controller after EphysProcessor
#    Contains timestamps in seconds (times) and indices in sample points (indices)
mock_ephys_data = {
    'times': np.array([10.0, 20.0, 30.0, 40.0]),
    'indices': np.array([300000, 600000, 900000, 1200000])
}


# --- 2. Initialize Processor and Add Segments using "Recipe" Mode ---

# Instantiate the processor
bhv_proc = BehavioralProcessor()

# Add the first segment: From behavioral logs, specifying anchors
bhv_proc.add_segment(
    segment_name='TrialLog',
    loader=DataFrameLoader(trial_id_col='TrialID'), # Tell loader which column is TrialID
    source=mock_mat_events
).with_anchors("BehavioralCode == 19") # Use Behavioral Code 19 as the "Anchor" for ephys alignment

# Add the second segment: From motion capture, all events are anchors
bhv_proc.add_segment(
    segment_name='MotionCapture',
    loader=DataFrameLoader(),
    source=mock_motion_events
) # Without .with_anchors(), all events in this segment are anchors by default

# Execute build to integrate all behavioral segments
bhv_proc.build()


# --- 3. Add Sync Context (Align Behavioral Data to Ephys) ---

bhv_proc.add_sync_context(
    context_name='A1', # Name for this ephys channel/probe
    ephys_times=mock_ephys_data['times'],
    ephys_indices=mock_ephys_data['indices'],
    sampling_rate=30000
)

# --- 4. Get and Display the Final Aligned DataFrame ---

final_df = bhv_proc.get_final_dataframe()

print("🎉 Kiana alignment complete! Final Event Timeline:")
# Selecting key columns for better display
display_cols = ['segment_name', 'EventTime', 'BehavioralCode', 
                'TrialID', 'is_anchor', 'EphysTime_A1', 'EphysIndice_A1']
print(final_df[display_cols])
```

### 📖 Understanding the Output

After running the code above, you will get a `pandas DataFrame` integrating all information. Please pay special attention to the last few columns:

* **`EphysTime_[controller_name]`** (e.g., `EphysTime_A1`):
    This is one of the **most critical** columns. It represents the precise time (in seconds) of each behavioral event on the *aligned* electrophysiology timeline. Any subsequent analysis requiring timing comparison with neural signals should use this column.

* **`EphysIndice_[controller_name]`** (e.g., `EphysIndice_A1`):
    This is the **most precise and reliable** column. It maps the aligned event time to the **sample point index** in the electrophysiology recording file. If you need to extract event-related neural signal fragments (Spike or LFP) from raw waveform data, please use this column as your "Gold Standard".

* **`AbsoluteDateTime`** (Inferred):
    `kiana` also calculates an approximate real-world calendar time for each event. **Note**: Due to clock drift between different device systems, this time is for **reference only** to help you quickly locate the approximate experiment period. **Do not** use it for any precise scientific analysis.

This optimized example truly reflects the toolkit's powerful capabilities in handling **heterogeneous data, anchor alignment, and multi-context synchronization**, allowing any new user to immediately grasp the core value of `kiana`.

## 🤝 Contributing

We welcome contributions in any form! Whether it's submitting bug reports, suggesting new features, or directly contributing code. Please feel free to open an issue or pull request on our GitHub page.