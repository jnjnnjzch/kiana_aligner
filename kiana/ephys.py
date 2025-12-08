import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Union, List

class EphysProcessor:
    """
    一个用于处理神经电生理会话数据的类。
    它负责加载、合并、过滤和计算来自多个控制器的时间和索引数据。
    """
    
    def __init__(self, session_id: str, root_path: Union[str, Path], f_s: int = 30000):
        """
        初始化 SessionProcessor。

        Args:
            session_id (str): 会话的唯一标识符 (例如 '20250321')。
            root_path (Union[str, Path]): 包含数据文件的根目录路径。
            f_s (int, optional): 采样率。默认为 30000。
        """
        self.session_id = session_id
        self.root_path = Path(root_path).expanduser()
        self.f_s = f_s
        
        # 公开属性，方便随时查看和调试
        self.data: Optional[pd.DataFrame] = None
        self.filtered_data: Optional[pd.DataFrame] = None
        self.cumulative_results: Dict[str, Dict] = {}

        print(f"✅ Processor initialized for session: {self.session_id}")

    def _find_data_files(self):
        """在路径中查找对应的json数据文件（内部方法）。"""
        time_dicts_path = self.root_path
        try:
            time_dict_file = next(time_dicts_path.glob(f"**/{self.session_id}*_time_dict.json"))
            indice_dict_file = next(time_dicts_path.glob(f"**/{self.session_id}*_indice_dict.json"))
            print(f"Found time file: {time_dict_file}")
            print(f"Found indice file: {indice_dict_file}")
            return time_dict_file, indice_dict_file
        except StopIteration:
            raise FileNotFoundError(f"❌ Error: Could not find data files for session '{self.session_id}' in {time_dicts_path}")

    def _parse_raw_data(self, raw_dict: dict, value_col_name: str) -> pd.DataFrame:
        """解析原始的嵌套字典数据为DataFrame（内部方法）。"""
        parsed_list = []
        for key, sub_dict in raw_dict.items():
            controller = str(key).split("/")[-2]
            for key2, value in sub_dict.items():
                parsed_list.append({
                    "filepath": key,
                    "filename": key2,
                    value_col_name: value,
                    "abs_start_time": datetime.strptime(key2, "Temp_%y%m%d_%H%M%S"),
                    "controller": controller,
                })
        return pd.DataFrame(parsed_list)

    def load_and_merge_data(self):
        """加载、解析并合并时间和索引数据。"""
        time_dict_file, indice_dict_file = self._find_data_files()

        with time_dict_file.open("r", encoding="utf-8") as f:
            time_dict = json.load(f)
        with indice_dict_file.open("r", encoding="utf-8") as f:
            indice_dict = json.load(f)

        time_data = self._parse_raw_data(time_dict, "time")
        indice_data = self._parse_raw_data(indice_dict, "indices")
        
        merged_data = pd.merge(time_data, indice_data, on=["filepath", "filename", "abs_start_time", "controller"])

        merged_data["time"] = merged_data["time"].apply(lambda x: x[-1])
        merged_data["indices"] = merged_data["indices"].apply(lambda x: x[-1])
        merged_data["diff_indice"] = merged_data["indices"].apply(lambda x: [x[i] - x[i-1] for i in range(1, len(x))])
        merged_data["num_timestamp"] = merged_data["time"].apply(len)
        merged_data["num_indice"] = merged_data["indices"].apply(len)
        
        merged_data.sort_values(by=["abs_start_time"], inplace=True)
        merged_data.reset_index(drop=True, inplace=True)
        
        self.data = merged_data
        self.filtered_data = self.data.copy()
        print(f"✅ Data loaded. Total {len(self.data)} rows.")
        return self

    def filter_by_time(self, start_time_str: str, end_time_str: str):
        """根据给定的开始和结束时间过滤数据。此操作会更新 self.filtered_data。"""
        if self.filtered_data is None:
            print("⚠️ Warning: Data not loaded yet.")
            return self
            
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

        mask = (self.filtered_data['abs_start_time'] >= start_time) & (self.filtered_data['abs_start_time'] <= end_time)
        self.filtered_data = self.filtered_data[mask].reset_index(drop=True)
        
        print(f"✅ Data filtered by time. {len(self.filtered_data)} rows remaining.")
        return self
    
    def filter_by_controller(self, keep: Optional[List[str]] = None, drop: Optional[List[str]] = None):
        """根据控制器名称过滤数据。此操作会更新 self.filtered_data。"""
        if self.filtered_data is None:
            print("⚠️ Warning: Data not loaded yet.")
            return self
        
        if keep:
            self.filtered_data = self.filtered_data[self.filtered_data['controller'].isin(keep)].reset_index(drop=True)
            print(f"✅ Kept controllers {keep}. {len(self.filtered_data)} rows remaining.")
        elif drop:
            self.filtered_data = self.filtered_data[~self.filtered_data['controller'].isin(drop)].reset_index(drop=True)
            print(f"✅ Dropped controllers {drop}. {len(self.filtered_data)} rows remaining.")
        return self

    def _calculate_cumulative_values(self, controller_rows: pd.DataFrame):
        """为单个控制器计算累积时间和索引（内部方法）。"""
        cum_indices, cum_times = [], []
        if controller_rows.empty:
            return cum_indices, cum_times
        
        controller_rows_sorted = controller_rows.sort_values(by="abs_start_time").reset_index(drop=True)
        
        # 将第一个记录段的开始时间作为基准时间 0
        base_time = controller_rows_sorted.iloc[0]["abs_start_time"]

        for _, row in controller_rows_sorted.iterrows():
            # 计算当前行相对于基准时间的偏移（秒）
            time_offset_seconds = (row["abs_start_time"] - base_time).total_seconds()
            # 这里有一个先验，采集系统的时间偏移是以分钟为单位的，因此应该四舍五入到60s的整数倍
            time_offset_seconds = round(time_offset_seconds / 60) * 60
            indice_offset = time_offset_seconds * self.f_s
            
            cum_indices.extend([indice + indice_offset for indice in row["indices"]])
            cum_times.extend([t + time_offset_seconds for t in row["time"]])

        return cum_indices, cum_times

    def process_controllers(self) -> Dict[str, Dict]:
        """处理过滤后的数据，计算所有控制器的累积时间和索引。"""
        if self.filtered_data is None or self.filtered_data.empty:
            print("⚠️ Warning: No data to process.")
            return {}

        self.cumulative_results = {}
        controllers = sorted(self.filtered_data["controller"].unique())

        for controller in controllers:
            print(f"Processing controller: {controller}...")
            controller_rows = self.filtered_data[self.filtered_data["controller"] == controller]
            cum_indices, cum_times = self._calculate_cumulative_values(controller_rows)
            
            self.cumulative_results[controller] = {'indices': cum_indices, 'times': cum_times}
            print(f"  -> Generated {len(cum_indices)} cumulative indices.")
        
        print("✅ All controllers processed.")
        return None # use get_result() to access results

    def get_result(self, controller: str) -> Optional[Dict[str, List]]:
        """获取指定控制器的处理结果。"""
        result = self.cumulative_results.get(controller)
        if result is None:
            print(f"⚠️ Warning: No result found for controller '{controller}'.")
        return result
