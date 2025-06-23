import pandas as pd
import numpy as np
import logging
from scipy.io import loadmat
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

def _array_to_datetime(arr):
    # 提取各字段（转换为整数）
    year = int(arr[0])
    month = int(arr[1])
    day = int(arr[2])
    hour = int(arr[3])
    minute = int(arr[4])
    
    # 处理秒和微秒
    seconds_total = arr[5]
    seconds = int(seconds_total)
    microseconds = int(round((seconds_total - seconds) * 1e6))  # 四舍五入
    
    # 返回 datetime 对象
    return datetime(year, month, day, hour, minute, seconds, microseconds)

class BaseLoader(ABC):
    def __init__(self, trial_id_col: str = 'TrialID'):
        """
        初始化加载器。
        
        Args:
            trial_id_col (str): 告诉加载器，在它加载的数据中，
                                  哪一列代表trial ID。默认为 'TrialID'。
        """
        self.trial_id_col = trial_id_col
        
    @abstractmethod
    def load(self, source, **kwargs) -> pd.DataFrame:
        pass

class MatLoader(BaseLoader):
    """
    为 MonkeyLogic 的 .mat 文件定制的加载器。
    在实例化时接收配置（如 notation_map），使其高度可配置。
    """
    def __init__(self, notation_map: dict = None, trial_id_col: str = 'TrialID'):
        super().__init__(trial_id_col=trial_id_col) 
        
        self.notation_map = notation_map or {}
        logging.info(f"MatLoader initialized with notation map: {list(self.notation_map.keys())}")

    def _get_trial_notation(self, trial_id: int) -> str:
        for name, (start, end) in self.notation_map.items():
            if start <= trial_id <= end:
                return name
        return "Unknown"

    def load(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        加载并解析.mat文件。
        【关键修复】: 不再静默处理FileNotFoundError。如果文件不存在，程序应立即报错。
        【关键修复】: 使用.get()方法安全地访问数据，避免因缺少键而崩溃。
        """
        logging.info(f"MatLoader: Loading from {filepath}...")
        try:
            data = loadmat(filepath, simplify_cells=True)
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            raise # 将异常抛出，让用户知道问题所在
            
        start_datetime = _array_to_datetime(data['Trial1']['TrialDateTime'])
        
        all_records = []
        total_trial_num = data['TrialRecord']['CurrentTrialNumber']
        
        for trial_id in range(1, total_trial_num):
            trial_key = f'Trial{trial_id}'
            if trial_key not in data:
                logging.warning(f"Trial {trial_id} not found in .mat file. Skipping.")
                continue

            trial_data = data[trial_key]
            try:
                start_time_ms = trial_data['AbsoluteTrialStartTime']
                
                # 【修复】使用.get()进行安全访问，提供默认值以防万一
                user_vars = trial_data.get('UserVars', {})
                variable_changes = trial_data.get('VariableChanges', {})
                delay_timing = variable_changes.get('delay_timing', [None, None])

                base_info = {
                    'TrialID': trial_id,
                    'TrialError': trial_data.get('TrialError', -1),
                    'Direction': user_vars.get('direction_thistrial', np.nan),
                    'Coherence': user_vars.get('rdm_coherence_thistrial', np.nan),
                    'DelayTime': delay_timing[1],
                    'Notation': self._get_trial_notation(trial_id)
                }

                behavioral_codes = trial_data.get('BehavioralCodes', {})
                codes = behavioral_codes.get('CodeNumbers', [])
                times = behavioral_codes.get('CodeTimes', [])

                for code, time_ms in zip(codes, times):
                    record = base_info.copy()
                    event_time_sec = (time_ms + start_time_ms) / 1000.0
                    record.update({
                        'BehavioralCode': code,
                        'EventTime': event_time_sec,
                        'AbsoluteDateTime': start_datetime + timedelta(seconds=event_time_sec)
                    })
                    all_records.append(record)
            except KeyError as e:
                logging.warning(f"KeyError {e} while processing Trial {trial_id}. Skipping trial.")

        return pd.DataFrame(all_records)

class DataFrameLoader(BaseLoader):
    """一个简单的加载器，直接使用传入的DataFrame作为数据源。"""

    def __init__(self, trial_id_col: str = None):
        # 如果用户没有为DataFrameLoader指定列名，我们假设它没有trial id列
        # 如果指定了，就传给父类
        super().__init__(trial_id_col=trial_id_col)
        logging.info(f"DataFrameLoader initialized. Expecting trial ID in column '{self.trial_id_col}'.")

    def load(self, df_source: pd.DataFrame, **kwargs) -> pd.DataFrame:
        logging.info("DataFrameLoader: Using pre-loaded DataFrame.")
        if 'EventTime' not in df_source.columns:
            raise ValueError("'EventTime' column is required in the source DataFrame.")
        
        df = df_source.copy()
        if 'AbsoluteDateTime' not in df.columns:
            df['AbsoluteDateTime'] = pd.NaT
        return df
