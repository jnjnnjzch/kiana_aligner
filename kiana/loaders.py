import pandas as pd
import numpy as np
import logging
from scipy.io import loadmat
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import struct
import os

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
    def __init__(self, notation_map: dict = None, trial_id_col: str = 'TrialID', load_all=False):
        super().__init__(trial_id_col=trial_id_col) 
        
        self.notation_map = notation_map or {}
        self.load_all = load_all
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
        增加load_all选项，允许用户选择是否加载所有UserVars和VariableChanges。
        """
        logging.info(f"MatLoader: Loading from {filepath}...")
        try:
            data = loadmat(filepath, simplify_cells=True)
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            raise # 将异常抛出，让用户知道问题所在
        if 'Trial1' not in data:
            raise ValueError(f"Invalid .mat file structure: Missing 'Trial1' key. Data keys: {data.keys()}")
        start_datetime = _array_to_datetime(data['Trial1']['TrialDateTime'])
        
        all_records = []
        if "TrialRecord" not in data:
            total_trial_num = 1
            while f"Trial{total_trial_num}" in data:
                total_trial_num += 1
        else:
            total_trial_num = data["TrialRecord"]["CurrentTrialNumber"]
        
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

                base_info = {
                    'TrialID': trial_id,
                    'TrialError': trial_data.get('TrialError', -1),
                    'Direction': user_vars.get('direction_thistrial', np.nan),
                    'Coherence': user_vars.get('rdm_coherence_thistrial', np.nan),
                    'DelayTime': variable_changes.get('delay_timing', [None, None]),
                    'TargetsID': user_vars.get('targets_id_thistrial', [None, None]),
                    'TargetChosen': user_vars.get('target_chosen', np.nan),
                    'TargetProbability': variable_changes.get('reward_probability', [None, None]),
                    'ReactionTime': trial_data.get('ReactionTime', np.nan),
                    'Notation': self._get_trial_notation(trial_id)
                }

                if self.load_all:
                    base_info.update(user_vars)
                    base_info.update(variable_changes)



                # AnalogData提取
                analog_data = trial_data.get('AnalogData', {})
                if self.load_all:
                    base_info.update(analog_data)
                
                def record_event(event_type, event_times):
                    for event_time in event_times:
                        record = base_info.copy()
                        event_time_sec = (event_time + start_time_ms) / 1000.0
                        record.update({
                            'BehavioralCode': event_type,
                            'EventTime': event_time_sec,
                            'AbsoluteDateTime': start_datetime + timedelta(seconds=event_time_sec)
                        })
                        all_records.append(record)
                
                touch_data = analog_data.get('Touch', np.array([]))
                button_data = analog_data.get('Button', {})
                if touch_data.size > 0:
                    touch_times = np.where((~pd.isna(np.array(trial_data['AnalogData']['Touch'])[:,0])==True))[0] # 触摸屏幕的时间，以ms采样，如果持续10ms，则结果类似于 [0, 1, 2, ..., 9]
                    touch_start_time = touch_times[np.where(np.diff(touch_times) > 1)[0]+1] # 找到触摸开始的时间点
                    touch_end_time = touch_times[np.where(np.diff(touch_times) > 1)[0]] # 找到触摸结束的时间点
                else:
                    touch_start_time = np.array([])
                    touch_end_time = np.array([])
                if button_data is not {}:
                    btn1_data = button_data.get('Btn1', []).astype(np.int8)  # 确保数据是整数类型
                    btn1_start_time = np.where(np.diff(btn1_data) == 1)[0] + 1  # 按下按钮的时间，以ms采样，如果持续10ms，则结果类似于 [0, 1, 2, ..., 9]
                    btn1_end_time = np.where(np.diff(btn1_data) == -1)[0] + 1  # 找到按钮释放的时间点
                else:
                    btn1_start_time = np.array([])
                    btn1_end_time = np.array([])

                record_event('TouchStart', touch_start_time)
                record_event('TouchEnd', touch_end_time)
                record_event('Button1Start', btn1_start_time)
                record_event('Button1End', btn1_end_time)

                # Behavioral提取
                behavioral_codes = trial_data.get('BehavioralCodes', {})
                codes = behavioral_codes.get('CodeNumbers', [])
                times = behavioral_codes.get('CodeTimes', [])

                for code, time_ms in zip(codes, times):
                    record_event(code, [time_ms])

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
    
class TrcLoader(BaseLoader):
    """一个简单的加载器，直接使用传入的trc文件作为数据源。"""

    def __init__(self, trial_id_col: str = None, pure: bool = False):
        super().__init__(trial_id_col=trial_id_col)
        self.pure = pure
        logging.info(f"TrcLoader initialized. Expecting trial ID in column '{self.trial_id_col}'.")

    def load(self, file_name: str, check: bool = True, **kwargs) -> pd.DataFrame:
        logging.info("TrcLoader: Using trc file.")

        total_hand_points = kwargs.get('total_hand_points', 8)
        total_body_points = kwargs.get('total_body_points', 4)
        body_name = kwargs.get('body_name', 'Body')
        hand_name = kwargs.get('hand_name', [])
        if type(hand_name) == list:
            if len(hand_name) == 0:
                total_hand_num = 1
            else:
                total_hand_num = len(hand_name)
        elif type(hand_name) == str:
            total_hand_num = 1
            hand_name = [hand_name]
        else:
            raise ValueError("hand_name should be either a string or a list of strings or an empty list.")
        if total_hand_num > 1 and type(total_hand_points) == int:
            total_hand_points = [total_hand_points] * total_hand_num
        elif total_hand_num == 1 and type(total_hand_points) == int:
            total_hand_points = [total_hand_points]

        # 这里只读取前6行头部
        header_lines = []
        with open(file_name, 'r') as file:
            for _ in range(6):
                header_lines.append(file.readline().strip())
        split_header = [line.split('\t') for line in header_lines]

        # 解析 config
        config = {}
        for i in range(len(split_header[1])):
            config[split_header[1][i]] = split_header[2][i]

        # 解析 part 和 num_labeled_points
        part = []
        num_labeled_points = 0
        for num_labeled_points in range(len(split_header[3])):
            if "unlabel" in split_header[3][num_labeled_points]:
                break
            elif len(split_header[3][num_labeled_points]) > 0 and num_labeled_points > 1:
                part.append(split_header[3][num_labeled_points])

        num_labeled_points = num_labeled_points - 2
        logging.info(f"num of labeled recorded points: {num_labeled_points}")
        logging.info(f"length of part: {len(part)}")
        for i in range(len(part)):
            logging.info(f"part{i+1}:{part[i]}")

        columns = ["frame", "time"] + split_header[4][:num_labeled_points]

        # 用 pandas 读取数据
        total_cols = num_labeled_points + 2
        rec = pd.read_csv(
            file_name,
            sep='\t',
            skiprows=6,
            header=None,
            usecols=range(total_cols),
            dtype=np.float64,  # 指定为 float32 会造成数值溢出，因此指定为 float64
            low_memory=False
        )
        rec.columns = columns
        rec["time"] = rec["time"] - rec["time"].values[0]

        print("finish loading and parsing")

        num_points = (rec.shape[1] - 2) // 3
        x_cols = [f"X{i+1}" for i in range(num_points)]
        y_cols = [f"Y{i+1}" for i in range(num_points)]
        z_cols = [f"Z{i+1}" for i in range(num_points)]
        sum_temp_all = (rec[x_cols].abs().sum(axis=0) +
                        rec[y_cols].abs().sum(axis=0) +
                        rec[z_cols].abs().sum(axis=0)).values
        idx_to_traverse = np.where(sum_temp_all != 0)[0]

        column_idx_with_data_hand = []
        column_idx_with_data_body = []

        # 改动3：添加索引保护，避免 part 越界（忽略未标记点）
        for i in idx_to_traverse:
            if i >= len(part):   # 如果当前点没有对应的标签名（即未标记点），直接跳过
                continue
            if body_name in part[i]:
                column_idx_with_data_body.append(i+1)

        if total_hand_num == 1:
            for i in idx_to_traverse:
                if i >= len(part):
                    continue
                if (body_name not in part[i]):
                    if (len(hand_name)==0) or (hand_name[0] in part[i]):
                        column_idx_with_data_hand.append(i+1)
        elif total_hand_num > 1:
            for j in range(total_hand_num):
                hand_j = []
                for i in idx_to_traverse:
                    if i >= len(part):
                        continue
                    if (body_name not in part[i]) and (hand_name[j] in part[i]):
                        hand_j.append(i+1)
                hand_j = sorted(hand_j)
                if check and (len(hand_j) != total_hand_points[j]):
                    raise ValueError(f"Number of detected hand points for {hand_name[j]} ({len(hand_j)}) does not match expected ({total_hand_points[j]}).")
                column_idx_with_data_hand.append(hand_j)

        if check:
            if (total_hand_num == 1) and (len(column_idx_with_data_hand) != total_hand_points[0]):
                raise ValueError(f"Number of detected hand points ({len(column_idx_with_data_hand)}) does not match expected ({total_hand_points}).")
            if len(column_idx_with_data_body) != total_body_points:
                raise ValueError(f"Number of detected body points ({len(column_idx_with_data_body)}) does not match expected ({total_body_points}).")

        trigger_points = np.array(column_idx_with_data_body)
        trigger_columns = np.array([[f"X{i}",f"Y{i}",f"Z{i}"] for i in trigger_points]).reshape([1,-1])
        for i in range(len(trigger_points)):
            # 同样增加保护，避免 part 越界（触发点一定在 part 范围内，但以防万一）
            idx = trigger_points[i] - 1
            if idx < len(part):
                logging.info(f"Choose point {trigger_points[i]} as {part[idx]}")
        logging.info(trigger_columns)

        if total_hand_num == 1:
            chosen_points = np.array(column_idx_with_data_hand)
            if len(column_idx_with_data_hand) == 0:
                chosen_points = np.array([1])
            chosen_columns = np.array([[f"X{i}",f"Y{i}",f"Z{i}"] for i in chosen_points]).reshape([1,-1])
            for i in range(len(chosen_points)):
                idx = chosen_points[i] - 1
                if idx < len(part):
                    logging.info(f"Choose point {chosen_points[i]} as {part[idx]}")
            logging.info(chosen_columns)

            hand_traj = pd.DataFrame()
            hand_traj['time'] = rec['time']
            hand_traj['frame'] = rec['frame']
            for i,item in enumerate(chosen_columns):
                hand_traj[item] = rec[item]

        else:
            chosen_points = [np.array(column_idx_with_data_hand[i]) for i in range(total_hand_num)]
            if len(column_idx_with_data_hand) == 0:
                raise ValueError("No hand points detected, but total_hand_pt_num > 1. Please check your data and configuration.")
            chosen_columns = [np.array([[f"X{i}",f"Y{i}",f"Z{i}"] for i in chosen_points[j]]).reshape([1,-1]) for j in range(total_hand_num)]
            for i in range(len(chosen_points)):
                for j in range(len(chosen_points[i])):
                    idx = chosen_points[i][j] - 1
                    if idx < len(part):
                        logging.info(f"Choose point {chosen_points[i][j]} as {part[idx]}")
            logging.info(chosen_columns)

            hand_traj = pd.DataFrame()
            hand_traj['time'] = rec['time']
            hand_traj['frame'] = rec['frame']
            for single_hand in chosen_columns:
                for item in single_hand:
                    hand_traj[item] = rec[item]

        print("finish checking, all good")

        trigger_df = pd.DataFrame()
        trigger_df['time'] = rec['time']
        trigger_df['frame'] = rec['frame']
        trigger_df["final_trigger"] = 0
        for i in column_idx_with_data_body:
            # 只有 body 点一定在 part 内，但为避免异常仍加保护
            idx = i - 1
            if idx < len(part):
                trigger_df[part[idx]] = np.where(rec[[f"X{i}",f"Y{i}",f"Z{i}"]].sum(axis=1) == 0, 0, 1)
            trigger_df["final_trigger"] = trigger_df["final_trigger"] + np.abs(rec[[f"X{i}",f"Y{i}",f"Z{i}"]].sum(axis=1))

        trigger_df["final_trigger"] = np.where(trigger_df["final_trigger"] == 0, 0, 1)
        trigger = trigger_df["final_trigger"].values
        diff = np.diff(np.concatenate(([0], trigger, [0])))
        onset_idx = np.where(diff == 1)[0] - 1
        offset_idx = np.where(diff == -1)[0] - 1

        trigger_df["label"] = "None"
        if len(onset_idx) > 0:
            trigger_df.loc[onset_idx, "label"] = "Trigger_Onset"
        if len(offset_idx) > 0:
            trigger_df.loc[offset_idx, "label"] = "Trigger_Offset"

        time_vals = trigger_df["time"].values
        data_rate = float(config["DataRate"])
        continous_trigger = []
        trigger_downside = []
        for start, end in zip(onset_idx, offset_idx):
            start_time = time_vals[start]
            end_time = time_vals[end]
            dict_temp = {
                "start": f"{start_time:.4f}",
                "end": f"{end_time:.4f}",
                "length": f"{end_time - start_time + 1.0/data_rate:.4f}"
            }
            continous_trigger.append(dict_temp)
            trigger_downside.append(end_time)

        print("finish packing")
        logging.info(continous_trigger)
        logging.info(trigger_downside)

        if self.pure:
            df = trigger_df[trigger_df["label"].str.contains("Trigger")]
        else:
            df = pd.merge(hand_traj, trigger_df, on=['time', 'frame'])

        df = df.rename(columns={'time': 'EventTime'})
        df['AbsoluteDateTime'] = pd.NaT

        return df

class SeqLoader(BaseLoader):
    """
    一个加载器，直接使用传入的seq文件作为数据源。
    使用异步预取方法提取时间戳，提升大文件处理效率。
    """

    def __init__(self, trial_id_col: str = None):
        super().__init__(trial_id_col=trial_id_col)
        logging.info(f"SeqLoader initialized. Expecting trial ID in column '{self.trial_id_col}'.")

    def load(self, file_name: str, **kwargs) -> pd.DataFrame:
        """
        加载并解析seq文件，提取时间戳信息。
        通过异步预取方法提升大文件处理效率。
        生成包含帧号、事件时间和参考时间的DataFrame。
        参考时间转换为指定时区的时间戳。
        允许用户通过kwargs: Timezone指定时区，默认为'Asia/Shanghai'。
        """
        logging.info("SeqLoader: Using seq file.")
        timestamps = self._extract_time_async_prefetch(file_name)
        frames = np.arange(1, len(timestamps) + 1)
        refer_time = timestamps - timestamps[0]
        df = pd.DataFrame({
            'frame': frames,
            'EventTime': refer_time,
            'ReferenceTime': pd.to_datetime(timestamps, unit='s')
        })

        # --- 可选优化：如果你需要特定的时区 (比如北京时间) ---
        TimeZone = kwargs.get('Timezone', 'Asia/Shanghai')
        df['ReferenceTime'] = df['ReferenceTime'].dt.tz_localize('UTC').dt.tz_convert(TimeZone)
        return df
    
    def _parser_seq_header(self, seq_file):
        with open(seq_file, 'rb') as f:
            header_bytes = f.read(1024)
        magicnumber = hex(struct.unpack_from('<L', header_bytes, 0)[0])
        if magicnumber != '0xfeed':
            raise ValueError("Not a valid .seq file")
        img_info = struct.unpack_from('<5L', header_bytes, 564)
        image_size_bytes = img_info[0] # Offset 564 (纯图像数据大小)
        total_frames = img_info[2]   # Offset 572
        true_img_size = img_info[4] # Offset 580 (包含padding的图像大小)
        padding_size = true_img_size - image_size_bytes - 8 # 8 bytes for timestamp
        return total_frames, true_img_size, image_size_bytes, padding_size
    
    def _extract_time_async_prefetch(self, SEQ_FILE, BATCH_SIZE=4096):
        HEADER_SIZE = 8192         
        TOTAL_FRAMES, STRIDE, IMG_SIZE, _ = self._parser_seq_header(SEQ_FILE)    
        dt = np.dtype([('sec', '<u4'), ('ms', '<u2'), ('us', '<u2')])
        raw_data = np.zeros(TOTAL_FRAMES, dtype=dt)
        BATCH_SIZE = min(BATCH_SIZE, TOTAL_FRAMES)
        with open(SEQ_FILE, 'rb', buffering=0) as f:
            fd = f.fileno()
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except:
                pass

            for batch_start in range(0, TOTAL_FRAMES, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, TOTAL_FRAMES)
                current_batch_size = batch_end - batch_start

                for i in range(current_batch_size):
                    frame_idx = batch_start + i
                    offset = HEADER_SIZE + (frame_idx * STRIDE) + IMG_SIZE
                    os.posix_fadvise(fd, offset, 8, os.POSIX_FADV_WILLNEED)

                for i in range(current_batch_size):
                    frame_idx = batch_start + i
                    offset = HEADER_SIZE + (frame_idx * STRIDE) + IMG_SIZE
                    data_bytes = os.pread(fd, 8, offset)
                    raw_data[frame_idx] = np.frombuffer(data_bytes, dtype=dt)[0]
        
        timestamps = raw_data['sec'] + raw_data['ms']/1000.0 + raw_data['us']/1.0e6
        return timestamps