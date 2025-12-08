import dtw
import numpy as np
from collections import defaultdict

def get_pair_via_dtw(template, query, step_pattern="symmetric2", verbose=False):
    template = np.diff(template)
    query = np.diff(query)
    dist_fun = lambda x_val, y_val: abs(x_val - y_val)
    alignment_default = dtw.dtw(template, query,
                        dist_method=dist_fun,
                        step_pattern=step_pattern, # 或者 rabinerJuangStepPattern(6, "c"))\
                        keep_internals=True)
    # 获取结果
    distance_default = alignment_default.distance         # DTW距离
    path_s1_default = alignment_default.index1            # s1 中点的索引序列
    path_s2_default = alignment_default.index2            # s2 中点的索引序列
    cost_matrix_default = alignment_default.costMatrix    # 累积代价矩阵
    local_cost_matrix_default = alignment_default.localCostMatrix # 局部代价矩阵


    path_pairs_default = list(zip(path_s1_default, path_s2_default))

    # 希望在最前面增加一个(-1,-1), 因为之前采取的是差值匹配，会少一个数值
    path_pairs_default = [(-1, -1)] + path_pairs_default
    # 对所有元素+1，这样回归到从0开始的索引
    path_pairs_default = [(i + 1, j + 1) for i, j in path_pairs_default]

    if verbose:
        print(f"--- 使用 dtw-python ( {step_pattern} 步进模式) ---")
        print(f"DTW 距离: {distance_default:.2f}")
        dtw.dtwPlot(alignment_default,type="twoway")
        dtw.dtwPlot(alignment_default,type="density")
        print(f"匹配结果（template点的id, query点的id） :\n {path_pairs_default}")
        # print(rabinerJuangStepPattern(6,"c"))
        # rabinerJuangStepPattern(6,"c").plot()

    return path_pairs_default

def _purify_pairs(pairs, key_index=0):
    """
    一个通用的配对列表清理函数，确保基于 key_index 的唯一性。

    Args:
        pairs (list): (id_A, id_B) 格式的配对列表。
        key_index (int): 指定作为Key的索引 (0 或 1)。

    Returns:
        list: 清理后的配对列表。
    """
    if not pairs:
        return []

    value_index = 1 - key_index
    
    # 按指定的 key 进行分组
    key_to_values_map = defaultdict(list)
    for pair in pairs:
        key = pair[key_index]
        value = pair[value_index]
        key_to_values_map[key].append(value)
        
    # 解决冲突，并构建纯净的配对列表
    purified_pairs = []
    for key, values in key_to_values_map.items():
        # 决策规则：保留value最小的那个
        best_value = min(values)
        
        # 重新组装配对，注意保持原始的 (id_A, id_B) 顺序
        if key_index == 0:
            purified_pairs.append((key, best_value))
        else:
            purified_pairs.append((best_value, key))
            
    return purified_pairs

def get_paired_ephys_event_index(task_ephys_dtw_pairs, conservative=False):
    """
    【最终重构版】
    将DTW配对转换为task索引到ephys索引的映射数组。
    通过连续调用两次通用清理函数来确保最终映射的双向唯一性。
    """
    # 原始DTW配对可能包含两种冲突
    
    if not conservative:
        # 第一步：清理“一个Ephys对多个Task”的冲突 (key_index=1, 按ephys_id分组)
        clean_pairs_stage1 = _purify_pairs(task_ephys_dtw_pairs, key_index=1)

        # 第二步：清理“一个Task对多个Ephys”的冲突 (key_index=0, 按task_id分组)
        # 输入是上一阶段的输出
        final_clean_pairs = _purify_pairs(clean_pairs_stage1, key_index=0)
    else:
        # 保守模式下，直接进行第二步清理
        final_clean_pairs = _purify_pairs(task_ephys_dtw_pairs, key_index=0)
    
    # 第三步：使用完全纯净的配对列表构建最终数组
    if not final_clean_pairs:
        return np.array([])
        
    max_task_id = max(pair[0] for pair in final_clean_pairs)
    final_numpy_array = np.full(max_task_id + 1, np.nan)
    
    # 因为 final_clean_pairs 中的 task_event_id 已经是唯一的，所以可以直接赋值
    for task_event_id, ephys_event_id in final_clean_pairs:
        final_numpy_array[task_event_id] = ephys_event_id

    return final_numpy_array

def get_spikes_in_windows(spike_train, event_windows):
    """
    使用列表推导式，更简洁地提取窗口内的相对spike time。
    """
    spike_train = np.asarray(spike_train)
    event_windows = np.asarray(event_windows)

    # 一行列表推导式完成所有操作
    return [
        spike_train[(spike_train >= start) & (spike_train < end)] - start
        for start, end in event_windows
    ]