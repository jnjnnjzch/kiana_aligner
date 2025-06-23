import dtw
import numpy as np

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

def get_paired_ephys_event_index(task_ephys_dtw_pairs):
	"""
	将任务事件与电生理事件配对的索引转换为numpy数组，必须是task与ephys的pair，其中task在前，ephys在后
	"""
	# dtw_pair_dict = {template: query for template, query in reversed(dtw_pairs)}
	# s = pd.Series(dtw_pair_dict)
	# full_index = range(max(dtw_pair_dict.keys()) + 1)
	# final_series = s.reindex(full_index)
	# final_numpy_array = final_series.values

	max_task_id = max(pair[0] for pair in task_ephys_dtw_pairs)
	final_numpy_array = np.full(max_task_id + 1, np.nan)
	for task_event_id, ephys_event_id in task_ephys_dtw_pairs:
		if np.isnan(final_numpy_array[task_event_id]):
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