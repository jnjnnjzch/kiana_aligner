import dtw
import numpy as np
from collections import defaultdict

def get_pair_via_dtw(template, query, step_pattern="symmetric2", verbose=False):
    template = np.asarray(template)
    query = np.asarray(query)
    
    # Need at least 2 elements to compute diff
    if len(template) < 2 or len(query) < 2:
        # Return empty list for insufficient data
        # The diff-based DTW requires at least 2 points
        return []
    
    template = np.diff(template)
    query = np.diff(query)
    
    # Reshape to 2D for compatibility with dtw library
    template = template.reshape(-1, 1)
    query = query.reshape(-1, 1)
    
    dist_fun = lambda x_val, y_val: abs(x_val[0] - y_val[0])
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

# Constants for validation
RETRY_GROWTH_FACTOR = 1.5  # Increase chunk size by 50% per retry
VALIDATION_TOLERANCE_RATIO = 0.1  # Allow 10% inconsistent intervals


def get_pair_via_dtw_minimal(template, query, chunk_size=50, tolerance=0.01, 
                              step_pattern="symmetric2", verbose=False, max_attempts=3):
    """
    Minimal pair method for DTW alignment using forward sliding window approach.
    
    This method improves efficiency when dealing with long queues by:
    1. Processing the first chunk to establish pairing relationship
    2. Validating if this relationship holds for remaining data
    3. If validation fails at some point, re-chunk from that point
    4. Continue forward until all data is processed
    
    This is a forward sliding window process, not block processing, 
    avoiding overlap problems.
    
    Args:
        template: First sequence (reference), array of time points
        query: Second sequence (to be aligned), array of time points
        chunk_size (int): Number of elements to process in initial chunk. Default: 50
        tolerance (float): Tolerance for interval consistency check. Default: 0.01
        step_pattern (str): DTW step pattern. Default: "symmetric2"
        verbose (bool): Whether to print detailed information. Default: False
        max_attempts (int): Maximum attempts to get valid chunk pairing. Default: 3
        
    Returns:
        list: List of paired indices in format [(template_idx, query_idx), ...]
    """
    template = np.asarray(template)
    query = np.asarray(query)
    
    if len(template) == 0 or len(query) == 0:
        return []
    
    # If sequences are short enough, use regular DTW
    if len(template) <= chunk_size and len(query) <= chunk_size:
        return get_pair_via_dtw(template, query, step_pattern, verbose)
    
    if verbose:
        print(f"Starting forward sliding window DTW with chunk_size={chunk_size}, tolerance={tolerance}")
        print(f"Template length: {len(template)}, Query length: {len(query)}")
    
    all_pairs = []
    position = 0  # Current position in the sequences
    
    while position < len(template) and position < len(query):
        # Step 1: Process a chunk starting from current position
        chunk_end_template = min(position + chunk_size, len(template))
        chunk_end_query = min(position + chunk_size, len(query))
        
        # Ensure we have enough data for DTW
        if chunk_end_template - position < 2 or chunk_end_query - position < 2:
            break
        
        template_chunk = template[position:chunk_end_template]
        query_chunk = query[position:chunk_end_query]
        
        if verbose:
            print(f"\nProcessing chunk from position {position}")
            print(f"  Template: [{position}:{chunk_end_template}], Query: [{position}:{chunk_end_query}]")
        
        # Try to get valid pairing for this chunk with retries
        chunk_pairs = None
        current_chunk_size = chunk_size
        
        for attempt in range(max_attempts):
            if attempt > 0:
                # Retry with larger chunk
                current_chunk_size = min(
                    int(chunk_size * (RETRY_GROWTH_FACTOR ** attempt)),
                    len(template) - position,
                    len(query) - position
                )
                chunk_end_template = min(position + current_chunk_size, len(template))
                chunk_end_query = min(position + current_chunk_size, len(query))
                
                if chunk_end_template - position < 2 or chunk_end_query - position < 2:
                    break
                
                template_chunk = template[position:chunk_end_template]
                query_chunk = query[position:chunk_end_query]
                
                if verbose:
                    print(f"  Retry {attempt} with chunk_size={current_chunk_size}")
            
            # Get pairs for this chunk
            chunk_pairs = get_pair_via_dtw(template_chunk, query_chunk, step_pattern, verbose=False)
            
            # Validate chunk pairs
            if len(chunk_pairs) > 0 and _validate_pairs(template_chunk, query_chunk, chunk_pairs, tolerance):
                if verbose:
                    print(f"  Chunk validation passed with {len(chunk_pairs)} pairs")
                break
            elif verbose:
                print(f"  Chunk validation failed (attempt {attempt + 1})")
        
        if not chunk_pairs or len(chunk_pairs) == 0:
            if verbose:
                print(f"  No valid pairs found, skipping position {position}")
            position += max(1, chunk_size // 10)
            continue
        
        # Step 2: Add chunk pairs to results (adjust to global indices)
        for i, j in chunk_pairs:
            all_pairs.append((position + i, position + j))
        
        # Step 3: Determine where to start validating remaining data
        last_template_idx = chunk_pairs[-1][0]
        last_query_idx = chunk_pairs[-1][1]
        
        # Step 4: Validate if the pairing relationship holds for remaining data
        if verbose:
            print(f"  Last chunk pair: template[{last_template_idx}] -> query[{last_query_idx}]")
            remaining_count = len(template) - (position + last_template_idx + 1)
            print(f"  Validating {remaining_count} remaining template points")
        
        # Find where validation fails and get additional validated pairs
        failure_pos, validated_pairs = _find_validation_failure(
            template, query, all_pairs.copy(),  # Pass a copy to avoid mutation
            position + last_template_idx + 1, 
            position + last_query_idx + 1,
            tolerance, verbose
        )
        
        if failure_pos is None:
            # All remaining data validates! Add validated pairs and we're done
            all_pairs.extend(validated_pairs)
            if verbose:
                print(f"  All remaining data validates successfully!")
            break
        else:
            # Validation failed at failure_pos, add valid pairs up to failure point
            all_pairs.extend([(i, j) for i, j in validated_pairs if i < failure_pos])
            
            if verbose:
                print(f"  Validation failed at position {failure_pos}, re-chunking from there")
            
            # Restart from failure position
            position = failure_pos
    
    if verbose:
        print(f"\nCompleted forward sliding window DTW with {len(all_pairs)} total pairs")
    
    return all_pairs


def _find_validation_failure(template, query, existing_pairs, start_template, start_query, tolerance, verbose=False):
    """
    Find the first position where the established pairing relationship breaks down.
    
    Uses the existing pairs to predict where subsequent template points should map to,
    and checks if the actual relationship holds.
    
    Args:
        template: Full template sequence
        query: Full query sequence  
        existing_pairs: Existing validated pairs (not modified)
        start_template: Starting template index to validate from
        start_query: Starting query index to validate from
        tolerance: Tolerance for validation
        verbose: Print debug info
        
    Returns:
        tuple: (failure_position or None, list of validated pairs)
               - failure_position: Template index where validation first fails, or None if all validates
               - validated_pairs: List of newly validated pairs before any failure
    """
    validated_pairs = []
    
    if len(existing_pairs) < 2:
        # Not enough pairs to establish relationship
        return None, validated_pairs
    
    if start_template >= len(template):
        return None, validated_pairs  # No more data to validate
    
    # Calculate the time ratio from existing pairs
    template_indices = np.array([i for i, j in existing_pairs])
    query_indices = np.array([j for i, j in existing_pairs])
    
    # Get the slope from the last few pairs
    n_pairs = min(10, len(existing_pairs))
    recent_template = template[template_indices[-n_pairs:]]
    recent_query = query[query_indices[-n_pairs:]]
    
    # Calculate time differences
    if len(recent_template) < 2:
        return None, validated_pairs
    
    template_span = recent_template[-1] - recent_template[0]
    query_span = recent_query[-1] - recent_query[0]
    
    if template_span == 0:
        time_ratio = 1.0
    else:
        time_ratio = query_span / template_span
    
    # Now validate remaining points
    # For each template point, predict where it should be in query
    current_query_idx = start_query
    
    for template_idx in range(start_template, len(template)):
        if current_query_idx >= len(query):
            # Ran out of query data
            return template_idx, validated_pairs
        
        # Predict query position based on template time and ratio
        template_time = template[template_idx]
        predicted_query_time = query[query_indices[-1]] + (template_time - template[template_indices[-1]]) * time_ratio
        
        # Find the closest query point
        # Search in a reasonable window around predicted position
        search_start = max(start_query, current_query_idx)
        search_end = min(len(query), search_start + 20)
        
        if search_start >= search_end:
            return template_idx, validated_pairs
        
        query_window = query[search_start:search_end]
        query_times_diff = np.abs(query_window - predicted_query_time)
        best_match_in_window = np.argmin(query_times_diff)
        actual_query_idx = search_start + best_match_in_window
        actual_query_time = query[actual_query_idx]
        
        # Check if this match is within tolerance
        time_error = abs(actual_query_time - predicted_query_time)
        
        # Use relative tolerance
        expected_diff = abs(template_time - template[template_indices[-1]]) * time_ratio
        relative_tolerance = tolerance * max(1.0, expected_diff)
        
        if time_error > relative_tolerance:
            # Validation failed at this position
            if verbose:
                print(f"    Validation failure at template[{template_idx}]: "
                      f"predicted query time {predicted_query_time:.3f}, "
                      f"actual {actual_query_time:.3f}, error {time_error:.3f}")
            return template_idx, validated_pairs
        
        # Add this validated pair
        validated_pairs.append((template_idx, actual_query_idx))
        current_query_idx = actual_query_idx + 1
    
    # All validated successfully
    return None, validated_pairs


def _validate_pairs(template, query, pairs, tolerance=0.01):
    """
    Validate DTW pairs by checking interval consistency.
    
    Args:
        template: Template sequence (numpy array expected)
        query: Query sequence (numpy array expected)
        pairs: List of paired indices
        tolerance: Tolerance for interval consistency
        
    Returns:
        bool: True if pairs are valid, False otherwise
    """
    if len(pairs) < 3:
        # Not enough pairs to validate intervals meaningfully
        return True
    
    # Extract indices
    template_indices = np.array([i for i, j in pairs])
    query_indices = np.array([j for i, j in pairs])
    
    # Extract matched values using numpy indexing
    template_vals = template[template_indices]
    query_vals = query[query_indices]
    
    # Calculate interval differences
    template_diffs = np.diff(template_vals)
    query_diffs = np.diff(query_vals)
    
    # Guard against empty diffs (shouldn't happen with len(pairs) >= 3, but be safe)
    if len(template_diffs) == 0 or len(query_diffs) == 0:
        return True
    
    interval_diffs = query_diffs - template_diffs
    
    # Check if all intervals are within tolerance
    inconsistent = np.abs(interval_diffs) > tolerance
    
    # Allow up to VALIDATION_TOLERANCE_RATIO of intervals to be inconsistent
    inconsistent_ratio = np.sum(inconsistent) / len(interval_diffs)
    
    return inconsistent_ratio <= VALIDATION_TOLERANCE_RATIO


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