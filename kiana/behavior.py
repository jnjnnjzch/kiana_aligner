import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional, List

# 从包内其他模块导入
from .loaders import BaseLoader
from .utils import get_pair_via_dtw, get_paired_ephys_event_index

class SegmentBuilder:
    """这是一个内部辅助类，用于实现链式调用，请勿直接实例化。"""
    def __init__(self, processor, segment_config: dict):
        self._processor = processor
        self._config = segment_config

    def with_slicing(self, rules: dict):
        """为当前数据段添加切片规则。"""
        self._config['slice_by'] = rules
        return self

    def with_anchors(self, query: str):
        """为当前数据段添加锚点查询语句。"""
        self._config['anchor_query'] = query
        return self

    def with_kwargs(self, **kwargs):
        """为当前段的Loader.load()提供额外的关键字参数。"""
        self._config['kwargs'] = kwargs
        return self
    

class BehavioralProcessor:
    """一个配方驱动的、支持多源数据整合与多上下文同步的平台。"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.recipe = None
        self.master_timeline_df = pd.DataFrame()
        self.solved_offsets = {}
        self._recipe_items = [] # 【新增】用于存放链式调用生成的配置
        self._is_timeline_built = False
        logging.info("BehavioralProcessor initialized.")

    # --- 全新的API ---
    def add_segment(self, segment_name: str, loader: BaseLoader, source) -> SegmentBuilder:
        """开始添加一个新的数据段，并返回一个配置构建器以进行链式调用。"""
        segment_config = {
            'segment_name': segment_name,
            'loader': loader,
            'source': source
        }
        self._recipe_items.append(segment_config)
        return SegmentBuilder(self, segment_config)

    def build(self):
        """根据所有已添加的数据段，最终构建会话。"""
        if not self._recipe_items:
            raise ValueError("No segments have been added. Please use .add_segment() first.")
        self.build_from_recipe(self._recipe_items)
        return self

    def build_from_recipe(self, recipe: list):
        """
        根据配方加载所有数据段，并提取锚点。
        【关键修正 2025-06-18 v4】: slice_by 逻辑被完全重写，使其变得通用和强大。
                                    - 支持对任意列名进行切片。
                                    - 支持元组(tuple)进行范围筛选和列表(list)进行离散值筛选。
        """
        logging.info("--- Building session from recipe... ---")
        self.recipe = recipe
        all_segments_list = []
        raw_sources_cache = {}

        for item in self.recipe:
            segment_name = item['segment_name']
            loader = item['loader']
            source = item['source']
            logging.info(f"Processing recipe item: '{segment_name}'")

            source_key = str(source) if isinstance(source, Path) else id(source)
            if source_key in raw_sources_cache:
                full_events_df = raw_sources_cache[source_key].copy()
            else:
                full_events_df = loader.load(source, **item.get('kwargs', {}))
                raw_sources_cache[source_key] = full_events_df
            
            segment_df = full_events_df # 先将完整数据赋值给 segment_df

            # 1. 首先，执行切片逻辑 (使用原始列名)
            # 【全新、灵活的切片逻辑】
            slice_rules = item.get('slice_by')
            if slice_rules and isinstance(slice_rules, dict):
                logging.info(f"  Applying slice rules: {slice_rules}")
                # 迭代字典中的所有规则，逐一应用筛选
                for column, criteria in slice_rules.items():
                    if column not in segment_df.columns:
                        logging.warning(f"  Slice column '{column}' not found. Skipping this rule.")
                        continue          
                    # 规则1: criteria 是一个双元素元组 -> 范围筛选
                    if isinstance(criteria, tuple) and len(criteria) == 2:
                        start, end = criteria
                        logging.info(f"    Slicing '{column}' for range >= {start} and <= {end}")
                        segment_df = segment_df[(segment_df[column] >= start) & (segment_df[column] <= end)]
                    # 规则2: criteria 是一个列表 -> 离散值筛选
                    elif isinstance(criteria, list):
                        logging.info(f"    Slicing '{column}' for discrete values in {criteria}")
                        segment_df = segment_df[segment_df[column].isin(criteria)]
                    # 规则3: 其他情况 -> 精确值匹配
                    else:
                        logging.info(f"    Slicing '{column}' for exact value == {criteria}")
                        segment_df = segment_df[segment_df[column] == criteria]
            
            # .copy() 确保我们操作的是一个独立的DataFrame，避免后续出现SettingWithCopyWarning
            segment_df = segment_df.copy()

            # 2. 【移动到这里】然后，标记 Anchor (同样使用原始列名)
            # 这样用户就可以在 query 中使用 'MyCustomID' 而不是 'TrialID'
            segment_df['is_anchor'] = False
            segment_df.sort_values('EventTime', inplace=True, ignore_index=True)

            anchor_query = item.get('anchor_query')
            if anchor_query:
                logging.info(f"  Applying anchor query: \"{anchor_query}\"")
                # query可能会在空DataFrame上失败，增加检查
                if not segment_df.empty:
                    anchors = segment_df.query(anchor_query)
                    segment_df.loc[anchors.index, 'is_anchor'] = True
            else:
                logging.info("  No anchor query, all events in segment are anchors.")
                segment_df['is_anchor'] = True

            # 3. 【移动到后面】在所有筛选和标记都完成后，才处理 Trial ID 的重命名，为统一输出做准备
            trial_id_col_from_loader = item['loader'].trial_id_col
    
            if trial_id_col_from_loader:
                if trial_id_col_from_loader in segment_df.columns:
                    # 如果loader指定的列名不是我们内部的标准名'TrialID'，则重命名
                    if trial_id_col_from_loader != 'TrialID':
                        logging.info(f"  Found trial ID column '{trial_id_col_from_loader}'. Renaming to 'TrialID'.")
                        segment_df.rename(columns={trial_id_col_from_loader: 'TrialID'}, inplace=True)
                    else:
                        logging.info(f"  Using standard 'TrialID' column.")
                else:
                    logging.warning(f"  Loader expected trial ID column '{trial_id_col_from_loader}' but it was not found.")

            # 4. 最后，添加元数据并准备合并
            segment_df['segment_name'] = segment_name
            all_segments_list.append(segment_df)
        
        self.master_timeline_df = pd.concat(all_segments_list, ignore_index=True)
        
        self._is_timeline_built = False
        logging.info("--- Session recipe built successfully. ---")
        return self

    def _solve_and_build_timeline_once(self, master_ephys_anchor_times):
        """
        首次同步时，求解各段偏移量并构建统一的主时间轴。
        【关键修正 2025-06-18 v5】: 移除了过于严格的DTW映射长度检查。
                                    现在可以优雅地处理部分锚点未能匹配的情况，
                                    只有在匹配完全失败时才报错。
        """
        logging.info("--- Solving offsets and building master timeline for the first time... ---")
        
        master_ephys_anchor_times_np = np.asarray(master_ephys_anchor_times)

        unaligned_anchors = self.master_timeline_df[self.master_timeline_df['is_anchor']].copy()
        if unaligned_anchors.empty:
            raise ValueError("No anchors found based on the recipe. Cannot build timeline.")

        beh_anchor_times = unaligned_anchors['EventTime'].values
        
        pairings = get_pair_via_dtw(beh_anchor_times, master_ephys_anchor_times_np)
        beh_to_ephys_mapping = get_paired_ephys_event_index(pairings)
        
        # 【修正点 1】: 不再进行严格的长度检查。
        # 而是将可能不等长的mapping结果安全地转换为一个Series，
        # 并使用 reindex 将其扩展到与所有行为锚点相同的长度，未匹配的会自动填充NaN。
        mapping_series = pd.Series(beh_to_ephys_mapping)
        target_index = pd.RangeIndex(len(beh_anchor_times))
        aligned_mapping = mapping_series.reindex(target_index)

        # 将这个干净、等长的Series赋值给DataFrame列
        # .values 确保按位置赋值，忽略Series的索引
        unaligned_anchors['ephys_idx'] = aligned_mapping.values
        
        # 【修正点 2】: 新的、更有意义的检查。
        # 只有在DTW完全失败，一个有效的配对都没找到的情况下，才抛出错误。
        if unaligned_anchors['ephys_idx'].isna().all():
            raise RuntimeError("DTW alignment failed completely. No valid anchor pairings were found.")

        # 后续的 dropna 会自然地处理掉那些因 reindex 产生的或原有的 NaN 值
        unaligned_anchors.dropna(subset=['ephys_idx'], inplace=True)
        unaligned_anchors['ephys_idx'] = unaligned_anchors['ephys_idx'].astype(int)

        valid_ephys_indices = unaligned_anchors['ephys_idx'].values
        unaligned_anchors['offset'] = master_ephys_anchor_times_np[valid_ephys_indices] - unaligned_anchors['EventTime'].values
        
        self.solved_offsets = unaligned_anchors.groupby('segment_name')['offset'].mean().to_dict()
        logging.info(f"Solved offsets: {self.solved_offsets}")

        time_offsets = self.master_timeline_df['segment_name'].map(self.solved_offsets).fillna(0)
        self.master_timeline_df['EventTime'] = self.master_timeline_df['EventTime'] + time_offsets
        
        self.master_timeline_df.sort_values('EventTime', inplace=True, ignore_index=True)
        
        self._is_timeline_built = True
        logging.info("--- Master timeline built successfully. ---")

        self._resolve_hetero_datetimes()

    def _resolve_hetero_datetimes(self):
        """
        为没有绝对时间的事件（异质数据）推算其datetime。
        
        此方法基于“刚性平移”原则：在已对齐的物理时间轴上，为未知事件
        寻找时间上最近的已知事件，然后利用它们的物理时间差来推算日历时间。
        【已修复】: 修复了因错误访问不存在的列而导致的 KeyError。
        """
        logging.info("Resolving datetimes for heterogeneous data...")
        
        # 1. 分离出拥有已知datetime的“参照物”和未知datetime的“待解析”事件
        known_dt_mask = self.master_timeline_df['AbsoluteDateTime'].notna()
        
        # 边缘情况处理：如果没有参照物或没有需要解析的事件，则直接退出
        if not known_dt_mask.any():
            print("No events with known datetime found to use as reference. Skipping datetime resolution.")
            logging.warning("No events with known datetime found to use as reference. Skipping datetime resolution.")
            return
        
        unknown_dt_indices = self.master_timeline_df.index[~known_dt_mask]
        if unknown_dt_indices.empty:
            logging.info("No heterogeneous datetimes to resolve.")
            return

        # 2. 准备左侧DataFrame：需要推算时间的事件
        events_to_resolve = self.master_timeline_df.loc[unknown_dt_indices, ['EventTime']].copy().sort_values('EventTime')
        
        # 3. 准备右侧DataFrame：作为参照物的事件
        ref_events = self.master_timeline_df.loc[known_dt_mask, ['EventTime', 'AbsoluteDateTime']].copy().sort_values('EventTime')
        
        # 4. 【关键步骤】重命名参照物中的列，以避免在合并时与左侧DataFrame的列名冲突
        ref_events.rename(columns={'EventTime': 'EventTime_ref', 'AbsoluteDateTime': 'AbsoluteDateTime_ref'}, inplace=True)

        # 5. 使用pd.merge_asof找到每个未知事件在时间上“最近的”参照物
        merged = pd.merge_asof(
            left=events_to_resolve,
            right=ref_events,
            left_on='EventTime',      # 左侧用 'EventTime' 作为键
            right_on='EventTime_ref', # 右侧用 'EventTime_ref' 作为键
            direction='nearest'       # 匹配策略：寻找最近的
        )

        # 6. 计算物理时间差，并将其应用到参照物的日历时间上（刚性平移）
        time_delta = pd.to_timedelta(merged['EventTime'] - merged['EventTime_ref'], unit='s')
        merged['ResolvedDateTime'] = merged['AbsoluteDateTime_ref'] + time_delta
        
        # 7. 将推算出的日历时间写回到主DataFrame中
        self.master_timeline_df.loc[events_to_resolve.index, 'AbsoluteDateTime'] = merged['ResolvedDateTime'].values
        
        logging.info(f"Resolved datetimes for {len(unknown_dt_indices)} events.")

    def _perform_match(
        self,
        events_to_sync: pd.DataFrame,
        anchors_to_use: pd.DataFrame,
        context_name: str,
        sampling_rate: int,
        match_by: list = None
    ) -> pd.DataFrame:
        """
        【第四版方案 v4.6 - 最终修正】
        根据用户洞察，移除了对TrialID的fillna(-1)预处理。
        此版本信任pandas原生处理NaN的能力，避免因数据重排导致的未知ValueError。
        """
        time_col = f'EphysTime_{context_name}'
        indice_col = f'EphysIndice_{context_name}'

        events_left = events_to_sync.copy()
        anchors_right = anchors_to_use.copy()

        # --- 【关键修正】移除对TrialID的预处理，让pandas原生处理NaN ---
        # if match_by:
        #     for key in match_by:
        #         if key == 'TrialID':
        #             events_left[key] = pd.to_numeric(events_left[key], errors='coerce').fillna(-1).astype(np.int64)
        #             anchors_right[key] = pd.to_numeric(anchors_right[key], errors='coerce').fillna(-1).astype(np.int64)
        # --- 修正结束 ---

        # 强制转换EventTime为数字类型，并移除无效行
        events_left['EventTime'] = pd.to_numeric(events_left['EventTime'], errors='coerce')
        anchors_right['EventTime'] = pd.to_numeric(anchors_right['EventTime'], errors='coerce')
        events_left.dropna(subset=['EventTime'], inplace=True)
        anchors_right.dropna(subset=['EventTime'], inplace=True)

        cols_to_use = ['EventTime', time_col, indice_col]
        if match_by:
            cols_to_use.extend(match_by)
        anchor_subset = anchors_right.reindex(columns=cols_to_use).copy()
        rename_dict = {c: f"{c}_ref" for c in anchor_subset.columns if c not in (match_by or [])}
        anchor_subset.rename(columns=rename_dict, inplace=True)
        
        sort_keys = ['EventTime']
        if match_by:
            sort_keys = match_by + sort_keys
        events_sorted = events_left.sort_values(sort_keys)
        sort_keys_ref = [key + '_ref' if key not in (match_by or []) else key for key in sort_keys]
        
        anchor_subset.drop_duplicates(subset=sort_keys_ref, keep='first', inplace=True)
        anchor_subset_sorted = anchor_subset.sort_values(sort_keys_ref)

        merged = pd.merge_asof(
            left=events_sorted.reset_index(),
            right=anchor_subset_sorted.reset_index(drop=True),
            left_on='EventTime',
            right_on='EventTime_ref',
            by=match_by,
            direction='nearest'
        ).set_index('index')

        valid_merges = merged[f'{time_col}_ref'].notna()
        time_delta = merged.loc[valid_merges, 'EventTime'] - merged.loc[valid_merges, 'EventTime_ref']
        merged.loc[valid_merges, time_col] = merged.loc[valid_merges, f'{time_col}_ref'] + time_delta
        merged.loc[valid_merges, indice_col] = merged.loc[valid_merges, f'{indice_col}_ref'] + (time_delta * sampling_rate)
        
        return merged[[time_col, indice_col]]

    def _apply_rigid_translation(self, context_name: str, sampling_rate: int, sync_within_trial: bool):
        """
        【第八版方案 - 最终版】
        遵从最终决定，采用最显式、最防御性的逻辑。
        在循环中为每个Segment彻底分离有无TrialID的事件，分别处理，杜绝一切歧义。
        """
        logging.info(f"Applying rigid translation for context '{context_name}' using 'Version 8.0 Plan'...")
        time_col = f'EphysTime_{context_name}'
        indice_col = f'EphysIndice_{context_name}'

        # --- 1. 数据准备 ---
        all_anchors = self.master_timeline_df.loc[self.master_timeline_df['is_anchor']].copy()
        if all_anchors.get(time_col, pd.Series(dtype=float)).isna().all():
            logging.warning(f"No primary anchors have valid time for '{context_name}'. Skipping.")
            return

        non_anchors_df = self.master_timeline_df[~self.master_timeline_df['is_anchor']]
        if non_anchors_df.empty:
            logging.info("No non-anchor events to process.")
            return

        processed_results = []

        # --- 2. 【核心】按Segment为单位，循环处理 ---
        for seg_name, segment_events in non_anchors_df.groupby('segment_name'):
            
            logging.info(f"Processing segment: '{seg_name}'...")
            local_anchors = all_anchors[all_anchors['segment_name'] == seg_name]
            
            # --- 2a. 决策路径：按优先级判断本Segment的匹配策略 ---

            # 策略3的条件: 本段内无可用锚点 -> 全局回退
            if local_anchors.get(time_col, pd.Series(dtype=float)).isna().all():
                logging.info(f"  -> No local anchors. Path: Global Fallback.")
                segment_result = self._perform_match(
                    events_to_sync=segment_events, anchors_to_use=all_anchors,
                    context_name=context_name, sampling_rate=sampling_rate,
                    match_by=None
                )
                processed_results.append(segment_result)
                continue

            # 策略1和2的条件：本段内有锚点
            can_use_trial_sync = (
                sync_within_trial and
                'TrialID' in segment_events.columns and
                'TrialID' in local_anchors.columns
            )

            # 如果不能进行Trial同步，则整个segment直接按Segment级别匹配
            if not can_use_trial_sync:
                logging.info(f"  -> Trial sync not applicable. Path: Segment-level Match.")
                segment_result = self._perform_match(
                    events_to_sync=segment_events, anchors_to_use=local_anchors,
                    context_name=context_name, sampling_rate=sampling_rate,
                    match_by=None
                )
                processed_results.append(segment_result)
                continue

            # --- 核心分离逻辑：如果能进行Trial同步，则显式分离数据流 ---
            logging.info(f"  -> Trial sync applicable. Explicitly separating event groups.")
            
            # 分离出有/无TrialID的事件
            has_trial_id_mask = segment_events['TrialID'].notna()
            trialed_events = segment_events[has_trial_id_mask]
            nontrialed_events = segment_events[~has_trial_id_mask]
            
            segment_processed_parts = []

            # 1. 处理有TrialID的事件 (A-1组)
            if not trialed_events.empty:
                trialed_anchors_in_segment = local_anchors.dropna(subset=['TrialID'])
                if not trialed_anchors_in_segment.empty:
                    logging.info(f"    -> Processing {len(trialed_events)} events with TrialID...")
                    # 尝试按Trial匹配
                    trial_match_result = self._perform_match(
                        events_to_sync=trialed_events, anchors_to_use=trialed_anchors_in_segment,
                        context_name=context_name, sampling_rate=sampling_rate,
                        match_by=['TrialID']
                    )
                    # 对失败者进行Segment降级匹配
                    failed_mask = trial_match_result[time_col].isna()
                    if failed_mask.any():
                        logging.info(f"    -> {failed_mask.sum()} events failed trial-match, falling back to segment-level...")
                        fallback_result = self._perform_match(
                            events_to_sync=trialed_events[failed_mask], anchors_to_use=local_anchors,
                            context_name=context_name, sampling_rate=sampling_rate,
                            match_by=None
                        )
                        trial_match_result.update(fallback_result)
                    segment_processed_parts.append(trial_match_result)
                else:
                    # 如果段内有Trial事件，但没有任何带Trial的锚点，则这些事件也直接进入段内匹配
                    nontrialed_events = pd.concat([nontrialed_events, trialed_events])

            # 2. 处理无TrialID的事件 (A-2组)
            if not nontrialed_events.empty:
                logging.info(f"    -> Processing {len(nontrialed_events)} events without TrialID at segment-level...")
                nontrialed_result = self._perform_match(
                    events_to_sync=nontrialed_events, anchors_to_use=local_anchors,
                    context_name=context_name, sampling_rate=sampling_rate,
                    match_by=None
                )
                segment_processed_parts.append(nontrialed_result)
            
            # 3. 合并本segment的处理结果
            if segment_processed_parts:
                processed_results.append(pd.concat(segment_processed_parts))

        # --- 3. 最终整合与更新 ---
        if not processed_results:
            logging.warning("No events were processed in rigid translation.")
            return

        final_results = pd.concat(processed_results)
        self.master_timeline_df.update(final_results.dropna(how='all'))
        
        valid_indices = self.master_timeline_df.get(indice_col, pd.Series(dtype=float)).notna()
        if valid_indices.any():
            self.master_timeline_df.loc[valid_indices, indice_col] = self.master_timeline_df.loc[valid_indices, indice_col].round().astype('Int64')

        logging.info(f"Rigid translation for '{context_name}' complete.")

    def _check_match_error(self, context_name: str, tolerance: float = 0.01):
        """
        【最终实现版本 - 2026-01-03】
        综合锚点健康检查工具。
        逻辑重构：
        1. 错误展示：保留 Contextual Anchor Data 表格，用于展示被污染的行（Dirty Set）。
        2. 安全报告：基于“Dirty Set”的补集计算“Safe Intervals”，并按 Trial 逻辑进行三明治合并报告。
        """
        import numpy as np
        import pandas as pd
        from itertools import groupby
        from operator import itemgetter

        logging.info(f"Performing comprehensive anchor check for context '{context_name}'...")
        ephys_time_col = f'EphysTime_{context_name}'
        ephys_indice_col = f'EphysIndice_{context_name}'

        all_anchors_df = self.master_timeline_df[self.master_timeline_df['is_anchor']].copy()

        # --- 第一部分：报告“匹配失败” (质检员A) ---
        pairing_failures_df = all_anchors_df[all_anchors_df[ephys_time_col].isna()]

        if not pairing_failures_df.empty:
            logging.info(f"  -> Found {len(pairing_failures_df)} anchors that failed to pair with ephys signals.")
            logging.info("\n--- [WARNING] ANCHOR PAIRING FAILURE REPORT ---")
            logging.info(f"Context: '{context_name}'")
            with pd.option_context('display.max_rows', None, 'display.width', 150):
                logging.info(pairing_failures_df)
            logging.info("--- END OF PAIRING FAILURE REPORT ---\n")
        # --- 第二部分：报告“性能不一致” (质检员B) ---
        paired_anchors_df = all_anchors_df.dropna(subset=[ephys_time_col]).copy()

        if len(paired_anchors_df) < 2:
            logging.info("  -> Less than 2 successfully paired anchors. Skipping interval consistency check.")
            return

        paired_anchors_df.sort_values('EventTime', inplace=True, ignore_index=True)

        beh_times = paired_anchors_df['EventTime'].to_numpy()
        ephys_times = paired_anchors_df[ephys_time_col].to_numpy()
        
        # 计算差异
        interval_diffs = np.diff(ephys_times) - np.diff(beh_times)
        inconsistent_indices = np.where(
            (interval_diffs > tolerance) | (interval_diffs < -tolerance)
        )[0]

        if len(inconsistent_indices) > 0:
            logging.info(f"  -> Found {len(inconsistent_indices)} inconsistent anchor intervals.")
            logging.info("\n--- [WARNING] ANCHOR TIMING INCONSISTENCY REPORT ---")
            print(f"Context: '{context_name}'")

            # ---------------------------------------------------------
            # 步骤 1: 确定“污染区” (The Dirty Set)
            # 逻辑：直接复用 Context 表的显示逻辑，凡是表里出现的，都算污染
            # ---------------------------------------------------------
            indices_to_show = set()
            for idx in inconsistent_indices:
                start = max(0, idx - 1)
                end = min(len(paired_anchors_df), idx + 2)
                indices_to_show.update(range(start, end))
            
            # 打印 Context 表 (这是你的原始需求，保留作为详细排查证据)
            logging.info("\nPART 1: Contextual Anchor Data (The Dirty Set)")
            context_df = paired_anchors_df.iloc[sorted(list(indices_to_show))]
            
            # 列排序优化
            preferred_order = []
            if 'TrialID' in context_df.columns: preferred_order.append('TrialID')
            preferred_order.extend(['EventTime', ephys_time_col, ephys_indice_col])
            new_col_order = preferred_order + [c for c in context_df.columns if c not in preferred_order]
            
            with pd.option_context('display.max_rows', None, 'display.width', 150, 'display.precision', 4):
                logging.info(context_df[new_col_order])

            # ---------------------------------------------------------
            # 步骤 2: 计算“安全区” (The Safe Set) —— 核心补集逻辑
            # ---------------------------------------------------------
            total_indices = set(range(len(paired_anchors_df)))
            safe_indices = sorted(list(total_indices - indices_to_show))
            safe_indices_set = set(safe_indices) # 用于快速查找

            # 辅助函数：List[int] -> "Start-End" 字符串
            def _to_range_str(nums):
                if not nums: return ""
                ranges = []
                for k, g in groupby(enumerate(nums), lambda x: x[0]-x[1]):
                    group = list(map(itemgetter(1), g))
                    ranges.append(f"{group[0]}-{group[-1]}" if group[0] != group[-1] else str(group[0]))
                return ", ".join(ranges)

            logging.info("\nPART 2: Safe Interval Report (The Clean Set)")
            
            # 2a. 输出第一行：物理 Index
            print(f"Index: {_to_range_str(safe_indices)}")

            # ---------------------------------------------------------
            # 步骤 3: 诊断 Trial 状态 & 三明治合并 (Sandwich Logic)
            # ---------------------------------------------------------
            if 'TrialID' in paired_anchors_df.columns:
                # 3a. 诊断阶段 (Diagnose)
                trial_nodes = []
                # 按 TrialID 及其连续性分组 (防止 T1...T2...T1 这种情况被错误合并)
                group_key = (paired_anchors_df['TrialID'].fillna('__NAN__') != paired_anchors_df['TrialID'].fillna('__NAN__').shift()).cumsum()
                
                for _, group in paired_anchors_df.groupby(group_key):
                    tid = group['TrialID'].iloc[0]
                    is_nan_block = pd.isna(tid)
                    
                    # 该 Trial 在原始数据中的所有 index
                    full_indices = group.index.tolist()
                    # 该 Trial 幸存下来的 safe index
                    valid_indices = [i for i in full_indices if i in safe_indices_set]
                    
                    if not valid_indices:
                        continue # 如果整个 Trial 都被污染了，就不出现在 Safe Report 里
                    
                    range_str = _to_range_str(valid_indices)
                    
                    if is_nan_block:
                        trial_nodes.append({
                            'type': 'nan', 
                            'str': f"nan({range_str})", 
                            'id': None, 
                            'status': 'nan'
                        })
                    else:
                        # 判断完整性: 幸存数量 == 原始数量
                        # (注意：因为 Indices_to_show 包含连续区间，这里简单比较长度即可，
                        # 除非 indices_to_show 恰好只挖掉了中间一个点，那也是 Partial)
                        is_perfect = (len(valid_indices) == len(full_indices))
                        
                        display_str = f"T{int(tid)}" if is_perfect else f"T{int(tid)}({range_str})"
                        status = 'perfect' if is_perfect else 'partial'
                        
                        trial_nodes.append({
                            'type': 'trial',
                            'str': display_str,
                            'id': int(tid),
                            'status': status
                        })

                # 3b. 组装阶段 (Assemble with Sandwich Rule)
                final_parts = []
                buffer = [] # 待合并序列

                def flush_buffer():
                    nonlocal buffer
                    if not buffer: return
                    
                    i = 0
                    while i < len(buffer):
                        # 贪婪搜索最长合并链
                        best_j = i
                        for j in range(i + 1, len(buffer)):
                            # 1. 检查 ID 连续性 (T1, T2, T3...)
                            if buffer[j]['id'] != buffer[j-1]['id'] + 1:
                                break 
                            
                            # 2. 检查夹心层 (Middle Layer must be Perfect)
                            # 如果 j = i+1，中间层为空，自动满足
                            is_sandwich_valid = True
                            if j > i + 1:
                                for k in range(i + 1, j):
                                    if buffer[k]['status'] != 'perfect':
                                        is_sandwich_valid = False
                                        break
                            
                            if not is_sandwich_valid:
                                # 注意：三明治法则断了，但我们不能 break 外层循环，
                                # 因为可能 T1(bad)-T2(bad) 没法合，但 T2(bad)-T3(ok) 也没法合...
                                # 这里的逻辑是：必须找到从 i 开始能连到的最远 j
                                # 简单的做法：只要中间坏了一个，就不能连到更远了
                                break 
                            
                            best_j = j
                        
                        # 输出
                        if best_j > i:
                            final_parts.append(f"{buffer[i]['str']}-{buffer[best_j]['str']}")
                            i = best_j + 1
                        else:
                            final_parts.append(buffer[i]['str'])
                            i += 1
                    buffer = []

                for node in trial_nodes:
                    if node['type'] == 'nan':
                        flush_buffer() # 异类隔离
                        final_parts.append(node['str'])
                    else:
                        # 是 Trial
                        if buffer:
                            # 如果 ID 不连续，必须先结算
                            if node['id'] != buffer[-1]['id'] + 1:
                                flush_buffer()
                        buffer.append(node)
                
                flush_buffer() # 结算最后剩余
                print(f"Trial: {', '.join(final_parts)} \n")

            else:
                print("Trial: (TrialID column missing) \n")

            logging.info("--- END OF TIMING INCONSISTENCY REPORT ---\n")
        else:
            logging.info("  -> Anchor timing consistency check passed.")

    def _resolve_ephys_data(self, times, indices, rate):
        """
        一个私有辅助函数，用于处理和验证电生理数据三元组 (时间, 索引, 采样率)。
        根据任意两个已知量，推算出第三个未知量。
        如果三者都提供，则检查其一致性。
        如果提供少于两个，则抛出错误。
        """
        # 标准化输入，将None转换为空数组以便统一处理
        times = np.asarray(times) if times is not None else np.array([])
        indices = np.asarray(indices) if indices is not None else np.array([])

        # --- 情况分析 ---
        times_provided = times.size > 0
        indices_provided = indices.size > 0
        rate_provided = rate is not None and rate > 0

        # 情况1：提供了三者中的两者，推算第三者
        if times_provided and indices_provided and not rate_provided:
            logging.info("Calculating sampling_rate from ephys_times and ephys_indices.")
            if len(times) < 2:
                raise ValueError("Cannot calculate sampling_rate with less than 2 data points.")
            # 使用差分的中位数来计算，更稳健，能抵抗个别点的噪声
            rate = np.nanmedian(np.diff(indices) / np.diff(times))
            logging.info(f"  -> Calculated sampling_rate: {rate:.2f} Hz")
        
        elif times_provided and rate_provided and not indices_provided:
            logging.info("Calculating ephys_indices from ephys_times and sampling_rate.")
            indices = np.round(times * rate).astype(np.int64)

        elif indices_provided and rate_provided and not times_provided:
            logging.info("Calculating ephys_times from ephys_indices and sampling_rate.")
            times = indices / rate
            
        # 情况2：提供了所有三者，进行一致性检查
        elif times_provided and indices_provided and rate_provided:
            logging.info("All three ephys data components provided. Checking for consistency.")
            if len(times) >= 2: # 只有超过2个点才能计算间隔和实际采样率
                # 1. 基于 times 和 indices 计算实际采样率
                calculated_rate = np.nanmedian(np.diff(indices) / np.diff(times))
                # 2. 比较计算出的采样率和用户提供的采样率 (例如，相对误差超过 1e-6 Hz就认为不一致)
                if not np.isclose(rate, calculated_rate, atol=1e-12):
                    # 3. 如果不一致，打印警告信息，包含两个值
                    logging.warning(
                        "Consistency Warning: The provided sampling_rate seems inconsistent with ephys_times and ephys_indices.\n"
                        "  -> Provided rate: %.2f Hz\n"
                        "  -> Calculated rate from data: %.2f Hz\n"
                        "  -> Proceeding with the user-provided sampling_rate and other data as requested.",
                        rate, calculated_rate
                    )
        
        # 情况3：提供的信息不足
        else:
            raise ValueError("Insufficient ephys data. At least two of [ephys_times, ephys_indices, sampling_rate] must be provided.")

        # 最终长度校验
        if times.size != indices.size:
             raise ValueError(f"Ephys times and indices must have the same length. Got {times.size} and {indices.size}.")

        return times, indices, rate

    def add_sync_context(self, context_name: str, ephys_times=None, ephys_indices=None, sampling_rate=30000.0, match_against: str = 'EventTime', sync_within_trial=True):
        """
        为某个Ephys控制器（如Plexon）添加同步上下文。
        如果主时间轴未构建，则使用此上下文作为主参照系来构建它。
        """
        # 推算并检验结果
        try:
            # --- 调用辅助函数，获取干净、完整的数据 ---
            ephys_times, ephys_indices, sampling_rate = self._resolve_ephys_data(
                times=ephys_times, 
                indices=ephys_indices, 
                rate=sampling_rate
            )
        except ValueError as e:
            logging.error(f"Failed to resolve ephys data for context '{context_name}': {e}")
            return self # 出错则直接返回，不继续执行
        
        if not self._is_timeline_built:
            logging.info(f"Master timeline not built. Using context '{context_name}' as the master reference.")
            self._solve_and_build_timeline_once(ephys_times)
        
        logging.info(f"--- Adding synchronization context: '{context_name}' (SR: {sampling_rate} Hz) ---")
        
        # 【重构】使用'is_anchor'列，逻辑更清晰健壮
        anchors_on_timeline = self.master_timeline_df[self.master_timeline_df['is_anchor']].copy()
        
        if match_against not in anchors_on_timeline.columns:
            raise ValueError(f"The specified `match_against` column '{match_against}' does not exist.")
        
        logging.info(f"Matching against '{match_against}' timeline.")
        template_times = anchors_on_timeline[match_against].values
        
        pairings = get_pair_via_dtw(template_times, ephys_times)
        mapping = get_paired_ephys_event_index(pairings)
        
        if len(mapping) != len(template_times):
            logging.warning(f"DTW mapping length mismatch ({len(mapping)} vs {len(template_times)}). Using conservative method for '{context_name}'.")
            mapping = get_paired_ephys_event_index(pairings, conservative=True)
            # return self
        
        time_col = f'EphysTime_{context_name}'
        indice_col = f'EphysIndice_{context_name}'
        
        # 直接为匹配上的锚点赋值 (Ground Truth)
        anchor_indices_in_master = anchors_on_timeline.index
        valid_mask = ~np.isnan(mapping)
        mapped_ephys_indices = mapping[valid_mask].astype(int)

        self.master_timeline_df.loc[anchor_indices_in_master[valid_mask], time_col] = ephys_times[mapped_ephys_indices]
        self.master_timeline_df.loc[anchor_indices_in_master[valid_mask], indice_col] = ephys_indices[mapped_ephys_indices]
        
        # 检查匹配情况
        self._check_match_error(context_name)

        # 为所有非锚点事件进行刚性平移
        self._apply_rigid_translation(context_name, sampling_rate, sync_within_trial)
        
        logging.info(f"--- Context '{context_name}' added successfully. ---")
        return self

    def get_final_dataframe(self):
        """返回最终对齐后的主时间轴DataFrame的一个副本。"""
        return self.master_timeline_df.copy()

