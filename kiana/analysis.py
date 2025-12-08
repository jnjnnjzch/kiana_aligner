import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
import logging

class SpikeTrainAnalyzer:
    """
    An advanced, modular analyzer for neuronal spike trains.
    """
    def __init__(self, spike_train, event_windows, alignment_times=None, extra_events=None, font_prop=None, **kwargs):
        self.spike_train = np.asarray(spike_train)
        self.event_windows = np.asarray(event_windows)
        self.num_trials = len(self.event_windows)
        if alignment_times is None:
            self.align_points = self.event_windows[:, 0]
        else:
            self.align_points = np.asarray(alignment_times)
            if len(self.align_points) != self.num_trials:
                raise ValueError("Length of `alignment_times` must match the number of rows in `event_windows`.")
        self.default_events = extra_events
        self.font_prop = font_prop
        self.plot_params = {
            'individual_trial_color': 'gray', 'individual_trial_alpha': 0.3,
            'mean_trace_color': 'blue', 'sem_shade_color': 'lightblue',
            'baseline_color': 'red', 'align_line_color': 'black',
        }
        self.plot_params.update(kwargs)
        self._precompute_relative_spikes()
        self._precompute_relative_events()
        self.time_vector, self.rates_matrix, self.calculation_mode, self.bin_size = None, None, None, None
        vivid_colors = [
            '#FF1A1A',  # 极鲜红 (Ultra Vivid Red)
            '#FF80B3',  # 亮粉红 (Bright Pink)
            '#E6194B',  # 树莓红 (Raspberry Red)
            '#FFA500',  # 鲜橙色 (Vivid Orange)
            '#FFD700',  # 金黄色 (Gold)
            '#B08000',  # 暗金色/赭石色 (Ochre)
            '#32CD32',  # 酸橙绿 (Lime Green)
            '#ADFF2F',  # 绿黄色 (Green-Yellow)
            '#008000',  # 标准绿 (Standard Green)
            '#00FFFF',  # 青色/水色 (Aqua)
            '#4363d8',  # 矢车菊蓝 (Cornflower Blue)
            '#00008B',  # 深蓝 (Dark Blue) - 保留深蓝，因为它与黑色仍有明显区别
            '#9400D3',  # 暗紫罗兰 (Dark Violet)
            '#FF00FF',  # 品红/紫红 (Fuchsia)
            '#E6BEFF',  # 薰衣草紫 (Lavender)
            '#40E0D0',  # 绿松石 (Turquoise)
            '#FA8072',  # 鲑鱼色/珊瑚粉 (Salmon)
        ]
        self.event_color_map = {}
        if self.default_events:
            # 获取matplotlib的默认颜色列表
            prop_cycler = plt.rcParams['axes.prop_cycle']
            colors = prop_cycler.by_key()['color']
            # colors = plt.cm.get_cmap('Paired').colors
            # colors = vivid_colors
            # 将每个事件名和一个颜色绑定
            for i, event_name in enumerate(self.default_events.keys()):
                self.event_color_map[event_name] = colors[i % len(colors)]

    @property
    def rates(self):
        if self.rates_matrix is None: print("Warning: Rates have not been calculated yet.")
        return self.rates_matrix
    @property
    def time_axis(self):
        if self.time_vector is None: print("Warning: Time axis has not been defined yet.")
        return self.time_vector
    @property
    def aligned_spike_train(self):
        """
        Returns a list of spike trains, where each entry corresponds to a trial.
        Spike times are relative to the trial's alignment point.
        """
        return self.relative_spikes

    def _precompute_relative_spikes(self):
        self.relative_spikes = []
        for i in range(self.num_trials):
            win_spikes = self.spike_train[(self.spike_train >= self.event_windows[i, 0]) & (self.spike_train < self.event_windows[i, 1])]
            self.relative_spikes.append(win_spikes - self.align_points[i])
    
    def _precompute_relative_events(self):
        self.relative_events = {}
        for event_name, event_times in self.default_events.items():
            relative_event_times = []
            for i in range(self.num_trials):
                trial_start = self.event_windows[i, 0]
                trial_end = self.event_windows[i, 1]
                trial_event_times = event_times[(event_times >= trial_start) & (event_times < trial_end)]
                relative_event_times.append(trial_event_times - self.align_points[i])
            self.relative_events[event_name] = relative_event_times

    def _determine_time_window(self, analysis_window=None):
        if analysis_window is None:
            all_spikes_flat = np.concatenate(self.relative_spikes)
            if all_spikes_flat.size == 0:
                print("Warning: No spikes found. Using default plot range [-1, 1].")
                return -1.0, 1.0
            min_time, max_time = all_spikes_flat.min(), all_spikes_flat.max()
            padding = (max_time - min_time) * 0.05
            return min_time - padding, max_time + padding
        else:
            return analysis_window[0], analysis_window[1]

    def calculate_rates_event_window(self, event_series, mode='gaussian', **kwargs):
        # 这个calculate rates是计算event_window范围内的
        self.calculation_mode = mode
        if mode == 'gaussian':
            std = kwargs.get('gaussian_std', 0.02); 
            time_bin_size = kwargs.get('high_res_bin', 0.001)
            post_processor = lambda rate_arr: gaussian_filter1d(rate_arr, sigma=std / time_bin_size); 
            self.bin_size = None
        elif mode == 'binned':
            time_bin_size = kwargs.get('bin_size', 0.1)
            post_processor = lambda rate_arr: rate_arr; 
            self.bin_size = time_bin_size
        else: raise ValueError(f"Mode '{mode}' not recognized. Use 'gaussian' or 'binned'.")
        all_trial_rates = []
        for rel_spikes, event_windows in zip(event_series, self.event_windows):
            min_t = 0
            max_t = event_windows[1] - event_windows[0]
            self.time_vector = np.arange(min_t, max_t, time_bin_size)
            histogram_bins = np.append(self.time_vector, self.time_vector[-1] + time_bin_size)
            counts, _ = np.histogram(rel_spikes, bins=histogram_bins)
            initial_rate = counts / time_bin_size
            all_trial_rates.append(post_processor(initial_rate))
        # print(f"Rates calculated via '{mode}' mode.")
        return all_trial_rates

    def calculate_stimulus_vector(self, stimulus, bin_size):
        stimulus_vector = []
        def _run_for_one_dim():
            for sti_time, event_window in zip(stimulus, self.event_windows):
                num_bins = int(np.ceil((event_window[1] - event_window[0]) / bin_size))
                stim_vector = np.zeros(num_bins)
                if np.any(np.isnan(sti_time)):
                    stimulus_vector.append(stim_vector)
                else:
                    stim_rel = sti_time - event_window[0]
                    stim_bin = int(np.floor(stim_rel / bin_size))
                    stim_vector[stim_bin] = 1
                    stimulus_vector.append(stim_vector)
            return stimulus_vector
        def _run_for_two_dim():
            for sti_time, event_window in zip(stimulus, self.event_windows):
                num_bins = int(np.ceil((event_window[1] - event_window[0]) / bin_size))
                stim_vector = np.zeros(num_bins)
                if np.any(np.isnan(sti_time)):
                    stimulus_vector.append(stim_vector)
                else:
                    stim_start_rel = sti_time[0] - event_window[0]
                    stim_end_rel = sti_time[1] - event_window[0]
                    start_bin = int(np.floor(stim_start_rel / bin_size))
                    end_bin = int(np.ceil(stim_end_rel / bin_size))
                    stim_vector[start_bin:end_bin] = 1
                    stimulus_vector.append(stim_vector)
            return stimulus_vector
        if stimulus.shape[0] != self.num_trials:
            raise ValueError("Length of `stimulus` must match the number of trials.")
        if len(stimulus.shape) == 1:
            return _run_for_one_dim()
        elif len(stimulus.shape) == 2:
            if stimulus.shape[1] == 2:
                return _run_for_two_dim()
            elif stimulus.shape[1] == 1:
                return _run_for_one_dim()
            else:
                raise ValueError("When `stimulus` is 2D, its second dimension must be 1 or 2.")
        else:
            raise ValueError("`stimulus` must be either 1D or 2D array-like.")

    def calculate_rates(self, mode='gaussian', analysis_window=None, **kwargs):
        self.calculation_mode = mode
        if mode == 'gaussian':
            std = kwargs.get('gaussian_std', 0.02); 
            time_bin_size = kwargs.get('high_res_bin', 0.001)
            post_processor = lambda rate_arr: gaussian_filter1d(rate_arr, sigma=std / time_bin_size); 
            self.bin_size = None
        elif mode == 'binned':
            time_bin_size = kwargs.get('bin_size', 0.1)
            post_processor = lambda rate_arr: rate_arr; 
            self.bin_size = time_bin_size
        else: raise ValueError(f"Mode '{mode}' not recognized. Use 'gaussian' or 'binned'.")
        min_t, max_t = self._determine_time_window(analysis_window)
        self.time_vector = np.arange(min_t, max_t, time_bin_size)
        all_trial_rates = []
        histogram_bins = np.append(self.time_vector, self.time_vector[-1] + time_bin_size)
        for rel_spikes in self.relative_spikes:
            counts, _ = np.histogram(rel_spikes, bins=histogram_bins)
            initial_rate = counts / time_bin_size
            all_trial_rates.append(post_processor(initial_rate))
        self.rates_matrix = np.array(all_trial_rates)
        print(f"Rates calculated via '{mode}' mode.")
        return self

    # --- Private Generic Helpers ---
    def _setup_ax(self, ax=None):
        if ax is None: return   
        return ax.get_figure(), ax
    def _calculate_baseline_rate(self, baseline_window):
        if baseline_window is None: return None
        total_spikes, total_duration = 0, 0
        for i in range(self.num_trials):
            abs_start, abs_end = self.align_points[i] + baseline_window[0], self.align_points[i] + baseline_window[1]
            total_spikes += np.sum((self.spike_train >= abs_start) & (self.spike_train < abs_end))
            total_duration += (abs_end - abs_start)
        return total_spikes / total_duration if total_duration > 0 else 0
    
    def _get_relative_events(self, extra_events):
        """
        Calculates relative event times by associating events to trials based on
        whether they fall within the trial's specific event_window.
        This is the corrected logic based on user feedback.
        """
        if extra_events is None:
            return
        # 提取所有窗口的开始和结束时间，方便调用
        window_starts = self.event_windows[:, 0]
        window_ends = self.event_windows[:, 1]
        # 遍历每一种事件类型 (例如: '注视点出现')
        for event_name, absolute_times in extra_events.items():
            absolute_times = np.asarray(absolute_times)
            if absolute_times.size == 0:
                continue
            events_col = absolute_times[:, np.newaxis] # (N_events, 1)
            starts_row = window_starts[np.newaxis, :] # (1, N_trials)
            ends_row = window_ends[np.newaxis, :] # (1, N_trials)

            hit_matrix = (events_col >= starts_row) & (events_col < ends_row) # hit_matrix[j, i] 为 True 代表事件j落在了窗口i中
            
            #    event_indices 是事件的索引 (行)
            #    trial_indices 是trial的索引 (列)
            event_indices, trial_indices = np.where(hit_matrix)
            
            # 如果没有任何匹配，则跳到下一种事件类型
            if event_indices.size == 0:
                continue

            # 4. 一步计算所有相对时间
            #    根据索引，直接选取对应的事件时间和对齐时间
            matched_event_times = absolute_times[event_indices]
            matched_align_points = self.align_points[trial_indices]
            
            relative_times = matched_event_times - matched_align_points
            
            # 5. Yield 最终结果
            yield event_name, relative_times, trial_indices
    def _unify_legends(self, axes, target_ax, legend_out=True):
        handles, labels = [], []
        for ax in axes:
            if ax:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h); labels.extend(l)
        if handles:
            by_label = dict(zip(labels, handles))
            if legend_out:
                target_ax.legend(by_label.values(), by_label.keys(), prop=self.font_prop, loc='upper left', bbox_to_anchor=(1.02, 1))
            else:
                target_ax.legend(by_label.values(), by_label.keys(), prop=self.font_prop, loc='upper right')

    # --- Private PSTH Plotting Helpers ---
    def _draw_rate_traces(self, ax, plot_x, drawstyle, params, show_individual=True, trial_labels=None):
        """Helper to draw rate traces, now with support for grouped plotting."""
        # 1. 绘制所有独立的trial轨迹 (作为背景)
        if show_individual:
            for rate_trace in self.rates_matrix:
                ax.plot(plot_x, rate_trace, color=params['individual_trial_color'], alpha=params['individual_trial_alpha'], lw=1.5, drawstyle=drawstyle)

        # 2. 检查窗口时长是否一致，若不一致则不绘制均值和SEM
        if len(np.unique(np.round(self.event_windows[:, 1] - self.event_windows[:, 0], decimals=6))) > 1:
            print("Info: Window durations are not consistent. Mean and SEM are not plotted.")
            return
            
        # 3. 根据是否提供 trial_labels，决定绘制总平均还是分组平均
        if trial_labels is None:
            # --- 原始行为: 绘制总平均 ---
            mean_rate = np.mean(self.rates_matrix, axis=0)
            sem_rate = sem(self.rates_matrix, axis=0)
            ax.plot(plot_x, mean_rate, color=params['mean_trace_color'], lw=2.5, label='Mean Rate', drawstyle=drawstyle)
            if drawstyle == 'default':
                ax.fill_between(plot_x, mean_rate - sem_rate, mean_rate + sem_rate, color=params['sem_shade_color'], alpha=0.5, label='SEM', zorder=-1)
        else:
            # --- 新功能: 按条件分组绘制 ---
            unique_labels = sorted(list(np.unique(trial_labels)))
            prop_cycler = plt.rcParams['axes.prop_cycle']
            colors = prop_cycler.by_key()['color']
            
            for i, label in enumerate(unique_labels):
                # 找到当前条件对应的所有trials
                indices = np.where(np.array(trial_labels) == label)[0]
                if len(indices) == 0: continue
                
                # 为当前条件计算均值和SEM
                group_rates = self.rates_matrix[indices]
                mean_rate = np.mean(group_rates, axis=0)
                sem_rate = sem(group_rates, axis=0)
                
                # 获取颜色并绘图
                color = colors[i % len(colors)]
                ax.plot(plot_x, mean_rate, color=color, lw=2.5, label=f'Mean ({label})', drawstyle=drawstyle)
                if drawstyle == 'default':
                    ax.fill_between(plot_x, mean_rate - sem_rate, mean_rate + sem_rate, color=color, alpha=0.3, zorder=-1)

    def _draw_auxiliary_lines(self, ax, baseline_window, params):
        baseline_rate = self._calculate_baseline_rate(baseline_window)
        if baseline_rate is not None: ax.axhline(baseline_rate, ls='--', color=params['baseline_color'], label=f'Baseline ({baseline_rate:.2f} Hz)')
        ax.axvline(0, ls='--', color=params['align_line_color'], label='Alignment (t=0)')
    def _draw_extra_events(self, ax, extra_events, style='rug'):
        if not hasattr(self, 'event_color_map'):
            self.event_color_map = {}
    
        for i, (event_name, relative_times, trial_indices) in enumerate(self._get_relative_events(extra_events)):
            color = self.event_color_map[event_name]
            if style == 'rug':
                ax.plot(relative_times, np.full_like(relative_times, 0.02), transform=ax.get_xaxis_transform(), marker='|', markersize=12, markeredgewidth=2, linestyle='none', color=color, label=event_name, clip_on=True)
            elif style == 'raster_tick':
                event_times_by_trial = [[t] for t in relative_times]
                ax.eventplot(event_times_by_trial, colors=color, lineoffsets=trial_indices, linelengths=0.8, linewidths=2.0, label=event_name, zorder=3)

    def _finalize_psth_plot(self, ax, plot_x, style):
        title = getattr(ax, 'get_title', lambda: '')() # Get existing title before overwriting
        ax.set_title(title, fontsize=16, fontproperties=self.font_prop)
        ax.set_xlabel("Time from Alignment (s)", fontproperties=self.font_prop)
        ax.set_ylabel("Firing Rate (spikes/s)", fontproperties=self.font_prop)
        ax.grid(True, linestyle=':', alpha=0.6)
        if ax.get_ylim()[0] >= 0 and any(line.get_marker() == '|' for line in ax.get_lines()):
             ax.set_ylim(bottom=-0.05 * ax.get_ylim()[1])
        ax.set_xlim(plot_x[0], plot_x[-1])
        ax.legend(prop=self.font_prop)

    # --- Private Raster Plotting Helpers ---
    def _setup_raster_canvas(self, show_psth, fig):
        ax_psth = None
        if fig is None:
            if show_psth: fig, (ax_raster, ax_psth) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            else: fig, ax_raster = plt.subplots(figsize=(12, 6))
        else:
            fig.clear()
            if show_psth:
                gs = fig.add_gridspec(2, 1, height_ratios=[3, 1]); ax_raster = fig.add_subplot(gs[0]); ax_psth = fig.add_subplot(gs[1], sharex=ax_raster)
            else: ax_raster = fig.add_subplot(1, 1, 1)
        return fig, ax_raster, ax_psth
    def _draw_raster_spikes(self, ax, **raster_kwargs):
        spike_color = raster_kwargs.get('spike_color', 'black'); 
        spike_lw = raster_kwargs.get('spike_lw', 1.0)
        ax.eventplot(self.relative_spikes, colors=spike_color, lineoffsets=np.arange(self.num_trials), linelengths=0.8, linewidths=spike_lw)
        # ax.eventplot(self.relative_spikes, lineoffsets=np.arange(self.num_trials), linelengths=0.05, linewidths=spike_lw)
    def _set_raster_yticklabels(self, ax, trial_labels):
        ax.set_ylim(self.num_trials - 0.5, -0.5)
        if not trial_labels:
            ax.set_ylabel("Trial ID", fontproperties=self.font_prop)
            return
        if len(trial_labels) != self.num_trials:
            print("Warning: 'trial_labels' length mismatch. Falling back to default trial IDs.")
            ax.set_ylabel("Trial ID", fontproperties=self.font_prop)
            return
        ax.set_ylabel("Trial Condition", fontproperties=self.font_prop)
        new_ticks, new_labels = [], []
        start_index = 0
        for i in range(1, self.num_trials + 1):
            if i == self.num_trials or trial_labels[i] != trial_labels[start_index]:
                end_index = i - 1
                tick_pos = (start_index + end_index) / 2.0
                label_text = trial_labels[start_index]
                new_ticks.append(tick_pos)
                new_labels.append(label_text)
                if i < self.num_trials:
                    ax.axhline(y=i - 0.5, color='cyan', linestyle='-', linewidth=0.75, alpha=0.75, zorder=5)
                start_index = i
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(new_labels, fontproperties=self.font_prop)

    # --- Public Plotting API ---
    # MODIFIED: Added `trial_labels` parameter for grouped plotting
    def plot_psth(self, style='line', ax=None, show_individual=True, trial_labels=None, **kwargs):
        """
        Plots the PSTH, with an option to group by condition.

        Args:
            style (str): 'line' or 'histogram'.
            ax (Axes): Matplotlib axes to plot on.
            show_individual (bool): Whether to show individual trial traces.
            trial_labels (list or array, optional): A list of labels corresponding to each trial.
                If provided, PSTH will be grouped, and mean traces will be plotted for each unique label.
                Defaults to None, which plots a single mean for all trials.
            **kwargs: Additional plotting parameters.
        """
        if self.rates_matrix is None: raise RuntimeError("Please call calculate_rates() before plotting.")
        if style == 'histogram' and self.calculation_mode != 'binned': raise TypeError("Histogram style is only for 'binned' mode.")
        
        # --- NEW: Validate trial_labels ---
        if trial_labels is not None and len(trial_labels) != self.num_trials:
            raise ValueError(f"Length of `trial_labels` ({len(trial_labels)}) must match the number of trials ({self.num_trials}).")
        
        fig, ax = self._setup_ax(ax)
        current_plot_params = self.plot_params.copy(); current_plot_params.update(kwargs)
        title = current_plot_params.pop('title', f"PSTH ({self.calculation_mode.capitalize()} / {style.capitalize()})")
        baseline_window = current_plot_params.pop('baseline_window', None)
        extra_events = current_plot_params.pop('extra_events', self.default_events)
        drawstyle = 'steps-post' if style == 'histogram' else 'default'
        
        # Adjust x-axis for binned line plots to center points on bins
        if self.calculation_mode == 'binned' and style != 'histogram':
            plot_x = self.time_vector + self.bin_size / 2
        else:
            plot_x = self.time_vector
            
        # --- MODIFIED: Pass trial_labels to the drawing helper ---
        self._draw_rate_traces(ax, plot_x, drawstyle, current_plot_params, show_individual=show_individual, trial_labels=trial_labels)
        
        self._draw_auxiliary_lines(ax, baseline_window, current_plot_params)
        self._draw_extra_events(ax, extra_events, style='rug')
        self._finalize_psth_plot(ax, plot_x, style)
        ax.set_title(title, fontproperties=self.font_prop)
        return ax

    def plot_raster(self, fig=None, **kwargs):
        # MODIFIED: Added 'trial_labels' to the list of psth kwargs to ensure it's passed down.
        # Note: The original implementation `k in self.plot_psth.__code__.co_varnames` is clever and
        # will automatically handle this new parameter without explicit changes here.

        trial_labels = kwargs.pop('trial_labels', None)  # Pop from kwargs
        legend_out = kwargs.pop('legend_out', True)  # Pop legend_out to control legend visibility
        psth_kwargs = {k: v for k, v in kwargs.items() if k in self.plot_psth.__code__.co_varnames or k in ['baseline_window', 'extra_events', 'title', 'mean_trace_color']}
        raster_kwargs = {k: v for k, v in kwargs.items() if k not in psth_kwargs}
        # trial_labels = raster_kwargs.pop('trial_labels', None) # Pop from raster_kwargs
        # if trial_labels is not None:
        #      psth_kwargs['trial_labels'] = trial_labels # But ensure it's in psth_kwargs for the PSTH plot
        
        show_psth = raster_kwargs.pop('show_psth', True)
        suptitle = raster_kwargs.pop('suptitle', "Raster Plot & PSTH")

        if show_psth and self.rates_matrix is None:
            print("Info: Rates not pre-calculated for PSTH. Running `calculate_rates()` with default parameters.")
            self.calculate_rates()

        fig, ax_raster, ax_psth = self._setup_raster_canvas(show_psth, fig)
        
        self._draw_raster_spikes(ax_raster, **raster_kwargs)
        self._set_raster_yticklabels(ax_raster, trial_labels)
        
        events_to_plot = psth_kwargs.get('extra_events', self.default_events)
        self._draw_extra_events(ax_raster, events_to_plot, style='raster_tick')

        if show_psth:
            # The `trial_labels` will be passed correctly via psth_kwargs
            self.plot_psth(ax=ax_psth, show_individual=False, trial_labels=trial_labels, **psth_kwargs)
            ax_psth.set_title(""); 
            ax_psth.set_xlabel("")
            ax_psth.legend().set_visible(False)
        
        fig.suptitle(suptitle, fontsize=16, fontproperties=self.font_prop)
        self._unify_legends([ax_raster, ax_psth] if ax_psth else [ax_raster], ax_raster, legend_out)
        if show_psth: ax_psth.set_xlabel("Time from Alignment (s)", fontproperties=self.font_prop)
        else: ax_raster.set_xlabel("Time from Alignment (s)", fontproperties=self.font_prop)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        return fig, (ax_raster, ax_psth) if show_psth else ax_raster
