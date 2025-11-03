# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:41:41 2024
Fixed and cleaned: file finding, blink logic, robust handling of missing streams,
collecting et-params safely, and safer merges.
@author: schakraborth (fixed)
"""
import os
import re
import copy
import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tsfel
import cateyes

thisdir = os.getcwd()
filenames = []
# find all .xdf files under current working dir
for r, d, f in os.walk(thisdir):
    for file in f:
        if file.lower().endswith(".xdf"):
            filenames.append(os.path.join(r, file))

def find_file_with_id(file_list, target_id):
    """Find a filename where 'sub-<id>_' appears (case-insensitive).
       Returns full path or None if not found."""
    for filename in file_list:
        base = os.path.basename(filename)
        match = re.search(r'sub-([A-Za-z0-9]+)_', base, flags=re.IGNORECASE)
        if match and target_id == match.group(1):
            return filename
    return None

def interpolate_times(start_time, end_time, num_samples):
    """
    Generate interpolated datetimes between start_time and end_time (inclusive).
    start_time, end_time: pandas.Timestamp / datetime
    num_samples: int >= 2
    Returns list of pandas.Timestamp
    """
    if num_samples < 2:
        raise ValueError("Number of samples must be at least 2.")
    # Use pandas date_range for robust interpolation
    return pd.date_range(start=start_time, end=end_time, periods=num_samples).to_pydatetime().tolist()

def blink(arr, t, minlen, maxlen):
    """
    Detect consecutive '1' runs in arr (1 = missing). Returns:
    (count, average_duration_samples, max_duration_samples)
    t is expected to be an array of timestamps (can be None) -- durations are returned in samples.
    """
    zero_durations = []
    count = 0
    current_length = 0
    n = len(arr)
    for val in arr:
        if val == 1:
            current_length += 1
        else:
            if minlen <= current_length <= maxlen:
                zero_durations.append(current_length)
                count += 1
            current_length = 0
    # final run
    if minlen <= current_length <= maxlen:
        zero_durations.append(current_length)
        count += 1

    if zero_durations:
        average_duration = sum(zero_durations) / len(zero_durations)
        max_duration = max(zero_durations)
    else:
        average_duration = 0
        max_duration = 0

    return count, average_duration, max_duration

def blink_detection(x, y, time, missing_val=0.0, minlen_samples=None, maxlen_samples=None, fs=60):
    """
    Detect blinks based on NaNs in both x and y (both NaN => blink).
    Returns (count, average_duration_samples, max_duration_samples).
    minlen_samples and maxlen_samples default to 100ms and 400ms at sampling rate fs.
    """
    # determine thresholds in samples if not provided
    if minlen_samples is None:
        minlen_samples = int(fs * 0.1)  # 100 ms
    if maxlen_samples is None:
        maxlen_samples = int(fs * 0.4)  # 400 ms

    # construct missing mask: 1 if both are NaN, else 0
    mx = np.isnan(x).astype(int)
    my = np.isnan(y).astype(int)
    miss = ((mx + my) == 2).astype(int)  # 1 => missing
    count, avg_dur, max_dur = blink(miss, time, minlen_samples, maxlen_samples)
    return count, avg_dur, max_dur

def get_pupil_features(df):
    """
    Extract some pupil features using tsfel functions (kept as in original).
    Expects columns: left_pupil_diam, right_pupil_diam, left_on_display_y, right_on_display_y
    """
    fs = 60
    # left
    left_df = df[df['left_on_display_y'].notna()]
    L_slope = tsfel.slope(left_df['left_pupil_diam'].to_numpy()) if not left_df.empty else np.nan
    L_max = tsfel.calc_max(left_df['left_pupil_diam'].to_numpy()) if not left_df.empty else np.nan
    L_mean = tsfel.calc_mean(left_df['left_pupil_diam'].to_numpy()) if not left_df.empty else np.nan
    L_entropy = tsfel.spectral_entropy(left_df['left_pupil_diam'].to_numpy(), fs) if not left_df.empty else np.nan
    L_variation = tsfel.spectral_variation(left_df['left_pupil_diam'].to_numpy(), fs) if not left_df.empty else np.nan
    # right
    right_df = df[df['right_on_display_y'].notna()]
    R_slope = tsfel.slope(right_df['right_pupil_diam'].to_numpy()) if not right_df.empty else np.nan
    R_max = tsfel.calc_max(right_df['right_pupil_diam'].to_numpy()) if not right_df.empty else np.nan
    R_mean = tsfel.calc_mean(right_df['right_pupil_diam'].to_numpy()) if not right_df.empty else np.nan
    R_entropy = tsfel.spectral_entropy(right_df['right_pupil_diam'].to_numpy(), fs) if not right_df.empty else np.nan
    R_variation = tsfel.spectral_variation(right_df['right_pupil_diam'].to_numpy(), fs) if not right_df.empty else np.nan

    return L_slope, L_max, L_mean, L_entropy, L_variation, R_slope, R_max, R_mean, R_entropy, R_variation

def get_gaze_entropy(df, w=1, h=1):
    """
    Compute a spatial entropy across fixation coordinates (left_on_display_x, left_on_display_y).
    Returns normalized entropy value; uses a fixed state space s and bin step sby (kept as original).
    """
    df_fix = df[df['Class'].str.contains('Fixation', na=False)]
    df_fix = df_fix[df_fix['left_on_display_y'].notna()]
    if df_fix.empty:
        return np.nan

    appended_data = []
    fix_id = df_fix['Segment'].unique()
    for fix in fix_id:
        dd = df_fix.loc[df_fix['Segment'] == fix]
        appended_data.append(dd[['left_on_display_x', 'left_on_display_y']])
    appended_data = pd.concat(appended_data, ignore_index=True)

    # parameters (as in your original code)
    s = 1
    sby = 0.10
    xydf = appended_data.rename({"left_on_display_x": "x", "left_on_display_y": "y"}, axis='columns')
    # clip values into [0, s) to avoid NaN bins (or let pd.cut place NaNs)
    xydf['x_range'] = pd.cut(xydf.x, np.arange(0, s + 1e-9, sby), right=False)
    xydf['y_range'] = pd.cut(xydf.y, np.arange(0, s + 1e-9, sby), right=False)
    xydf = xydf.groupby(['x_range', 'y_range']).size().reset_index().rename(columns={0: 'count'})
    xydf['p'] = xydf['count'] / xydf['count'].sum()

    # compute p * log2(p) safely
    pvals = xydf['p'].to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        p_log = np.where(pvals > 0, pvals * np.log2(pvals), 0.0)
    return abs(p_log.sum())

def fixation_saccade_detection(data):
    """Return summary stats for saccades and fixations based on Class & Segment grouping."""
    data = data[data['left_on_display_y'].notna()]
    if data.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    counts = data.groupby(['Class', 'Segment']).size()
    # Saccades
    try:
        saccades_count = counts.loc['Saccade']
        average_saccadic_duration = saccades_count.mean()
        maximum_saccadic_duration = saccades_count.max()
        saccadic_count = saccades_count.count()
    except Exception:
        average_saccadic_duration = np.nan
        maximum_saccadic_duration = np.nan
        saccadic_count = np.nan
    # Fixations
    try:
        fixation_count_series = counts.loc['Fixation']
        average_fixation_duration = fixation_count_series.mean()
        maximum_fixation_duration = fixation_count_series.max()
        fixation_count = fixation_count_series.count()
    except Exception:
        fixation_count = np.nan
        average_fixation_duration = np.nan
        maximum_fixation_duration = np.nan

    return average_saccadic_duration, maximum_saccadic_duration, saccadic_count, average_fixation_duration, maximum_fixation_duration, fixation_count

def get_et_params(data):
    """
    Run through gaze data and compute multiple ET-derived parameters.
    Returns a tuple of 20 items matching the original order you expected.
    """
    if data is None or data.empty:
        return tuple([np.nan] * 20)

    x = data['left_on_display_x'].to_numpy()
    y = data['left_on_display_y'].to_numpy()
    t = data['timestamps'].to_numpy() if 'timestamps' in data.columns else np.arange(len(x))

    blink_count, average_blink_duration, max_blink_duration = blink_detection(x, y, t)
    average_saccadic_duration, maximum_saccadic_duration, saccadic_count, average_fixation_duration, maximum_fixation_duration, fixation_count = fixation_saccade_detection(data)
    try:
        pupil_feats = get_pupil_features(data)
    except Exception:
        pupil_feats = tuple([np.nan] * 10)
    try:
        gaze_entropy = get_gaze_entropy(data)
    except Exception:
        gaze_entropy = np.nan

    return (blink_count, average_blink_duration, max_blink_duration,
            average_saccadic_duration, maximum_saccadic_duration, saccadic_count,
            average_fixation_duration, maximum_fixation_duration, fixation_count,
            gaze_entropy) + tuple(pupil_feats)

# ------------------------------------------------
# main processing: compute et params for all participants
# ------------------------------------------------
results_df = pd.read_csv('time_intervals_eye_tracking.csv')
if 'et_params' not in results_df.columns:
    results_df['et_params'] = np.nan

participantsList = results_df['Name'].unique()

# collect rows (each row corresponds to one event window for one participant)
et_params_list = []

for participant in participantsList:
    try:
        file = find_file_with_id(filenames, str(participant))
        if file is None:
            raise FileNotFoundError(f'.xdf for participant {participant} not found in search paths.')
        data, header = pyxdf.load_xdf(file)
        df_et = pd.DataFrame()
        tags = pd.DataFrame()  # ensure tags exists even if no event stream found

        # parse streams
        for stream in data:
            name = ''.join(stream['info']['name'])
            # Tobii data stream
            if name == 'Tobii Pro Nano':
                y = stream['time_series']
                t = stream['time_stamps']
                infos = stream['info']['desc'][0]['channels'][0]['channel']
                et_headers = []
                for info in infos:
                    header_label = ''.join(info['label'])
                    et_headers.append(header_label)
                df_et = pd.DataFrame(y, columns=et_headers)
                df_et.insert(0, "timestamps", t)
            # event streams; accept either of these names
            elif name in ('Tobii Pro Nano events', 'I-pad events', 'I-pad events '):
                y = stream.get('time_series', [])
                t = stream.get('time_stamps', [])
                if len(y) > 0:
                    tags = pd.DataFrame(y)
                    # try to name the single channel with a reasonable header if available
                    if tags.shape[1] == 1:
                        tags.columns = ['systemtime']
                    else:
                        # fallback - try to find a column named 'systemtime' or use first column
                        if 'systemtime' not in tags.columns:
                            tags.columns = [f'col{i}' for i in range(tags.shape[1])]
                            tags = tags.rename(columns={tags.columns[0]: 'systemtime'})
                    tags.insert(0, "timestamps", t)
        # if no tobii data found, raise
        if df_et.empty:
            raise ValueError(f"Tobii Pro Nano stream not found or empty in {file}")

        # merge tags/events if present
        if not tags.empty:
            df_et = pd.merge(df_et, tags, on='timestamps', how='outer')
        else:
            # ensure systemtime column exists (fill with NaT)
            df_et['systemtime'] = pd.NaT

        df_et = df_et.sort_values(by=['timestamps'], ignore_index=True)

        # try to coerce systemtime to datetime; allow NaT
        if 'systemtime' in df_et.columns:
            try:
                df_et['systemtime'] = pd.to_datetime(df_et['systemtime'], unit='ns', errors='coerce')
            except Exception:
                df_et['systemtime'] = pd.to_datetime(df_et['systemtime'], errors='coerce')

        # classify fixations/saccades
        # make sure left/right arrays exist (cateyes expects numeric arrays; NaNs are ok)
        segments, classes = cateyes.classify_uneye(
            df_et['left_on_display_x'].to_numpy(),
            df_et['left_on_display_y'].to_numpy(),
            np.arange(len(df_et['left_on_display_y'].to_numpy())),
            min_sacc_dur=2,
            min_sacc_dist=1,
            return_discrete=False,
            return_orig_output=False,
            weight_set='weights_synthetic'
        )
        df_et["Segment"] = segments
        df_et["Class"] = classes

        # Build events_df from results_df for this participant
        mask_participant = results_df["Name"] == participant
        events_data = {
            'event': results_df.loc[mask_participant, 'Task'].values,
            'starttime': pd.to_datetime(results_df.loc[mask_participant, 'StartTime'], unit='ns', errors='coerce'),
            'endtime': pd.to_datetime(results_df.loc[mask_participant, 'EndTime'], unit='ns', errors='coerce')
        }
        events_df = pd.DataFrame(events_data)
        # drop rows where start or end are NaT
        events_df = events_df.dropna(subset=['starttime', 'endtime'], how='any')
        if events_df.empty:
            # if no events, still compute a single run over whole recording
            start = df_et['systemtime'].min() if df_et['systemtime'].notna().any() else pd.Timestamp.now(tz='UTC')
            end = df_et['systemtime'].max() if df_et['systemtime'].notna().any() else start
            length = df_et.shape[0] if df_et.shape[0] >= 2 else 2
            interpolated_times = interpolate_times(start, end, length)
            df_et['systemtime'] = interpolated_times
            et_param = get_et_params(df_et)
            et_params_list.append(list(et_param))
        else:
            start = events_df['starttime'].min()
            end = events_df['endtime'].max()
            length = df_et.shape[0] if df_et.shape[0] >= 2 else 2
            # create evenly spaced timestamps across recording span
            interpolated_times = interpolate_times(start, end, length)
            df_et['systemtime'] = interpolated_times
            # mark event labels and compute params for each event window
            df_et['event'] = ''
            for _, event_row in events_df.iterrows():
                mask = (df_et['systemtime'] > event_row['starttime']) & (df_et['systemtime'] < event_row['endtime'])
                df_et.loc[mask, 'event'] = event_row['event']
                if not df_et.loc[mask].empty:
                    et_param = get_et_params(df_et.loc[mask])
                else:
                    et_param = tuple([np.nan] * 20)
                et_params_list.append(list(et_param))

    except Exception as e:
        # robust fallback: append NaN rows for each event that belonged to this participant in results_df
        print(f'ET processing failed for participant {participant}: {str(e)}')
        mask_participant = results_df["Name"] == participant
        num_events = mask_participant.sum()
        if num_events == 0:
            # at least add one row of NaNs
            et_params_list.append(list([np.nan] * 20))
        else:
            for _ in range(int(num_events)):
                et_params_list.append(list([np.nan] * 20))

# convert into DataFrame and append to results_df
params_df = pd.DataFrame(et_params_list)
params_df.columns = [
    'blink_count', 'average_blink_duration', 'max_blink_duration',
    'average_saccadic_duration', 'maximum_saccadic_duration', 'saccadic_count',
    'average_fixation_duration', 'maximum_fixation_duration', 'fixation_count',
    'gaze_entropy',
    'L_slope', 'L_max', 'L_mean', 'L_entropy', 'L_variation',
    'R_slope', 'R_max', 'R_mean', 'R_entropy', 'R_variation'
]

# If shape mismatch in rows vs results_df, align by index length:
# If params_df has fewer rows than results_df, pad with NaN rows
if params_df.shape[0] < results_df.shape[0]:
    n_missing = results_df.shape[0] - params_df.shape[0]
    pad = pd.DataFrame(np.nan, index=range(n_missing), columns=params_df.columns)
    params_df = pd.concat([params_df, pad], ignore_index=True)
elif params_df.shape[0] > results_df.shape[0]:
    # truncate extra rows (they correspond to extra events detected from xdf files)
    params_df = params_df.iloc[:results_df.shape[0], :].reset_index(drop=True)

df_t = pd.concat([results_df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)
df_t.to_csv('ET_results.csv', index=False)
print("Saved ET_results1.csv")
