"""
Data Pipeline -- PHM Multi-Domain Dataset Loader & Processor
=============================================================
Loads real PHM datasets via the ``phmd`` library, resamples signals to a
common rate, windows them with per-channel z-score normalisation, and stores
the results in HDF5 files ready for PyTorch training.

Supported datasets (5 domains):
  - CWRU            1-ch bearing vibration      (classification)
  - PRONOSTIA       2-ch bearing degradation     (RUL regression)
  - CMAPSS          14-ch turbofan engine        (classification + RUL)
  - Paderborn       1-ch bearing vibration       (classification)
  - XJTU-SY         2-ch bearing degradation     (classification + RUL)
"""

import gc
import warnings

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.signal import resample as scipy_resample
from sklearn.preprocessing import LabelEncoder

from utils import load_config, set_seed, ensure_dirs, RUL_SENTINEL, CLS_SENTINEL

# Suppress phmd informational noise during bulk loading
warnings.filterwarnings("ignore", category=FutureWarning)


# ===================================================================
# 1.  Resampling
# ===================================================================

def resample_signal(signal, original_fs, target_fs):
    """
    Resample a multi-channel signal from *original_fs* to *target_fs*.

    Parameters
    ----------
    signal : ndarray, shape (C, T)
        Multi-channel time-series (channels-first).
    original_fs : float
        Original sampling frequency in Hz.
    target_fs : float
        Desired sampling frequency in Hz.

    Returns
    -------
    ndarray, shape (C, T')
        Resampled signal.  ``scipy.signal.resample`` provides built-in
        anti-aliasing via its Fourier method.
    """
    if abs(original_fs - target_fs) < 1e-6:
        return signal
    C, T = signal.shape
    T_new = int(round(T * target_fs / original_fs))
    if T_new < 1:
        T_new = 1
    resampled = np.empty((C, T_new), dtype=np.float32)
    for ch in range(C):
        resampled[ch] = scipy_resample(signal[ch], T_new).astype(np.float32)
    return resampled


# ===================================================================
# 2.  Windowing & Normalisation  (multivariate)
# ===================================================================

def window_and_normalize(signals, labels, rul_targets, window_length, stride):
    """
    Slide a window over each unit's signal and z-score-normalise per channel.

    Parameters
    ----------
    signals : list of ndarray, each (C, T_i)
        Variable-length multi-channel signals, one per unit/sample.
    labels : ndarray (N,)
        Classification label per unit (CLS_SENTINEL = -1 if no cls target).
    rul_targets : ndarray (N,) of scalars  OR  list of ndarray each (T_i,)
        If scalar array: all windows from that unit get the same RUL value.
        If list of per-timestep arrays: each window gets rul[end_of_window].
        Use RUL_SENTINEL = -1.0 when there is no regression target.
    window_length : int
    stride : int

    Returns
    -------
    windows    : ndarray (M, C, L)
    win_labels : ndarray (M,)  int64
    win_rul    : ndarray (M,)  float32
    """
    # Detect whether rul_targets is per-timestep (list of arrays) or per-unit (1-D scalar array)
    per_timestep_rul = (
        isinstance(rul_targets, list)
        and len(rul_targets) > 0
        and isinstance(rul_targets[0], np.ndarray)
    )

    all_windows, all_labels, all_rul = [], [], []
    for i in range(len(signals)):
        sig = signals[i]          # (C, T_i)
        C, T = sig.shape
        if T < window_length:
            continue
        for start in range(0, T - window_length + 1, stride):
            end = start + window_length
            w = sig[:, start:end].copy()
            # Per-channel z-score
            for ch in range(C):
                mu = w[ch].mean()
                std = w[ch].std()
                w[ch] = (w[ch] - mu) / std if std > 1e-8 else w[ch] - mu
            all_windows.append(w)
            all_labels.append(labels[i])
            if per_timestep_rul:
                # Use the RUL at the last timestep of this window
                rul_series = rul_targets[i]
                idx = min(end - 1, len(rul_series) - 1)
                all_rul.append(float(rul_series[idx]))
            else:
                all_rul.append(float(rul_targets[i]))

    if len(all_windows) == 0:
        C_out = signals[0].shape[0] if len(signals) > 0 else 1
        return (np.empty((0, C_out, window_length), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32))

    return (np.array(all_windows, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            np.array(all_rul, dtype=np.float32))


# ===================================================================
# 3.  Per-Dataset Loaders  (phmd)
# ===================================================================

def _load_phmd_task(phmd_name, task_name):
    """
    Load a task from phmd and normalize to a pandas DataFrame.

    phmd's `task.load()` may return:
      - A pandas DataFrame  (best case)
      - A tuple (X, y) where X is a numpy array and y is labels/targets
      - A tuple (X,)  (no target)

    Normalised output is always a DataFrame where:
      - Signal columns contain 1-D array cells (one per recording)
      - An optional '_target' column holds per-recording labels / RUL values
    """
    import pandas as pd
    from phmd.datasets import Dataset as PHMDDataset

    ds = PHMDDataset(phmd_name)
    task = ds[task_name]
    result = task.load()

    # ---- Already a DataFrame ----
    if isinstance(result, pd.DataFrame):
        print(f"    phmd [{phmd_name}/{task_name}] → DataFrame "
              f"shape={result.shape}, cols={list(result.columns)[:8]}")
        return result

    # ---- List → treat as tuple (phmd returns lists for CMAPSS, MFPT, etc.) ----
    if isinstance(result, list):
        print(f"    phmd [{phmd_name}/{task_name}] → list "
              f"len={len(result)}, types={[type(r).__name__ for r in result[:4]]}")
        result = tuple(result)

    # ---- Tuple (X, y) or (X,) ----
    if isinstance(result, tuple):
        print(f"    phmd [{phmd_name}/{task_name}] → tuple "
              f"len={len(result)}, types={[type(r).__name__ for r in result]}")

        X = result[0]
        y = result[1] if len(result) > 1 else None

        # Unwrap nested lists (phmd sometimes returns [[df]] or [[array]])
        if isinstance(X, list):
            print(f"    X is nested list len={len(X)}, unwrapping...")
            if len(X) == 1:
                X = X[0]
            else:
                # Multiple items — concatenate DataFrames or arrays
                if all(isinstance(item, pd.DataFrame) for item in X):
                    X = pd.concat(X, ignore_index=True)
                elif all(isinstance(item, np.ndarray) for item in X):
                    X = np.concatenate(X, axis=0)
                else:
                    X = X[0]
        if isinstance(y, list):
            if len(y) == 1:
                y = y[0]
            elif all(isinstance(item, np.ndarray) for item in y):
                y = np.concatenate(y, axis=0)

        if hasattr(X, 'shape'):
            print(f"    X.shape={X.shape}, dtype={getattr(X, 'dtype', '?')}")
        if y is not None and hasattr(y, 'shape'):
            print(f"    y.shape={y.shape}, dtype={getattr(y, 'dtype', '?')}")

        # Convert X → DataFrame with array-valued cells
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            if X.ndim == 1:
                # Single recording
                df = pd.DataFrame({'ch_0': [X]})
            elif X.ndim == 2:
                # (N_recordings, T) — one channel
                N = X.shape[0]
                df = pd.DataFrame({'ch_0': [X[i] for i in range(N)]})
            elif X.ndim == 3:
                # (N_recordings, C, T) — channels first
                N, C = X.shape[0], X.shape[1]
                df = pd.DataFrame(
                    {f'ch_{c}': [X[i, c] for i in range(N)] for c in range(C)}
                )
            else:
                raise ValueError(f"Unexpected X.ndim={X.ndim} for "
                                 f"{phmd_name}/{task_name}")
        else:
            raise ValueError(f"Unexpected X type {type(X).__name__} for "
                             f"{phmd_name}/{task_name}")

        # Attach target column (only when lengths match — phmd sometimes returns
        # a y array with a different length than X, in which case the target
        # information is already inside the X DataFrame as a named column)
        if y is not None:
            y_arr = y if isinstance(y, np.ndarray) else np.asarray(y)
            y_flat = y_arr.ravel()
            if len(y_flat) == len(df):
                df['_target'] = y_flat
            else:
                print(f"    Warning: y length {len(y_flat)} != df length {len(df)} "
                      f"— skipping _target (target col should already be in df)")

        print(f"    → converted DataFrame shape={df.shape}, "
              f"cols={list(df.columns)}")
        return df

    raise ValueError(
        f"phmd returned unexpected type {type(result).__name__} "
        f"for {phmd_name}/{task_name}"
    )


def _available_columns(df, requested_cols):
    """
    Return the subset of *requested_cols* that exist in *df*.
    If none match (e.g. phmd used generic 'ch_0' names instead of 'DE'),
    fall back to all non-metadata / non-target columns.
    """
    available = [c for c in requested_cols if c in df.columns]
    if available:
        return available
    # Fallback: use every column that isn't a metadata or target column
    meta_cols = {'_target', 'unit', 'cycle', 'time', 'timestamp', 'index'}
    fallback = [c for c in df.columns if c.lower() not in meta_cols]
    if fallback:
        print(f"    Warning: signal columns {requested_cols} not found — "
              f"using fallback columns: {fallback}")
        return fallback
    return list(df.columns)


# ------------------------------------------------------------------
#  Tall-format helpers
# ------------------------------------------------------------------

def _is_tall_format(df, sig_cols):
    """Return True if signal columns contain scalar values (tall format)."""
    try:
        val = df[sig_cols[0]].iloc[0]
        return np.isscalar(val) or (isinstance(val, np.ndarray) and val.ndim == 0)
    except Exception:
        return False


def _find_group_col(df, sig_cols, target_col):
    """
    Find the column that identifies individual units/bearings/engines.
    Looks for common group-identifier names first, then falls back to
    any remaining column that is not a signal or target column.
    """
    group_keywords = {'unit', 'bearing', 'engine', 'id', 'experiment',
                      'machine', 'run', 'subject', 'sample', 'trial'}
    sig_set = set(sig_cols)
    # Prefer a column whose name matches a known group keyword
    for col in df.columns:
        if col not in sig_set and col != target_col and col.lower() in group_keywords:
            return col
    # Fallback: first remaining column
    for col in df.columns:
        if col not in sig_set and col != target_col:
            return col
    return None


def _group_tall_cls(df, sig_cols, target_col, rul_col=None,
                    max_rul_clip=125, normalize_rul=True):
    """
    Group a tall DataFrame by its unit column.
    Returns (signals, labels, rul_array) suitable for window_and_normalize.

    For classification: label = mode of the group's target column.
    For RUL: returns a *list* of per-timestep RUL arrays (one per unit),
             so window_and_normalize assigns end-of-window RUL to each window.
    """
    group_col = _find_group_col(df, sig_cols, target_col)

    if group_col is None:
        # No grouping column — treat entire DataFrame as one recording
        sig_arrays = [df[c].values.astype(np.float32) for c in sig_cols]
        min_len = min(len(s) for s in sig_arrays)
        stacked = np.stack([s[:min_len] for s in sig_arrays], axis=0)
        label = int(df[target_col].iloc[0]) if target_col else CLS_SENTINEL
        rul = RUL_SENTINEL
        return [stacked], np.array([label], dtype=np.int64), np.array([rul], dtype=np.float32)

    print(f"    Grouping by '{group_col}' (tall format) ...")
    signals, labels_list, rul_list = [], [], []
    per_timestep = (rul_col is not None)

    for unit_id, grp in df.groupby(group_col, sort=True):
        # Build (C, T) signal array for this unit
        sig_arrays = [grp[c].values.astype(np.float32) for c in sig_cols]
        min_len = min(len(s) for s in sig_arrays)
        stacked = np.stack([s[:min_len] for s in sig_arrays], axis=0)  # (C, T)
        signals.append(stacked)

        # Classification label: most common value in this group
        if target_col and target_col in grp.columns:
            from scipy.stats import mode as scipy_mode
            lbl_vals = grp[target_col].values
            try:
                lbl = scipy_mode(lbl_vals, keepdims=True).mode[0]
            except Exception:
                lbl = lbl_vals[0]
            labels_list.append(lbl)
        else:
            labels_list.append(CLS_SENTINEL)

        # RUL: per-timestep array
        if per_timestep and rul_col in grp.columns:
            raw_rul = grp[rul_col].values[:min_len].astype(np.float32)
            raw_rul = np.clip(raw_rul, 0, max_rul_clip)
            if normalize_rul:
                raw_rul = raw_rul / max_rul_clip
            rul_list.append(raw_rul)   # ndarray — triggers per-timestep path
        else:
            rul_list.append(RUL_SENTINEL)   # scalar — same for all windows

    labels_arr = np.array(labels_list)
    # ALWAYS apply LabelEncoder to guarantee contiguous 0-indexed labels
    # (phmd may return 1-based or non-contiguous integer labels)
    le = LabelEncoder()
    labels_arr = le.fit_transform(labels_arr).astype(np.int64)
    print(f"    Label classes: {le.classes_} → encoded to {np.unique(labels_arr)}")

    # If all RUL values are scalars → return plain ndarray
    if all(np.isscalar(r) for r in rul_list):
        rul_out = np.array(rul_list, dtype=np.float32)
    else:
        rul_out = rul_list   # list of arrays → per-timestep mode

    print(f"    → {len(signals)} units, signal shapes: {signals[0].shape} ... "
          f"{signals[-1].shape}")
    return signals, labels_arr, rul_out


# ------------------------------------------------------------------
#  CWRU  (classification: 4 fault classes)
# ------------------------------------------------------------------

def load_cwru(ds_cfg, rul_cfg):
    task_cfg = ds_cfg["tasks"][0]
    df = _load_phmd_task(ds_cfg["phmd_name"], task_cfg["task_name"])
    sig_cols = _available_columns(df, ds_cfg["signal_columns"])
    if not sig_cols:
        raise ValueError(f"CWRU: none of {ds_cfg['signal_columns']} found. "
                         f"Cols={list(df.columns)}")
    target_col = _find_target_column(df, sig_cols)

    if _is_tall_format(df, sig_cols):
        signals, labels, rul = _group_tall_cls(df, sig_cols, target_col)
    else:
        le = LabelEncoder()
        encoded = le.fit_transform(df[target_col].values)
        signals, labels = [], []
        for idx in range(len(df)):
            arrs = [_to_1d_array(df[c].iloc[idx]) for c in sig_cols]
            ml = min(len(a) for a in arrs)
            signals.append(np.stack([a[:ml] for a in arrs], axis=0).astype(np.float32))
            labels.append(encoded[idx])
        labels = np.array(labels, dtype=np.int64)
        rul = np.full(len(labels), RUL_SENTINEL, dtype=np.float32)

    print(f"  CWRU: {len(signals)} units, classes={np.unique(labels)}")
    return signals, labels, rul


# ------------------------------------------------------------------
#  PRONOSTIA  (RUL regression)
# ------------------------------------------------------------------

def load_pronostia(ds_cfg, rul_cfg):
    task_cfg = ds_cfg["tasks"][0]
    df = _load_phmd_task(ds_cfg["phmd_name"], task_cfg["task_name"])
    sig_cols = _available_columns(df, ds_cfg["signal_columns"])
    if not sig_cols:
        raise ValueError(f"PRONOSTIA: none of {ds_cfg['signal_columns']} found. "
                         f"Cols={list(df.columns)}")
    rul_col = _find_target_column(df, sig_cols)
    max_rul_clip = rul_cfg.get("max_rul_clip", 125)
    normalize_rul = rul_cfg.get("normalize", True)

    if _is_tall_format(df, sig_cols):
        # RUL varies per timestep — pass rul_col for per-timestep extraction
        signals, labels, rul = _group_tall_cls(
            df, sig_cols, target_col=None, rul_col=rul_col,
            max_rul_clip=max_rul_clip, normalize_rul=normalize_rul)
        labels = np.full(len(signals), CLS_SENTINEL, dtype=np.int64)
    else:
        signals, rul_vals = [], []
        for idx in range(len(df)):
            arrs = [_to_1d_array(df[c].iloc[idx]) for c in sig_cols]
            ml = min(len(a) for a in arrs)
            signals.append(np.stack([a[:ml] for a in arrs], axis=0).astype(np.float32))
            rv = min(float(df[rul_col].iloc[idx]), max_rul_clip)
            rul_vals.append(rv / max_rul_clip if normalize_rul else rv)
        labels = np.full(len(signals), CLS_SENTINEL, dtype=np.int64)
        rul = np.array(rul_vals, dtype=np.float32)

    print(f"  PRONOSTIA: {len(signals)} units")
    return signals, labels, rul


# ------------------------------------------------------------------
#  CMAPSS  (classification + RUL)
# ------------------------------------------------------------------

def load_cmapss(ds_cfg, rul_cfg):
    """CMAPSS: classification + RUL. Load the RUL task (which includes signals)."""
    max_rul_clip = rul_cfg.get("max_rul_clip", 125)
    normalize_rul = rul_cfg.get("normalize", True)

    cls_task, rul_task = None, None
    for t in ds_cfg["tasks"]:
        if t["type"] == "classification":
            cls_task = t
        elif t["type"] == "regression":
            rul_task = t

    # Load the primary task (prefer RUL task as it has full sensor data)
    df = None
    task_used = None
    for task_cfg in ([rul_task, cls_task] if rul_task else [cls_task]):
        if task_cfg is None:
            continue
        try:
            df = _load_phmd_task(ds_cfg["phmd_name"], task_cfg["task_name"])
            task_used = task_cfg
            break
        except Exception as e:
            print(f"  CMAPSS: could not load {task_cfg['task_name']}: {e}")

    if df is None:
        raise ValueError("CMAPSS: could not load any task")

    sig_cols = _available_columns(df, ds_cfg["signal_columns"])
    if not sig_cols:
        raise ValueError(f"CMAPSS: none of {ds_cfg['signal_columns']} found. "
                         f"Cols={list(df.columns)}")

    target_col = _find_target_column(df, sig_cols)
    has_rul = (task_used is not None and task_used["type"] == "regression")

    if _is_tall_format(df, sig_cols):
        rul_col = target_col if has_rul else None
        # For classification label in RUL task, try to find a separate label col
        cls_col = None
        if not has_rul:
            cls_col = target_col
        signals, labels, rul = _group_tall_cls(
            df, sig_cols,
            target_col=cls_col,
            rul_col=rul_col,
            max_rul_clip=max_rul_clip,
            normalize_rul=normalize_rul)
        if cls_col is None:
            labels = np.full(len(signals), CLS_SENTINEL, dtype=np.int64)
        if rul_col is None:
            rul = np.full(len(signals), RUL_SENTINEL, dtype=np.float32)
    else:
        signals, labels_list, rul_list = [], [], []
        for idx in range(len(df)):
            arrs = [_to_1d_array(df[c].iloc[idx]) for c in sig_cols]
            ml = min(len(a) for a in arrs)
            signals.append(np.stack([a[:ml] for a in arrs], axis=0).astype(np.float32))
            tv = df[target_col].iloc[idx]
            if has_rul:
                rv = min(float(tv), max_rul_clip)
                rul_list.append(rv / max_rul_clip if normalize_rul else rv)
                labels_list.append(CLS_SENTINEL)
            else:
                labels_list.append(tv)
                rul_list.append(RUL_SENTINEL)
        labels = np.array(labels_list, dtype=np.int64)
        rul = np.array(rul_list, dtype=np.float32)

    print(f"  CMAPSS: {len(signals)} units")
    return signals, labels, rul


# ------------------------------------------------------------------
#  Paderborn (PUBD16)  (classification: 3 fault classes)
# ------------------------------------------------------------------

def load_paderborn(ds_cfg, rul_cfg):
    task_cfg = ds_cfg["tasks"][0]
    df = _load_phmd_task(ds_cfg["phmd_name"], task_cfg["task_name"])
    sig_cols = _available_columns(df, ds_cfg["signal_columns"])
    if not sig_cols:
        raise ValueError(f"Paderborn: none of {ds_cfg['signal_columns']} found. "
                         f"Cols={list(df.columns)}")
    target_col = _find_target_column(df, sig_cols)

    if _is_tall_format(df, sig_cols):
        signals, labels, rul = _group_tall_cls(df, sig_cols, target_col)
    else:
        le = LabelEncoder()
        encoded = le.fit_transform(df[target_col].values)
        signals, labels = [], []
        for idx in range(len(df)):
            arrs = [_to_1d_array(df[c].iloc[idx]) for c in sig_cols]
            ml = min(len(a) for a in arrs)
            signals.append(np.stack([a[:ml] for a in arrs], axis=0).astype(np.float32))
            labels.append(encoded[idx])
        labels = np.array(labels, dtype=np.int64)
        rul = np.full(len(labels), RUL_SENTINEL, dtype=np.float32)

    print(f"  Paderborn: {len(signals)} units, classes={np.unique(labels)}")
    return signals, labels, rul


# ------------------------------------------------------------------
#  MFPT  (classification: 3 classes — normal / inner / outer race)
# ------------------------------------------------------------------

def load_mfpt(ds_cfg, rul_cfg):
    task_cfg = ds_cfg["tasks"][0]
    df = _load_phmd_task(ds_cfg["phmd_name"], task_cfg["task_name"])
    sig_cols = _available_columns(df, ds_cfg["signal_columns"])
    if not sig_cols:
        raise ValueError(f"MFPT: none of {ds_cfg['signal_columns']} found. "
                         f"Cols={list(df.columns)}")
    target_col = _find_target_column(df, sig_cols)

    if _is_tall_format(df, sig_cols):
        signals, labels, rul = _group_tall_cls(df, sig_cols, target_col)
    else:
        le = LabelEncoder()
        encoded = le.fit_transform(df[target_col].values)
        signals, labels = [], []
        for idx in range(len(df)):
            arrs = [_to_1d_array(df[c].iloc[idx]) for c in sig_cols]
            ml = min(len(a) for a in arrs)
            signals.append(np.stack([a[:ml] for a in arrs], axis=0).astype(np.float32))
            labels.append(encoded[idx])
        labels = np.array(labels, dtype=np.int64)
        rul = np.full(len(labels), RUL_SENTINEL, dtype=np.float32)

    print(f"  MFPT: {len(signals)} units, classes={np.unique(labels)}")
    return signals, labels, rul


# ------------------------------------------------------------------
#  XJTU-SY  (classification + RUL)  [kept for reference]
# ------------------------------------------------------------------

def load_xjtu_sy(ds_cfg, rul_cfg):
    max_rul_clip = rul_cfg.get("max_rul_clip", 125)
    normalize_rul = rul_cfg.get("normalize", True)

    cls_task, rul_task = None, None
    for t in ds_cfg["tasks"]:
        if t["type"] == "classification":
            cls_task = t
        elif t["type"] == "regression":
            rul_task = t

    # Load whichever task is available; prefer one with both signals and RUL
    df, has_cls_col, has_rul_col = None, False, False
    for task_cfg, is_rul in [(rul_task, True), (cls_task, False)]:
        if task_cfg is None:
            continue
        try:
            df_try = _load_phmd_task(ds_cfg["phmd_name"], task_cfg["task_name"])
            df = df_try
            has_rul_col = is_rul
            has_cls_col = not is_rul
            break
        except Exception as e:
            print(f"  XJTU-SY: could not load {task_cfg['task_name']}: {e}")

    if df is None:
        raise ValueError("XJTU-SY: could not load any task")

    sig_cols = _available_columns(df, ds_cfg["signal_columns"])
    if not sig_cols:
        raise ValueError(f"XJTU-SY: none of {ds_cfg['signal_columns']} found. "
                         f"Cols={list(df.columns)}")
    target_col = _find_target_column(df, sig_cols)

    if _is_tall_format(df, sig_cols):
        rul_col = target_col if has_rul_col else None
        cls_col = target_col if has_cls_col else None
        signals, labels, rul = _group_tall_cls(
            df, sig_cols,
            target_col=cls_col,
            rul_col=rul_col,
            max_rul_clip=max_rul_clip,
            normalize_rul=normalize_rul)
        if cls_col is None:
            labels = np.full(len(signals), CLS_SENTINEL, dtype=np.int64)
        if rul_col is None:
            rul = np.full(len(signals), RUL_SENTINEL, dtype=np.float32)
    else:
        signals, labels_list, rul_list = [], [], []
        for idx in range(len(df)):
            arrs = [_to_1d_array(df[c].iloc[idx]) for c in sig_cols]
            ml = min(len(a) for a in arrs)
            signals.append(np.stack([a[:ml] for a in arrs], axis=0).astype(np.float32))
            tv = df[target_col].iloc[idx]
            if has_rul_col:
                rv = min(float(tv), max_rul_clip)
                rul_list.append(rv / max_rul_clip if normalize_rul else rv)
                labels_list.append(CLS_SENTINEL)
            else:
                labels_list.append(tv)
                rul_list.append(RUL_SENTINEL)
        labels = np.array(labels_list, dtype=np.int64)
        rul = np.array(rul_list, dtype=np.float32)

    print(f"  XJTU-SY: {len(signals)} units, classes={np.unique(labels)}")
    return signals, labels, rul


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

# Dispatch table mapping dataset names to loader functions
DATASET_LOADERS = {
    "CWRU": load_cwru,
    "PRONOSTIA": load_pronostia,
    "CMAPSS": load_cmapss,
    "Paderborn": load_paderborn,
    "MFPT": load_mfpt,
    "XJTU-SY": load_xjtu_sy,   # kept for reference, not in default config
}


def _to_1d_array(val):
    """Convert a phmd cell value (scalar, list, ndarray, Series) to a 1-D float32 array."""
    if isinstance(val, np.ndarray):
        return val.astype(np.float32).ravel()
    try:
        import pandas as pd
        if isinstance(val, pd.Series):
            return val.values.astype(np.float32).ravel()
    except ImportError:
        pass
    if isinstance(val, (list, tuple)):
        return np.array(val, dtype=np.float32)
    # Scalar
    return np.array([float(val)], dtype=np.float32)


def _find_target_column(df, signal_cols):
    """
    Return the name of the target column in *df*.

    Priority:
      1. '_target'  — injected by _load_phmd_task for (X, y) returns
      2. First non-signal, non-metadata column
      3. Last column as fallback
    """
    if df is None:
        return None
    # Prefer explicit target column injected from tuple returns
    if '_target' in df.columns:
        return '_target'
    meta_cols = {"unit", "cycle", "time", "timestamp", "index"}
    sig_set = set(signal_cols)
    for col in df.columns:
        if col not in sig_set and col.lower() not in meta_cols:
            return col
    return df.columns[-1]


# ===================================================================
# 4.  HDF5 Storage
# ===================================================================

def _store_dataset_hdf5(path, windows, labels, rul, sampling_freq,
                        dataset_id, num_channels, c_max):
    """
    Write one dataset's processed windows to an HDF5 file.

    Signals are zero-padded along the channel axis to ``c_max``.
    """
    M, C, L = windows.shape
    if C < c_max:
        pad = np.zeros((M, c_max - C, L), dtype=np.float32)
        windows = np.concatenate([windows, pad], axis=1)

    with h5py.File(path, "w") as f:
        f.create_dataset("signals", data=windows, chunks=True, compression="gzip")
        f.create_dataset("labels", data=labels, chunks=True, compression="gzip")
        f.create_dataset("rul_targets", data=rul, chunks=True, compression="gzip")
        f.create_dataset("sampling_freqs",
                         data=np.full(M, sampling_freq, dtype=np.float32),
                         chunks=True, compression="gzip")
        f.create_dataset("dataset_id",
                         data=np.full(M, dataset_id, dtype=np.int64),
                         chunks=True, compression="gzip")
        f.create_dataset("num_channels",
                         data=np.full(M, num_channels, dtype=np.int64),
                         chunks=True, compression="gzip")
        f.attrs["window_length"] = L
        f.attrs["c_max"] = c_max
        f.attrs["num_windows"] = M

    print(f"    -> saved {path}: {M} windows, padded channels {C}->{c_max}, L={L}")


def combine_hdf5_files(per_dataset_paths, combined_path, dataset_cfgs):
    """
    Concatenate per-dataset HDF5 files into a single combined file.
    """
    all_sigs, all_labels, all_rul = [], [], []
    all_freqs, all_dsids, all_nch = [], [], []

    for p in per_dataset_paths:
        with h5py.File(p, "r") as f:
            all_sigs.append(f["signals"][:])
            all_labels.append(f["labels"][:])
            all_rul.append(f["rul_targets"][:])
            all_freqs.append(f["sampling_freqs"][:])
            all_dsids.append(f["dataset_id"][:])
            all_nch.append(f["num_channels"][:])

    signals = np.concatenate(all_sigs)
    labels = np.concatenate(all_labels)
    rul = np.concatenate(all_rul)
    freqs = np.concatenate(all_freqs)
    dsids = np.concatenate(all_dsids)
    nch = np.concatenate(all_nch)

    with h5py.File(combined_path, "w") as f:
        f.create_dataset("signals", data=signals, chunks=True, compression="gzip")
        f.create_dataset("labels", data=labels, chunks=True, compression="gzip")
        f.create_dataset("rul_targets", data=rul, chunks=True, compression="gzip")
        f.create_dataset("sampling_freqs", data=freqs, chunks=True, compression="gzip")
        f.create_dataset("dataset_id", data=dsids, chunks=True, compression="gzip")
        f.create_dataset("num_channels", data=nch, chunks=True, compression="gzip")

        f.attrs["num_datasets"] = len(dataset_cfgs)
        for i, dc in enumerate(dataset_cfgs):
            f.attrs[f"dataset_{i}_name"] = dc["name"]

    print(f"\nCombined HDF5 -> {combined_path}")
    print(f"  signals:       {signals.shape}")
    print(f"  labels:        {labels.shape}  (CLS_SENTINEL={CLS_SENTINEL})")
    print(f"  rul_targets:   {rul.shape}     (RUL_SENTINEL={RUL_SENTINEL})")
    print(f"  sampling_freqs:{freqs.shape}")
    print(f"  dataset_id:    {dsids.shape}   unique={np.unique(dsids)}")
    print(f"  num_channels:  {nch.shape}     unique={np.unique(nch)}")
    for i, dc in enumerate(dataset_cfgs):
        mask = dsids == i
        print(f"  [{i}] {dc['name']}: {mask.sum()} windows")


# ===================================================================
# 5.  Main Generation Function
# ===================================================================

def generate_all_datasets(config_path="configs/config.yaml"):
    """
    Load all 5 datasets from phmd, process them, and store in HDF5.

    Datasets are processed ONE AT A TIME to control peak memory usage
    (XJTU-SY alone can be ~19 GB on disk).
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    data_cfg = cfg["data"]
    rul_cfg = cfg.get("rul", {})
    target_fs = data_cfg["target_sample_rate"]
    window_length = data_cfg["window_length"]
    cmapss_window_length = data_cfg["cmapss_window_length"]
    stride_div = data_cfg["stride_divisor"]
    hdf5_dir = Path(data_cfg["hdf5_dir"])
    combined_path = data_cfg["combined_hdf5"]
    ensure_dirs(hdf5_dir, Path(combined_path).parent)

    dataset_cfgs = cfg["datasets"]

    # Determine C_max across all datasets
    c_max = max(dc["num_channels"] for dc in dataset_cfgs)
    print(f"C_max (max channels across datasets) = {c_max}")

    per_dataset_paths = []

    for ds_idx, ds_cfg in enumerate(dataset_cfgs):
        name = ds_cfg["name"]
        is_cycle_based = ds_cfg.get("is_cycle_based", False)
        original_fs = ds_cfg["original_sampling_freq"]
        num_ch = ds_cfg["num_channels"]

        print(f"\n{'='*60}")
        print(f"[{ds_idx}] Processing {name}  (phmd={ds_cfg['phmd_name']}, "
              f"fs={original_fs}, channels={num_ch}, cycle_based={is_cycle_based})")
        print(f"{'='*60}")

        # --- Load raw data ---
        loader_fn = DATASET_LOADERS.get(name)
        if loader_fn is None:
            raise ValueError(f"No loader registered for dataset '{name}'")
        signals, labels, rul_targets = loader_fn(ds_cfg, rul_cfg)

        # --- Resample (skip for cycle-based datasets) ---
        if not is_cycle_based:
            print(f"  Resampling {original_fs} Hz -> {target_fs} Hz ...")
            signals = [resample_signal(s, original_fs, target_fs) for s in signals]
            wl = window_length
        else:
            print(f"  Cycle-based dataset: skipping resampling, using cmapss_window_length={cmapss_window_length}")
            wl = cmapss_window_length

        stride = max(1, wl // stride_div)

        # --- Window & normalise ---
        print(f"  Windowing: L={wl}, stride={stride} ...")
        windows, win_labels, win_rul = window_and_normalize(
            signals, labels, rul_targets, wl, stride
        )
        print(f"  Windows: {windows.shape}, labels unique={np.unique(win_labels)}")

        # --- Validate labels vs config num_classes ---
        for task in ds_cfg["tasks"]:
            if task["type"] == "classification":
                cfg_nc = task["num_classes"]
                valid_labels = win_labels[win_labels >= 0]
                if len(valid_labels) > 0:
                    actual_max = int(valid_labels.max())
                    actual_nc = len(np.unique(valid_labels))
                    if actual_max >= cfg_nc:
                        print(f"  *** WARNING: max label {actual_max} >= "
                              f"config num_classes {cfg_nc}! Adjusting config.")
                        task["num_classes"] = actual_nc
                    print(f"  Label check: {actual_nc} unique classes, "
                          f"range [0, {actual_max}], config num_classes={task['num_classes']}")

        # --- Pad cycle-based windows to standard length ---
        # CMAPSS has L=50 but the transformer needs L=window_length=2560.
        # We zero-pad along the time axis so all datasets share the same shape.
        if is_cycle_based and windows.shape[2] < window_length:
            M_w, C_w, L_w = windows.shape
            padded = np.zeros((M_w, C_w, window_length), dtype=np.float32)
            padded[:, :, :L_w] = windows
            windows = padded
            print(f"  Padded cycle-based windows: L={L_w} -> {window_length}")

        # --- Store per-dataset HDF5 ---
        ds_hdf5 = hdf5_dir / f"{name.lower().replace('-', '_')}.h5"
        _store_dataset_hdf5(
            str(ds_hdf5), windows, win_labels, win_rul,
            sampling_freq=original_fs,
            dataset_id=ds_idx,
            num_channels=num_ch,
            c_max=c_max,
        )
        per_dataset_paths.append(str(ds_hdf5))

        # Free memory before next dataset
        del signals, labels, rul_targets, windows, win_labels, win_rul
        gc.collect()

    # --- Combine into one file ---
    combine_hdf5_files(per_dataset_paths, combined_path, dataset_cfgs)

    return combined_path


# ===================================================================
# 6.  PyTorch Dataset  (in-memory)
# ===================================================================

class PHMDataset(Dataset):
    """
    In-memory PyTorch Dataset backed by an HDF5 file.

    All data is loaded into RAM on construction to avoid slow
    per-sample I/O on network file-systems.

    Parameters
    ----------
    hdf5_path : str
        Path to the combined (or per-dataset) HDF5 file.
    indices : array-like, optional
        Subset of global row indices to load.
    dataset_id : int, optional
        If given, load only rows belonging to this dataset.
    """

    def __init__(self, hdf5_path, indices=None, dataset_id=None):
        with h5py.File(hdf5_path, "r") as f:
            if dataset_id is not None:
                dsids = f["dataset_id"][:]
                indices = np.where(dsids == dataset_id)[0]
            elif indices is not None:
                indices = np.asarray(indices)
            else:
                indices = np.arange(len(f["signals"]))

            # h5py requires indices in increasing order for fancy indexing
            sorted_idx = np.sort(indices)

            self.indices = indices
            self.signals = f["signals"][sorted_idx]
            self.labels = f["labels"][sorted_idx]
            self.rul_targets = f["rul_targets"][sorted_idx]
            self.freqs = f["sampling_freqs"][sorted_idx]
            self.ds_ids = f["dataset_id"][sorted_idx]
            self.num_channels = f["num_channels"][sorted_idx]

            # Restore original ordering so shuffled splits stay shuffled
            if not np.array_equal(indices, sorted_idx):
                inv = np.argsort(np.argsort(indices))
                self.signals = self.signals[inv]
                self.labels = self.labels[inv]
                self.rul_targets = self.rul_targets[inv]
                self.freqs = self.freqs[inv]
                self.ds_ids = self.ds_ids[inv]
                self.num_channels = self.num_channels[inv]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sig = torch.from_numpy(self.signals[idx])          # (C_max, L)
        label = int(self.labels[idx])                       # -1 if no cls target
        rul = float(self.rul_targets[idx])                  # -1.0 if no rul target
        freq = float(self.freqs[idx])
        ds_id = int(self.ds_ids[idx])
        n_ch = int(self.num_channels[idx])
        return sig, label, rul, freq, ds_id, n_ch


# ===================================================================
# 7.  Train / Val / Test Splits
# ===================================================================

def get_split_indices(hdf5_path, dataset_id, train_r=0.7, val_r=0.15, seed=42):
    """
    Return train / val / test index arrays for a specific ``dataset_id``.

    Splits are performed on the *global* row indices so they can be
    passed directly to ``PHMDataset(hdf5_path, indices=...)``.
    """
    with h5py.File(hdf5_path, "r") as f:
        dsids = f["dataset_id"][:]
    all_idx = np.where(dsids == dataset_id)[0]
    rng = np.random.RandomState(seed)
    rng.shuffle(all_idx)
    n = len(all_idx)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return all_idx[:n_train], all_idx[n_train:n_train + n_val], all_idx[n_train + n_val:]


def get_all_split_indices(hdf5_path, num_datasets, train_r=0.7, val_r=0.15, seed=42):
    """Return combined train / val / test indices across all datasets."""
    train_all, val_all, test_all = [], [], []
    for ds_id in range(num_datasets):
        tr, va, te = get_split_indices(hdf5_path, ds_id, train_r, val_r, seed)
        train_all.append(tr)
        val_all.append(va)
        test_all.append(te)
    return np.concatenate(train_all), np.concatenate(val_all), np.concatenate(test_all)


def make_loader(hdf5_path, indices, batch_size, shuffle=True, num_workers=0):
    """Create a DataLoader from a subset of the HDF5 file."""
    ds = PHMDataset(hdf5_path, indices=indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# ===================================================================
# 8.  Verification
# ===================================================================

def verify_datasets(config_path="configs/config.yaml"):
    """
    Verify the combined HDF5 file: shapes, sentinels, and PyTorch loading.
    """
    cfg = load_config(config_path)
    combined_path = cfg["data"]["combined_hdf5"]
    wl = cfg["data"]["window_length"]
    cmapss_wl = cfg["data"]["cmapss_window_length"]
    dataset_cfgs = cfg["datasets"]
    c_max = max(dc["num_channels"] for dc in dataset_cfgs)

    print("=== HDF5 Verification ===")
    with h5py.File(combined_path, "r") as f:
        signals = f["signals"]
        labels = f["labels"]
        rul = f["rul_targets"]
        freqs = f["sampling_freqs"]
        dsids = f["dataset_id"]
        nch = f["num_channels"]

        print(f"signals shape:       {signals.shape}  (expected (N, {c_max}, L))")
        print(f"labels shape:        {labels.shape}")
        print(f"rul_targets shape:   {rul.shape}")
        print(f"sampling_freqs shape:{freqs.shape}")
        print(f"dataset_id shape:    {dsids.shape}")
        print(f"num_channels shape:  {nch.shape}")

        assert signals.shape[1] == c_max, (
            f"Channel dim mismatch: {signals.shape[1]} vs c_max={c_max}")

        dsids_arr = dsids[:]
        labels_arr = labels[:]
        rul_arr = rul[:]

        for i, dc in enumerate(dataset_cfgs):
            mask = dsids_arr == i
            n = mask.sum()
            cls_vals = np.unique(labels_arr[mask])
            rul_vals = rul_arr[mask]
            valid_rul = rul_vals[rul_vals > RUL_SENTINEL + 0.5]
            rul_info = (f"RUL [{valid_rul.min():.2f}, {valid_rul.max():.2f}]"
                        if len(valid_rul) > 0 else "no RUL")
            print(f"  [{i}] {dc['name']:12s}: {n:>7d} windows, "
                  f"classes={cls_vals}, {rul_info}")

    # Test PyTorch Dataset loading
    print("\n--- PyTorch Dataset test ---")
    ds = PHMDataset(combined_path, dataset_id=0)
    sig, lbl, rul_val, freq, did, n_ch = ds[0]
    print(f"Sample 0: signal={sig.shape}, label={lbl}, rul={rul_val:.4f}, "
          f"freq={freq}, ds_id={did}, n_ch={n_ch}")
    assert sig.shape[0] == c_max, f"Channel mismatch in tensor: {sig.shape[0]} vs {c_max}"

    # Test DataLoader
    loader = make_loader(combined_path, np.arange(min(256, len(ds))), batch_size=32)
    for batch in loader:
        sigs, lbls, ruls, frs, dids, nchs = batch
        print(f"Batch: signals={sigs.shape}, labels={lbls.shape}, "
              f"rul={ruls.shape}, freqs={frs.shape}, ds_ids={dids.shape}, n_ch={nchs.shape}")
        break

    # Test splits
    print("\n--- Split test ---")
    num_ds = len(dataset_cfgs)
    train_idx, val_idx, test_idx = get_all_split_indices(
        combined_path, num_ds,
        train_r=cfg["data"]["train_ratio"],
        val_r=cfg["data"]["val_ratio"],
        seed=cfg["seed"],
    )
    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}, Total: {total}")

    print("\nAll verifications passed.")


# ===================================================================
# 9.  Entry Point
# ===================================================================

if __name__ == "__main__":
    generate_all_datasets()
    verify_datasets()
