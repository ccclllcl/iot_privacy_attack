"""
数据预处理：读取 CSV、时间对齐、重采样、滑窗、标签构造与数据集划分。

设计为「通用智能家居事件流」框架：列名通过配置文件映射；无标签时可切换为规则生成。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import ExperimentConfig
from src.features import extract_stat_features_matrix
from src.utils import ensure_dir, save_json, set_seed

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """预处理结果元数据（数组保存在 npz 中）。"""

    feature_names: List[str]
    class_names: List[str]
    seq_len: int
    freq: str


class LabelRuleEngine:
    """
    基于规则的标签推断（示例）：可替换为人工标注或更复杂模型。

    使用窗口内统计量与「窗口结束时刻」的小时数，结合设备分组（厨房/电脑等）做简单判定。
    优先级：sleep -> away -> cooking -> using_computer -> other（可按需调整顺序）。
    """

    def __init__(
        self,
        classes: List[str],
        rules_cfg: Dict[str, Any],
        device_groups: Dict[str, List[str]],
        feature_names: List[str],
    ) -> None:
        self.classes = classes
        self.rules_cfg = rules_cfg
        self.device_groups = device_groups
        self.feature_names = feature_names
        self._class_to_idx = {c: i for i, c in enumerate(classes)}

    def _indices_for_devices(self, names: List[str]) -> List[int]:
        idx = []
        for n in names:
            if n in self.feature_names:
                idx.append(self.feature_names.index(n))
        return idx

    def predict_window(
        self, window: np.ndarray, end_time: pd.Timestamp
    ) -> str:
        """
        window: shape (seq_len, n_features)，与 feature_names 对齐。
        end_time: 窗口最后一个时间步对应的时间戳。
        """
        hour = int(end_time.hour)
        mean_act = float(window.mean())
        peak = float(window.max())
        kitchen_idx = self._indices_for_devices(
            self.device_groups.get("kitchen_devices", [])
        )
        comp_idx = self._indices_for_devices(
            self.device_groups.get("computer_devices", [])
        )

        def in_ranges(ranges: List[List[int]], h: int) -> bool:
            for a, b in ranges:
                if a <= h < b:
                    return True
                # 支持跨午夜 [22,24) 与 [0,7) 分开配置
            return False

        def hour_in_rule(rule_name: str) -> bool:
            rc = self.rules_cfg.get(rule_name, {})
            ranges = rc.get("hour_ranges", [])
            return in_ranges(ranges, hour)

        # sleep
        sc = self.rules_cfg.get("sleep", {})
        if hour_in_rule("sleep") and mean_act <= float(
            sc.get("max_mean_activity", 1.0)
        ):
            return "sleep"

        # away
        ac = self.rules_cfg.get("away", {})
        if hour_in_rule("away"):
            if mean_act <= float(ac.get("max_mean_activity", 1.0)) and peak <= float(
                ac.get("max_peak_power", 1.0)
            ):
                return "away"

        # cooking（厨房设备能量占窗口总能量的比例）
        kc = self.rules_cfg.get("cooking", {})
        total_energy = float(window.sum()) + 1e-8
        if kitchen_idx:
            kitchen_sum = float(window[:, kitchen_idx].sum())
            kitchen_ratio = kitchen_sum / total_energy
        else:
            kitchen_ratio = 0.0
        if hour_in_rule("cooking"):
            if kitchen_ratio >= float(kc.get("min_kitchen_ratio", 0.2)) and mean_act >= float(
                kc.get("min_mean_activity", 0.0)
            ):
                return "cooking"

        # computer
        cc = self.rules_cfg.get("using_computer", {})
        if comp_idx:
            comp_sum = float(window[:, comp_idx].sum())
            comp_ratio = comp_sum / total_energy
        else:
            comp_ratio = 0.0
        if hour_in_rule("using_computer"):
            if comp_ratio >= float(cc.get("min_computer_ratio", 0.25)) and mean_act >= float(
                cc.get("min_mean_activity", 0.0)
            ):
                return "using_computer"

        return "other"


def _read_raw_dataframe(cfg: ExperimentConfig) -> pd.DataFrame:
    """读取原始 CSV 并按配置重命名列。"""
    colmap = cfg.nested("columns")
    path = cfg.path("paths", "raw_csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到原始数据文件: {path}。请先放置 Smart* 数据或运行 generate_mock_data.py 生成示例。"
        )
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    ts_col = colmap.get("timestamp", "timestamp")
    dev_col = colmap.get("device_id", "device_id")
    if ts_col not in df.columns or dev_col not in df.columns:
        raise ValueError(
            f"CSV 缺少必要列。需要 timestamp 映射列 {ts_col!r} 与 device_id 映射列 {dev_col!r}。"
        )

    rename: Dict[str, str] = {ts_col: "timestamp", dev_col: "device_id"}

    val_key = colmap.get("value")
    st_key = colmap.get("state")
    lb_key = colmap.get("label")

    if val_key and val_key in df.columns:
        vals = pd.to_numeric(df[val_key], errors="coerce").fillna(0.0).astype(float)
    else:
        vals = pd.Series(1.0, index=df.index, dtype=float)

    if st_key and st_key in df.columns:
        st = df[st_key].astype(str).str.lower()
        state_num = st.map(lambda x: 1.0 if x in ("on", "1", "true", "open") else 0.0)
        if val_key and val_key in df.columns:
            vals = vals + 0.5 * state_num
        else:
            vals = state_num

    df["value"] = vals

    if lb_key and lb_key in df.columns:
        rename[lb_key] = "behavior_label"

    df = df.rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad = df["timestamp"].isna().sum()
    if bad:
        logger.warning("丢弃无法解析的时间戳行数: %d", int(bad))
        df = df.dropna(subset=["timestamp"])
    df["device_id"] = df["device_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _apply_outliers(df: pd.DataFrame, method: str, iqr_k: float, zlim: float) -> pd.DataFrame:
    """对数值列做异常值处理（仅对设备通道列）。"""
    if method == "none" or df.empty:
        return df
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return df
    if method == "iqr":
        for c in num.columns:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
            df[c] = df[c].clip(lo, hi)
    elif method == "zscore_clip":
        for c in num.columns:
            mu = df[c].mean()
            sig = df[c].std()
            if sig == 0 or np.isnan(sig):
                continue
            z = (df[c] - mu) / sig
            df[c] = np.where(z.abs() > zlim, mu + np.sign(z) * zlim * sig, df[c])
    else:
        logger.warning("未知 outlier_method=%s，跳过异常值处理", method)
    return df


def _resample_wide_and_labels(
    df: pd.DataFrame,
    freq: str,
    fill_method: str,
    label_source: str,
    classes: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    将长表转为固定频率宽表；返回 (宽表数值, 每个时间点的行为标签)。

    标签列模式：同一时间段内取该桶内最后一条非空标签，再整体前向填充。
    """
    df = df.copy()
    df = df.set_index("timestamp")
    # 数值宽表：各设备在桶内求和（事件强度/功耗积分代理）
    wide = df.pivot_table(
        index=pd.Grouper(freq=freq),
        columns="device_id",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    )
    wide = wide.sort_index()
    if wide.shape[0] == 0:
        raise ValueError("重采样后无数据，请检查 CSV 时间范围与 freq 设置。")

    # 补齐完整时间轴，避免滑窗因「缺桶」而不连续
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    wide = wide.reindex(full_idx, fill_value=0.0)

    if label_source == "column" and "behavior_label" in df.columns:
        lb = df[df["behavior_label"].notna()]["behavior_label"]
        if lb.empty:
            label_series = pd.Series(index=full_idx, dtype=object)
        else:
            label_series = lb.groupby(pd.Grouper(freq=freq)).apply(
                lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan
            )
            label_series = label_series.reindex(full_idx)
        label_series = label_series.ffill().bfill()
        label_series = label_series.fillna("other")
        cls_set = set(classes)
        label_series = label_series.map(lambda s: s if str(s) in cls_set else "other")
    else:
        label_series = pd.Series("other", index=full_idx)

    if fill_method == "ffill":
        wide = wide.ffill().fillna(0.0)
    elif fill_method == "bfill":
        wide = wide.bfill().ffill().fillna(0.0)
    elif fill_method == "zero":
        wide = wide.fillna(0.0)
    elif fill_method == "drop_rows":
        wide = wide.dropna(how="any")
        label_series = label_series.reindex(wide.index).ffill().bfill()
    else:
        wide = wide.fillna(0.0)

    wide = wide.astype(np.float32)
    return wide, label_series.astype(str)


def _sliding_windows(
    wide: pd.DataFrame,
    label_series_int: pd.Series,
    seq_len: int,
    stride: int,
    label_source: str,
    rule_engine: Optional[LabelRuleEngine],
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """生成 X: (N, seq_len, F), y: (N,), 以及每个窗口结束时间。"""
    values = wide.to_numpy(dtype=np.float32)
    times = list(wide.index)
    n, f = values.shape
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ends: List[pd.Timestamp] = []

    for t in range(0, n - seq_len + 1, stride):
        w = values[t : t + seq_len]
        end_ts = pd.Timestamp(times[t + seq_len - 1])
        if label_source == "rules":
            if rule_engine is None:
                raise ValueError("label_source=rules 时必须构造 LabelRuleEngine")
            label_str = rule_engine.predict_window(w, end_ts)
            idx_map = rule_engine._class_to_idx
            y = int(idx_map.get(label_str, idx_map.get("other", 0)))
        else:
            y = int(label_series_int.iloc[t + seq_len - 1])
        xs.append(w)
        ys.append(y)
        ends.append(end_ts)

    if not xs:
        raise ValueError(
            "时间序列太短，无法构造滑窗。请增加数据或减小 preprocess.seq_len / 增大采样间隔。"
        )
    return np.stack(xs, axis=0), np.array(ys, dtype=np.int64), ends


def run_preprocess(cfg: ExperimentConfig) -> PreprocessResult:
    """
    执行完整预处理并保存到 data/processed/。

    输出文件：
    - sequences.npz: X_train, y_train, X_val, y_val, X_test, y_test
    - mlp_features.npz（可选）: 对应划分的统计特征矩阵
    - meta.json: 特征名、类别名、seq_len、freq 等
    """
    pp = cfg.nested("preprocess")
    lm = cfg.nested("label_mapping")
    paths = cfg.nested("paths")
    set_seed(cfg.random_seed())

    freq = str(pp.get("freq", "5min"))
    seq_len = int(pp.get("seq_len", 12))
    stride = int(pp.get("window_stride", 1))
    fill_method = str(pp.get("fill_method", "ffill"))
    out_meth = str(pp.get("outlier_method", "none"))
    iqr_k = float(pp.get("iqr_multiplier", 1.5))
    zlim = float(pp.get("zscore_limit", 3.0))

    classes: List[str] = list(lm.get("classes", []))
    if not classes:
        raise ValueError("label_mapping.classes 不能为空")

    label_source = str(lm.get("source", "column")).lower().strip()
    if label_source not in ("column", "rules"):
        raise ValueError("label_mapping.source 只能是 column 或 rules")

    df = _read_raw_dataframe(cfg)
    if label_source == "column" and "behavior_label" not in df.columns:
        logger.warning(
            "label_mapping.source=column 但原始 CSV 无标签列（检查 columns.label）。"
            "当前将全部时间步视为 other，建议改为 rules 或补充标注列。"
        )

    wide, label_series = _resample_wide_and_labels(
        df.reset_index(drop=True),
        freq=freq,
        fill_method=fill_method,
        label_source=label_source,
        classes=classes,
    )
    wide = _apply_outliers(wide, out_meth, iqr_k, zlim)

    le = LabelEncoder()
    le.fit(classes)

    rule_engine: Optional[LabelRuleEngine] = None
    if label_source == "rules":
        rule_engine = LabelRuleEngine(
            classes=classes,
            rules_cfg=lm.get("rules", {}),
            device_groups=cfg.nested("device_groups"),
            feature_names=list(wide.columns),
        )

    # 列模式：每个时间步的标签已编码为 0..C-1（与 classes 顺序一致）
    if label_source == "column":
        y_time = label_series.map(lambda s: s if s in classes else "other")
        label_codes = le.transform(y_time.astype(str))
        label_series_int = pd.Series(label_codes, index=wide.index, dtype=np.int64)
    else:
        label_series_int = pd.Series(0, index=wide.index, dtype=np.int64)

    X, y, _ = _sliding_windows(
        wide,
        label_series_int=label_series_int,
        seq_len=seq_len,
        stride=stride,
        label_source=label_source,
        rule_engine=rule_engine,
    )

    tr = float(pp.get("train_ratio", 0.7))
    va = float(pp.get("val_ratio", 0.15))
    te = float(pp.get("test_ratio", 0.15))
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 之和必须为 1.0")

    seed = cfg.random_seed()
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=te, random_state=seed, stratify=y
        )
        val_ratio_adj = va / (tr + va)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adj, random_state=seed, stratify=y_temp
        )
    except ValueError as e:
        logger.warning("分层划分失败（可能某类样本过少），改为随机划分: %s", e)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=te, random_state=seed
        )
        val_ratio_adj = va / (tr + va)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adj, random_state=seed
        )

    out_dir = ensure_dir(cfg.path("paths", "processed_dir"))
    np.savez_compressed(
        out_dir / "sequences.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    feat_cfg = cfg.nested("features")
    save_mlp = bool(pp.get("save_mlp_features", True))
    if save_mlp:
        feat_names = list(wide.columns)
        Xm_tr = extract_stat_features_matrix(X_train, feat_names, feat_cfg)
        Xm_va = extract_stat_features_matrix(X_val, feat_names, feat_cfg)
        Xm_te = extract_stat_features_matrix(X_test, feat_names, feat_cfg)
        np.savez_compressed(
            out_dir / "mlp_features.npz",
            X_train=Xm_tr,
            X_val=Xm_va,
            X_test=Xm_te,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
        mlp_dim = int(Xm_tr.shape[1])
    else:
        mlp_dim = 0

    meta = {
        "feature_names": list(wide.columns),
        "class_names": classes,
        "seq_len": seq_len,
        "freq": freq,
        "label_source": label_source,
        "mlp_feature_dim": mlp_dim,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(meta, out_dir / "meta.json")

    logger.info(
        "预处理完成: train=%d val=%d test=%d, F=%d, seq_len=%d",
        len(y_train),
        len(y_val),
        len(y_test),
        wide.shape[1],
        seq_len,
    )
    return PreprocessResult(
        feature_names=list(wide.columns),
        class_names=classes,
        seq_len=seq_len,
        freq=freq,
    )
