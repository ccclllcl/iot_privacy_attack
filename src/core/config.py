"""
配置加载模块：从 YAML 读取路径与超参数，并解析为相对于项目根目录的绝对路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 文件为字典。"""
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是字典结构")
    return data


def resolve_path(project_root: Path, maybe_relative: str) -> Path:
    """将配置中的路径转为绝对路径（相对路径相对于项目根目录）。"""
    p = Path(maybe_relative)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


class ExperimentConfig:
    """
    实验配置容器：持有原始字典，并提供路径解析方法。

    说明：保持为「扁平访问 raw dict」以便与 YAML 结构一致，避免维护大量 dataclass 字段。
    """

    def __init__(self, raw: Dict[str, Any], project_root: Path) -> None:
        self.raw = raw
        self.project_root = project_root.resolve()

    @classmethod
    def from_yaml(cls, yaml_path: Path, project_root: Optional[Path] = None) -> "ExperimentConfig":
        if project_root is None:
            project_root = yaml_path.resolve().parent.parent
        raw = load_yaml(yaml_path)
        return cls(raw, project_root)

    def path(self, *keys: str) -> Path:
        """按嵌套键读取字符串路径并解析，例如 path('paths', 'raw_csv')。"""
        d: Any = self.raw
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                raise KeyError(f"配置缺少键: {'.'.join(keys)}")
            d = d[k]
        if not isinstance(d, str):
            raise TypeError(f"路径配置必须是字符串: {'.'.join(keys)}")
        return resolve_path(self.project_root, d)

    def get(self, *keys: str, default: Any = None) -> Any:
        """安全嵌套 get。"""
        d: Any = self.raw
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def nested(self, *keys: str) -> Dict[str, Any]:
        """返回子字典（不存在则空 dict）。"""
        d: Any = self.raw
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return {}
            d = d[k]
        return d if isinstance(d, dict) else {}

    def random_seed(self) -> int:
        """
        全局随机种子：优先 `experiment.random_seed`，兼容旧配置中的
        `preprocess.random_seed` 或 `defense.random_seed`。
        """
        ex = self.nested("experiment")
        if "random_seed" in ex:
            return int(ex["random_seed"])
        pp = self.nested("preprocess")
        if "random_seed" in pp:
            return int(pp["random_seed"])
        dc = self.nested("defense")
        if "random_seed" in dc:
            return int(dc["random_seed"])
        return 42
