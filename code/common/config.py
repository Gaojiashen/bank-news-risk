"""读取YAML配置，路径以项目根目录为基准。"""
from pathlib import Path
import yaml
from .paths import ROOT

def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return result

def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
