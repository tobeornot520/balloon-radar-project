from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from scipy.io import loadmat, whosmat
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "classification_dataset"

CLASS_SPECS = (
    (0, "uav", ("无人机", "#U65e0#U4eba#U673a", "uav")),
    (1, "balloon_line_array", ("气球+10米线阵", "balloon_line_array")),
    (2, "balloon_solar_panel", ("气球+太阳能板", "balloon_solar_panel")),
    (3, "balloon_box", ("气球+方盒子", "气球+方盒", "balloon_box")),
    (4, "balloon_circuit_board", ("气球+电路板", "balloon_circuit_board")),
)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _polar5(h: np.ndarray, v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    ah = 20.0 * np.log10(np.abs(h) + eps)
    av = 20.0 * np.log10(np.abs(v) + eps)
    rlog = np.log((np.abs(h) + eps) / (np.abs(v) + eps))
    phase_diff = np.angle(h * np.conj(v))
    corr = np.real(h * np.conj(v)) / (np.abs(h) * np.abs(v) + eps)
    return np.stack((ah, av, rlog, phase_diff, np.clip(corr, -1.0, 1.0)), axis=0).astype(np.float32)


def _find_class_dir(split_root: Path, aliases: tuple[str, ...]) -> Optional[Path]:
    for alias in aliases:
        p = split_root / alias
        if p.is_dir():
            return p
    return None


class ClassificationRadarDatasetV2(Dataset):
    """读取当前分类目录。一个 MAT 第三维上的每个索引会展开成一个样本。"""

    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        split: Literal["train", "val", "test"] = "train",
        input_mode: Literal["polar5", "ri4"] = "polar5",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = _resolve_path(data_root)
        self.split = split
        self.input_mode = input_mode
        split_root = self.data_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"找不到分类目录: {split_root}")

        records: list[dict[str, Any]] = []
        for label, class_name, aliases in CLASS_SPECS:
            class_dir = _find_class_dir(split_root, aliases)
            if class_dir is None:
                continue
            for mat_path in sorted(class_dir.glob("*.mat")):
                schema = {name: shape for name, shape, _dtype in whosmat(mat_path)}
                if "UAV_h" not in schema or "UAV_v" not in schema:
                    raise KeyError(f"{mat_path} 缺少 UAV_h/UAV_v，实际字段 {sorted(schema)}")
                h_shape = tuple(schema["UAV_h"])
                v_shape = tuple(schema["UAV_v"])
                if h_shape != v_shape or len(h_shape) != 3 or h_shape[1] < 3:
                    raise ValueError(f"{mat_path.name} 形状异常: H={h_shape}, V={v_shape}")
                for sample_index in range(h_shape[2]):
                    records.append({
                        "mat_path": mat_path,
                        "sample_index": sample_index,
                        "label": label,
                        "class_name": class_name,
                    })

        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise ValueError(f"{split_root} 中未发现可用分类样本")
        self.records = records
        self._cache_path: Optional[Path] = None
        self._cache_h: Optional[np.ndarray] = None
        self._cache_v: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.records)

    def _load_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_path != path:
            data = loadmat(path)
            h = np.asarray(data["UAV_h"])
            v = np.asarray(data["UAV_v"])
            if not np.iscomplexobj(h) or not np.iscomplexobj(v):
                raise TypeError(f"{path.name} 不是复数 IQ")
            if not np.isfinite(h).all() or not np.isfinite(v).all():
                raise ValueError(f"{path.name} 包含 NaN 或 Inf")
            self._cache_path, self._cache_h, self._cache_v = path, h, v
        assert self._cache_h is not None and self._cache_v is not None
        return self._cache_h, self._cache_v

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        h_all, v_all = self._load_file(rec["mat_path"])
        i = rec["sample_index"]
        angle_h = np.real(h_all[:, :2, i]).astype(np.float32)
        angle_v = np.real(v_all[:, :2, i]).astype(np.float32)
        h = h_all[:, 2:, i]
        v = v_all[:, 2:, i]
        if self.input_mode == "polar5":
            x = _polar5(h, v)
        elif self.input_mode == "ri4":
            x = np.stack((h.real, h.imag, v.real, v.imag), axis=0).astype(np.float32)
        else:
            raise ValueError(f"未知 input_mode: {self.input_mode}")
        return {
            "input": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "label": torch.tensor(rec["label"], dtype=torch.long),
            "class_name": rec["class_name"],
            "sample_index": int(i),
            "mat_path": str(rec["mat_path"]),
            "angle_h": torch.from_numpy(angle_h),
            "angle_v": torch.from_numpy(angle_v),
        }
