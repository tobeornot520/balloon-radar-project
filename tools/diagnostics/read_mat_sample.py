from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    mat_files = sorted((PROJECT_ROOT / "data/raw/IQ_Data").glob("*.mat"))
    if not mat_files:
        print("No .mat files found under data/raw/IQ_Data")
        return 1

    sample = mat_files[0]
    print(f"Reading: {sample.name}")
    data = loadmat(sample)
    print(f"Variables: {list(data.keys())}")

    for key in ("local_data_H", "local_data_V", "H", "V", "data"):
        if key in data:
            array = data[key]
            print(
                f"{key}: shape={array.shape}, dtype={array.dtype}, "
                f"is_complex={np.iscomplexobj(array)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
