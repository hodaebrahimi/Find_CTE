#!/usr/bin/env python3
"""Collect CD-patient CTE scans into CTEs_cd_patients/ for transfer to another server.

Reads cte_cd_data.csv (produced by list_ct_data.py) and copies each NIfTI
(+ its JSON sidecar when available) into a flat output folder with sequential naming.
"""

import shutil
from pathlib import Path
import pandas as pd

# ── Configuration ──────────────────────────────────────────────
CSV_PATH  = Path('/data/ailab/hoda2/ibd_data_arrangement/cte_cd_data.csv')
OUT_DIR   = Path('/data/ailab/hoda2/ibd_data_arrangement/CTEs_cd_patients')
MAX_FILES = 1000
# ───────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Rows in CSV: {len(df)}")

copied, skipped = 0, 0

for idx, row in df.iterrows():
    nifti_src = Path(str(row.get('NIfTI_Full_Path', '')).strip())

    if not nifti_src.exists():
        skipped += 1
        continue

    # Sequential naming: CTE_CD_000001.nii.gz
    ext = '.nii.gz' if nifti_src.name.endswith('.nii.gz') else nifti_src.suffix
    new_name = f"CTE_CD_{copied + 1:06d}"

    shutil.copy2(nifti_src, OUT_DIR / f"{new_name}{ext}")

    # Copy JSON sidecar if available
    json_src_str = str(row.get('JSON_Full_Path', '') or '').strip()
    if json_src_str:
        json_src = Path(json_src_str)
        if json_src.exists():
            shutil.copy2(json_src, OUT_DIR / f"{new_name}.json")

    copied += 1
    if copied % 500 == 0:
        print(f"  ... copied {copied} files so far")
    if copied >= MAX_FILES:
        break

if copied >= MAX_FILES:
    print(f"\nReached MAX_FILES limit ({MAX_FILES}).")
print(f"Done — copied {copied}, skipped {skipped} (missing source), total rows {len(df)}")
print(f"Output: {OUT_DIR}")