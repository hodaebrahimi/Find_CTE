#!/usr/bin/env python3
"""Extract (copy/symlink) CT and CTE images from the uncleaned dataset.

Workflow:
1) Run list_ct_data.py to generate ct_all_scans_combined.csv (and cte_*.csv)
2) Run this script on the combined CSV to physically organize files:
   - All CT scans (modality=CT)
   - CTE-only subset (Is_CTE==True)

It also copies/symlinks JSON sidecars when present.

Author: Hoda
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return

    if mode == 'symlink':
        dst.symlink_to(src)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(f'Unsupported mode: {mode}')


def build_source_paths(row: pd.Series, base_data_dir: Path) -> Tuple[Path, Optional[Path]]:
    """Reconstruct source NIfTI (and JSON) paths from CSV row."""
    patient_type = str(row['Patient_Type']).lower()
    patient_id = int(row['Patient_ID'])
    year = str(row['Year'])
    date = str(row['Date'])
    accession_deid = str(row.get('accession_deid', ''))
    scan_file = str(row['CT_Scan_File'])

    patient_folder = f"{patient_type}_{patient_id:06d}_{year}"
    date_folder = f"{accession_deid}_{date}" if accession_deid else f"{date}"

    nifti_path = base_data_dir / patient_folder / date_folder / scan_file

    json_path: Optional[Path] = None
    if scan_file.endswith('.nii.gz'):
        json_candidate = scan_file.replace('.nii.gz', '.json')
        p = nifti_path.parent / json_candidate
        json_path = p if p.exists() else None
    elif scan_file.endswith('.nii'):
        p = nifti_path.with_suffix('.json')
        json_path = p if p.exists() else None

    return nifti_path, json_path


def organize_from_csv(
    csv_path: Path,
    base_data_dir: Path,
    output_dir: Path,
    mode: str,
    dry_run: bool,
    include_uncertain: bool
) -> None:
    df = pd.read_csv(csv_path)

    total = len(df)
    if total == 0:
        logger.warning('No rows found in CSV.')
        return

    # Prepare outputs
    ct_root = output_dir / 'CT_all'
    cte_root = output_dir / 'CTE_only'

    for patient_type in ('cd', 'uc', 'ibdu'):
        ensure_dir(ct_root / patient_type)
        ensure_dir(cte_root / patient_type)

    ct_map: List[Dict] = []
    cte_map: List[Dict] = []

    ct_counter = 1
    cte_counter = 1

    for _, row in df.iterrows():
        pt = str(row.get('Patient_Type', '')).lower()
        if pt not in ('cd', 'uc', 'ibdu'):
            continue

        nifti_src, json_src = build_source_paths(row, base_data_dir)
        if not nifti_src.exists():
            logger.debug(f"Missing source NIfTI: {nifti_src}")
            continue

        # Always write CT_all
        ct_ext = '.nii.gz' if nifti_src.name.endswith('.nii.gz') else nifti_src.suffix
        ct_id = f"CT_{ct_counter:06d}"
        ct_dst = ct_root / pt / f"{ct_id}{ct_ext}"
        ct_json_dst = ct_root / pt / f"{ct_id}.json"

        is_cte = bool(row.get('Is_CTE', False))
        cte_label = str(row.get('CTE_Label', ''))
        is_uncertain = (cte_label.upper() == 'UNCERTAIN')

        if not dry_run:
            safe_link_or_copy(nifti_src, ct_dst, mode)
            if json_src and json_src.exists():
                safe_link_or_copy(json_src, ct_json_dst, mode)

        ct_map.append({
            'New_ID': ct_id,
            'New_Path': str(ct_dst),
            'Original_Path': str(nifti_src),
            'Original_JSON': str(json_src) if json_src else '',
            'Patient_Type': pt.upper(),
            'Patient_ID': row.get('Patient_ID', ''),
            'Year': row.get('Year', ''),
            'Date': row.get('Date', ''),
            'accession_deid': row.get('accession_deid', ''),
            'Is_CTE': is_cte,
            'CTE_Label': cte_label,
            'CTE_Score': row.get('CTE_Score', ''),
            'CTE_Score_Reasons': row.get('CTE_Score_Reasons', '')
        })
        ct_counter += 1

        # Optionally also write CTE_only
        if is_cte or (include_uncertain and is_uncertain):
            cte_ext = ct_ext
            cte_id = f"CTE_{cte_counter:06d}"
            cte_dst = cte_root / pt / f"{cte_id}{cte_ext}"
            cte_json_dst = cte_root / pt / f"{cte_id}.json"

            if not dry_run:
                safe_link_or_copy(nifti_src, cte_dst, mode)
                if json_src and json_src.exists():
                    safe_link_or_copy(json_src, cte_json_dst, mode)

            cte_map.append({
                'New_ID': cte_id,
                'New_Path': str(cte_dst),
                'Original_Path': str(nifti_src),
                'Original_JSON': str(json_src) if json_src else '',
                'Patient_Type': pt.upper(),
                'Patient_ID': row.get('Patient_ID', ''),
                'Year': row.get('Year', ''),
                'Date': row.get('Date', ''),
                'accession_deid': row.get('accession_deid', ''),
                'Is_CTE': is_cte,
                'CTE_Label': cte_label,
                'CTE_Score': row.get('CTE_Score', ''),
                'CTE_Score_Reasons': row.get('CTE_Score_Reasons', '')
            })
            cte_counter += 1

    # Save mappings
    ensure_dir(output_dir)
    pd.DataFrame(ct_map).to_csv(output_dir / 'CT_all_mapping.csv', index=False)
    pd.DataFrame(cte_map).to_csv(output_dir / 'CTE_only_mapping.csv', index=False)

    logger.info(f"Rows in CSV: {total}")
    logger.info(f"CT_all exported: {len(ct_map)}")
    logger.info(f"CTE_only exported: {len(cte_map)} (include_uncertain={include_uncertain})")
    logger.info(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract CT and CTE files from combined CSV.')
    parser.add_argument('--csv', type=Path, default=Path('ct_all_scans_combined.csv'))
    parser.add_argument('--base-data-dir', type=Path, default=Path('/data/ibd/data/patients'))
    parser.add_argument('--output-dir', type=Path, default=Path('extracted_CT_and_CTE'))
    parser.add_argument('--mode', choices=['copy', 'symlink'], default='copy')
    parser.add_argument('--include-uncertain', action='store_true', default=False,
                        help='Also include CTE_Label==UNCERTAIN in the CTE_only output.')
    parser.add_argument('--dry-run', action='store_true', default=False)

    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f'CSV not found: {args.csv}')
    if not args.base_data_dir.exists():
        raise FileNotFoundError(f'Base data dir not found: {args.base_data_dir}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    organize_from_csv(
        csv_path=args.csv,
        base_data_dir=args.base_data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        dry_run=args.dry_run,
        include_uncertain=args.include_uncertain,
    )


if __name__ == '__main__':
    main()
