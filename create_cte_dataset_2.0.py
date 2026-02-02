#!/usr/bin/env python3
"""
CT/CTE Dataset Organizer

This script organizes imaging files into a clean directory structure.

Preferred workflow (uncleaned dataset):
1) Run list_ct_data.py to generate ct_all_scans_combined.csv (with CTE scoring)
2) Run this script to export both:
    - CT_all/   (all CT scans)
    - CTE_only/ (CTE subset, optionally include UNCERTAIN)

It supports copying or symlinking NIfTI and JSON sidecars.

Author: Hoda
"""

from __future__ import annotations

import argparse
import pandas as pd
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DEFAULT_EXCLUDE_SUBSTRINGS = [
    'seg', 'mask', 'label', 'roi', 'annotation', 'annot', 'pred', 'prediction'
]


def create_directory_structure(base_output_dir: Path, top_level: str) -> None:
    """
    Create the organized directory structure.
    
    Args:
        base_output_dir: Base output directory path
        top_level: Top-level folder name (e.g. 'CT_all', 'CTE_only')
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if top_level:
        base_output_dir = base_output_dir / top_level
        base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each patient type
    (base_output_dir / 'cd').mkdir(exist_ok=True)
    (base_output_dir / 'uc').mkdir(exist_ok=True)
    (base_output_dir / 'ibdu').mkdir(exist_ok=True)
    
    logger.info(f"Created directory structure at: {base_output_dir}")


def _is_excluded_filename(name: str, exclude_substrings: List[str]) -> bool:
    name_lower = name.lower()
    return any(s.lower() in name_lower for s in exclude_substrings)


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == 'symlink':
        dst.symlink_to(src)
    elif mode == 'copy':
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def find_source_file(row: pd.Series, base_data_dir: Path) -> Path:
    """
    Find the source file path based on CSV row information.
    
    Args:
        row: CSV row with file information
        base_data_dir: Base directory for patient data
        
    Returns:
        Path to source file
    """
    patient_type = row['Patient_Type']
    patient_id = int(row['Patient_ID'])  # Convert to int first
    year = row['Year']
    date = row['Date']
    scan_file = row['CT_Scan_File']
    
    # Construct the expected path structure with 6-digit patient_id
    patient_folder = f"{patient_type}_{patient_id:06d}_{year}"
    date_folder = f"{row['accession_deid']}_{date}"
    
    source_path = base_data_dir / patient_folder / date_folder / scan_file
    
    return source_path


def resolve_source_paths_from_row(row: pd.Series, base_data_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve source NIfTI + JSON paths.

    Priority:
    1) Use full paths if present in combined CSV (NIfTI_Full_Path/JSON_Full_Path)
    2) Fallback to reconstructing from base_data_dir + folder structure
    """
    nifti_path: Optional[Path] = None
    json_path: Optional[Path] = None

    nifti_full = str(row.get('NIfTI_Full_Path', '') or '').strip()
    if nifti_full:
        p = Path(nifti_full)
        if p.exists():
            nifti_path = p

    json_full = str(row.get('JSON_Full_Path', '') or '').strip()
    if json_full:
        p = Path(json_full)
        if p.exists():
            json_path = p

    if nifti_path is None:
        try:
            nifti_path = find_source_file(row, base_data_dir)
        except Exception:
            nifti_path = None

    if json_path is None and nifti_path is not None and nifti_path.exists():
        name = nifti_path.name
        if name.endswith('.nii.gz'):
            candidate = nifti_path.parent / name.replace('.nii.gz', '.json')
            json_path = candidate if candidate.exists() else None
        elif name.endswith('.nii'):
            candidate = nifti_path.with_suffix('.json')
            json_path = candidate if candidate.exists() else None

    return nifti_path, json_path


def _get_patient_type_folder(row: pd.Series) -> Optional[str]:
    pt = str(row.get('Patient_Type', '')).strip().lower()
    if pt in ('cd', 'uc', 'ibdu'):
        return pt
    return None


def organize_from_combined_csv(
    combined_csv: Path,
    base_data_dir: Path,
    output_dir: Path,
    mode: str,
    include_uncertain: bool,
    exclude_substrings: List[str],
    prefix_ct: str = 'CT',
    prefix_cte: str = 'CTE'
) -> None:
    df = pd.read_csv(combined_csv)
    if df.empty:
        logger.warning(f"No rows found in {combined_csv}")
        return

    create_directory_structure(output_dir, 'CT_all')
    create_directory_structure(output_dir, 'CTE_only')

    ct_map: List[Dict] = []
    cte_map: List[Dict] = []

    ct_counter = 1
    cte_counter = 1

    for _, row in df.iterrows():
        pt_folder = _get_patient_type_folder(row)
        if pt_folder is None:
            continue

        scan_file = str(row.get('CT_Scan_File', '') or '')
        if scan_file and _is_excluded_filename(scan_file, exclude_substrings):
            continue

        nifti_src, json_src = resolve_source_paths_from_row(row, base_data_dir)
        if nifti_src is None or not nifti_src.exists():
            continue

        # Export to CT_all
        ext = '.nii.gz' if nifti_src.name.endswith('.nii.gz') else nifti_src.suffix
        ct_id = f"{prefix_ct}_{ct_counter:06d}"
        ct_dst = output_dir / 'CT_all' / pt_folder / f"{ct_id}{ext}"
        ct_json_dst = output_dir / 'CT_all' / pt_folder / f"{ct_id}.json"

        safe_link_or_copy(nifti_src, ct_dst, mode)
        if json_src and json_src.exists():
            safe_link_or_copy(json_src, ct_json_dst, mode)

        is_cte = bool(row.get('Is_CTE', False))
        cte_label = str(row.get('CTE_Label', '') or '')
        is_uncertain = cte_label.upper() == 'UNCERTAIN'

        ct_map.append({
            'New_ID': ct_id,
            'New_Path': str(ct_dst),
            'Original_Path': str(nifti_src),
            'Original_JSON': str(json_src) if json_src else '',
            'Patient_Type': str(row.get('Patient_Type', '')).upper(),
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

        # Export to CTE_only
        if is_cte or (include_uncertain and is_uncertain):
            cte_id = f"{prefix_cte}_{cte_counter:06d}"
            cte_dst = output_dir / 'CTE_only' / pt_folder / f"{cte_id}{ext}"
            cte_json_dst = output_dir / 'CTE_only' / pt_folder / f"{cte_id}.json"

            safe_link_or_copy(nifti_src, cte_dst, mode)
            if json_src and json_src.exists():
                safe_link_or_copy(json_src, cte_json_dst, mode)

            cte_map.append({
                'New_ID': cte_id,
                'New_Path': str(cte_dst),
                'Original_Path': str(nifti_src),
                'Original_JSON': str(json_src) if json_src else '',
                'Patient_Type': str(row.get('Patient_Type', '')).upper(),
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

    pd.DataFrame(ct_map).to_csv(output_dir / 'CT_all_mapping.csv', index=False)
    pd.DataFrame(cte_map).to_csv(output_dir / 'CTE_only_mapping.csv', index=False)

    logger.info(f"Combined rows: {len(df)}")
    logger.info(f"Exported CT_all: {len(ct_map)}")
    logger.info(f"Exported CTE_only: {len(cte_map)} (include_uncertain={include_uncertain})")


def copy_and_rename_files(csv_files: Dict[str, Path], base_data_dir: Path, 
                         output_dir: Path, combined_numbering: bool = True) -> List[Dict]:
    """
    Copy and rename files according to the organized structure.
    
    Args:
        csv_files: Dictionary mapping patient types to CSV file paths
        base_data_dir: Base directory containing original patient data
        output_dir: Output directory for organized files
        combined_numbering: If True, use combined numbering across types
        
    Returns:
        List of dictionaries with mapping information
    """
    mapping_records = []
    
    if combined_numbering:
        # Combined numbering across all patient types
        counter = 1
        
        for patient_type, csv_path in csv_files.items():
            logger.info(f"Processing {patient_type.upper()} cases from: {csv_path}")
            
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Found {len(df)} {patient_type.upper()} cases")
            
            # Process each case
            for _, row in df.iterrows():
                try:
                    # Find source file
                    source_file = find_source_file(row, base_data_dir)
                    
                    if not source_file.exists():
                        logger.warning(f"Source file not found: {source_file}")
                        continue
                    
                    # Create new filename with sequential numbering
                    # Handle .nii.gz extension properly
                    if source_file.name.endswith('.nii.gz'):
                        file_extension = '.nii.gz'
                    else:
                        file_extension = source_file.suffix
                    new_filename = f"IBD_{counter:04d}{file_extension}"
                    
                    # Destination path
                    dest_dir = output_dir / patient_type.lower()
                    dest_path = dest_dir / new_filename
                    
                    # Copy file
                    shutil.copy2(source_file, dest_path)
                    
                    # Record mapping with NEW CSV fields
                    mapping_record = {
                        'IBD_ID': f"IBD_{counter:04d}",
                        'Original_File': source_file.name,
                        'Original_Path': str(source_file),
                        'New_Path': str(dest_path),
                        'Patient_Type': patient_type.upper(),
                        'Patient_ID': row['Patient_ID'],
                        'Date': row['Date'],
                        'File_Size_MB': row['File_Size_MB'],
                        
                        # NEW: Robust CTE scoring fields
                        'Is_CTE': row.get('Is_CTE', False),
                        'CTE_Score': row.get('CTE_Score', 0),
                        'CTE_Score_Reasons': row.get('CTE_Score_Reasons', ''),
                        'Has_Enterography': row.get('Has_Enterography', False),
                        'Has_Contrast': row.get('Has_Contrast', False),
                        'Has_Valid_Anatomy': row.get('Has_Valid_Anatomy', False),
                        'Anatomy_Found': row.get('Anatomy_Found', ''),
                        'Phase_Classification': row.get('Phase_Classification', ''),
                        'Phase_Type': row.get('Phase_Type', ''),
                        
                        # JSON metadata fields
                        'json_ProcedureStepDescription': row.get('json_ProcedureStepDescription', ''),
                        'json_ProtocolName': row.get('json_ProtocolName', ''),
                        'json_SeriesDescription': row.get('json_SeriesDescription', ''),
                        'json_BodyPartExamined': row.get('json_BodyPartExamined', ''),
                        
                        # Source tracking
                        'identification_source': row.get('identification_source', ''),
                        'Enterography_Found_In': row.get('Enterography_Found_In', ''),
                        'Anatomy_Found_In': row.get('Anatomy_Found_In', ''),
                        'Phase_Found_In': row.get('Phase_Found_In', '')
                    }
                    
                    mapping_records.append(mapping_record)
                    
                    logger.debug(f"Copied: {source_file.name} -> {new_filename}")
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row {row.get('Patient_ID', 'unknown')}: {e}")
                    continue
    
    else:
        # Separate numbering for each patient type
        for patient_type, csv_path in csv_files.items():
            logger.info(f"Processing {patient_type.upper()} cases from: {csv_path}")
            
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Found {len(df)} {patient_type.upper()} cases")
            
            counter = 1  # Reset counter for each patient type
            
            # Process each case
            for _, row in df.iterrows():
                try:
                    # Find source file
                    source_file = find_source_file(row, base_data_dir)
                    
                    if not source_file.exists():
                        logger.warning(f"Source file not found: {source_file}")
                        continue
                    
                    # Create new filename with sequential numbering
                    # Handle .nii.gz extension properly
                    if source_file.name.endswith('.nii.gz'):
                        file_extension = '.nii.gz'
                    else:
                        file_extension = source_file.suffix
                    new_filename = f"IBD_{counter:04d}{file_extension}"
                    
                    # Destination path
                    dest_dir = output_dir / patient_type.lower()
                    dest_path = dest_dir / new_filename
                    
                    # Copy file
                    shutil.copy2(source_file, dest_path)
                    
                    # Record mapping with NEW CSV fields
                    mapping_record = {
                        'IBD_ID': f"IBD_{counter:04d}",
                        'Original_File': source_file.name,
                        'Original_Path': str(source_file),
                        'New_Path': str(dest_path),
                        'Patient_Type': patient_type.upper(),
                        'Patient_ID': row['Patient_ID'],
                        'Date': row['Date'],
                        'File_Size_MB': row['File_Size_MB'],
                        
                        # NEW: Robust CTE scoring fields
                        'Is_CTE': row.get('Is_CTE', False),
                        'CTE_Score': row.get('CTE_Score', 0),
                        'CTE_Score_Reasons': row.get('CTE_Score_Reasons', ''),
                        'Has_Enterography': row.get('Has_Enterography', False),
                        'Has_Contrast': row.get('Has_Contrast', False),
                        'Has_Valid_Anatomy': row.get('Has_Valid_Anatomy', False),
                        'Anatomy_Found': row.get('Anatomy_Found', ''),
                        'Phase_Classification': row.get('Phase_Classification', ''),
                        'Phase_Type': row.get('Phase_Type', ''),
                        
                        # JSON metadata fields
                        'json_ProcedureStepDescription': row.get('json_ProcedureStepDescription', ''),
                        'json_ProtocolName': row.get('json_ProtocolName', ''),
                        'json_SeriesDescription': row.get('json_SeriesDescription', ''),
                        'json_BodyPartExamined': row.get('json_BodyPartExamined', ''),
                        
                        # Source tracking
                        'identification_source': row.get('identification_source', ''),
                        'Enterography_Found_In': row.get('Enterography_Found_In', ''),
                        'Anatomy_Found_In': row.get('Anatomy_Found_In', ''),
                        'Phase_Found_In': row.get('Phase_Found_In', '')
                    }
                    
                    mapping_records.append(mapping_record)
                    
                    logger.debug(f"Copied: {source_file.name} -> {new_filename}")
                    counter += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row {row.get('Patient_ID', 'unknown')}: {e}")
                    continue
    
    return mapping_records


def save_mapping_file(mapping_records: List[Dict], output_dir: Path) -> None:
    """
    Save the mapping file with original to new filename mappings.
    
    Args:
        mapping_records: List of mapping dictionaries
        output_dir: Output directory
    """
    mapping_df = pd.DataFrame(mapping_records)
    mapping_file = output_dir / 'IBD_file_mapping.csv'
    mapping_df.to_csv(mapping_file, index=False)
    
    logger.info(f"Saved mapping file: {mapping_file}")
    logger.info(f"Total files processed: {len(mapping_records)}")


def main():
    parser = argparse.ArgumentParser(description='Organize CT and CTE datasets from CSVs.')
    subparsers = parser.add_subparsers(dest='command', required=False)

    # Preferred: organize from combined CSV (exports CT_all and CTE_only)
    p_combined = subparsers.add_parser('from-combined', help='Export CT_all and CTE_only from combined CSV.')
    p_combined.add_argument('--combined-csv', type=Path, default=Path('ct_all_scans_combined.csv'))
    p_combined.add_argument('--base-data-dir', type=Path, default=Path('/data/ibd/data/patients'))
    p_combined.add_argument('--output-dir', type=Path, default=Path('organized_CT_and_CTE'))
    p_combined.add_argument('--mode', choices=['copy', 'symlink'], default='copy')
    p_combined.add_argument('--include-uncertain', action='store_true', default=False)
    p_combined.add_argument('--exclude-substrings', nargs='*', default=DEFAULT_EXCLUDE_SUBSTRINGS)
    p_combined.add_argument('--prefix-ct', type=str, default='CT')
    p_combined.add_argument('--prefix-cte', type=str, default='CTE')

    # Legacy: organize only CTE from the 3 per-type CSVs
    p_legacy = subparsers.add_parser('legacy-cte-only', help='Legacy: organize only CTE from cte_uc/cd/ibdu CSVs.')
    p_legacy.add_argument('--base-data-dir', type=Path, default=Path('/data/ibd/data/patients'))
    p_legacy.add_argument('--output-dir', type=Path, default=Path('/data/ailab/hoda2/ibd_data_arrangement/CTE_cases_2.0'))
    p_legacy.add_argument('--combined-numbering', action='store_true', default=True)
    p_legacy.add_argument('--csv-uc', type=Path, default=Path('/data/ailab/hoda2/ibd_data_arrangement/cte_uc_data.csv'))
    p_legacy.add_argument('--csv-cd', type=Path, default=Path('/data/ailab/hoda2/ibd_data_arrangement/cte_cd_data.csv'))
    p_legacy.add_argument('--csv-ibdu', type=Path, default=Path('/data/ailab/hoda2/ibd_data_arrangement/cte_ibdu_data.csv'))

    args = parser.parse_args()

    # Default to from-combined if no subcommand provided
    command = args.command or 'from-combined'

    if command == 'from-combined':
        combined_csv = args.combined_csv
        base_data_dir = args.base_data_dir
        output_dir = args.output_dir

        if not combined_csv.exists():
            logger.error(f"Combined CSV not found: {combined_csv}")
            return
        if not base_data_dir.exists():
            logger.error(f"Base data directory not found: {base_data_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Starting CT/CTE organization from combined CSV...")
        logger.info(f"Combined CSV: {combined_csv}")
        logger.info(f"Base data directory: {base_data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Include UNCERTAIN: {args.include_uncertain}")

        organize_from_combined_csv(
            combined_csv=combined_csv,
            base_data_dir=base_data_dir,
            output_dir=output_dir,
            mode=args.mode,
            include_uncertain=args.include_uncertain,
            exclude_substrings=args.exclude_substrings,
            prefix_ct=args.prefix_ct,
            prefix_cte=args.prefix_cte
        )
        logger.info("Done.")
        return

    # Legacy mode
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir
    combined_numbering = args.combined_numbering

    csv_files = {
        'uc': args.csv_uc,
        'cd': args.csv_cd,
        'ibdu': args.csv_ibdu
    }

    logger.info("Starting LEGACY CTE-only file organization...")
    logger.info(f"Source data directory: {base_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Combined numbering: {combined_numbering}")

    existing_csv_files = {}
    for patient_type, csv_path in csv_files.items():
        if csv_path.exists():
            existing_csv_files[patient_type] = csv_path
            logger.info(f"Found {patient_type.upper()} CSV: {csv_path}")
        else:
            logger.warning(f"CSV file not found (skipping): {csv_path}")

    if not existing_csv_files:
        logger.error("No CSV files found!")
        return

    if not base_data_dir.exists():
        logger.error(f"Source data directory not found: {base_data_dir}")
        return

    # Legacy output keeps the old structure
    create_directory_structure(output_dir, top_level='')
    mapping_records = copy_and_rename_files(
        csv_files=existing_csv_files,
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        combined_numbering=combined_numbering
    )
    save_mapping_file(mapping_records, output_dir)
    logger.info("File organization complete!")


if __name__ == "__main__":
    main()