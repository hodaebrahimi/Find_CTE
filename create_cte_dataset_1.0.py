#!/usr/bin/env python3
"""
CTE Cases File Organizer

This script reads CTE case CSV files and copies the corresponding scan files
to an organized directory structure with sequential IBD_#### naming.

Author: Hoda
"""

import pandas as pd
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directory_structure(base_output_dir: Path) -> None:
    """
    Create the organized directory structure.
    
    Args:
        base_output_dir: Base output directory path
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each patient type
    (base_output_dir / 'cd').mkdir(exist_ok=True)
    (base_output_dir / 'uc').mkdir(exist_ok=True)
    (base_output_dir / 'ibdu').mkdir(exist_ok=True)
    
    logger.info(f"Created directory structure at: {base_output_dir}")


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
                        
                        # NEW: Comprehensive classification fields
                        'Is_CTE': row.get('Is_CTE', False),
                        'Has_Enterography': row.get('Has_Enterography', False),
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
                        
                        # NEW: Comprehensive classification fields
                        'Is_CTE': row.get('Is_CTE', False),
                        'Has_Enterography': row.get('Has_Enterography', False),
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
    """
    Main function to organize CTE case files.
    """
    # Configuration - UPDATED to use new CSV filenames
    csv_files = {
        'uc': Path('/data/ailab/hoda2/ibd_data_arrangement/cte_uc_data.csv'),
        'cd': Path('/data/ailab/hoda2/ibd_data_arrangement/cte_cd_data.csv'),
        'ibdu': Path('/data/ailab/hoda2/ibd_data_arrangement/cte_ibdu_data.csv')
    }
    
    base_data_dir = Path('/data/ibd/data/patients')
    output_dir = Path('/data/ailab/hoda2/ibd_data_arrangement/CTE_cases')
    
    # Settings
    combined_numbering = True  # Set to False for separate numbering per patient type
    
    logger.info("Starting CTE cases file organization...")
    logger.info(f"Source data directory: {base_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Combined numbering: {combined_numbering}")
    
    # Verify CSV files exist
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
    
    # Verify source data directory exists
    if not base_data_dir.exists():
        logger.error(f"Source data directory not found: {base_data_dir}")
        return
    
    # Create directory structure
    create_directory_structure(output_dir)
    
    # Copy and rename files
    mapping_records = copy_and_rename_files(
        csv_files=existing_csv_files,
        base_data_dir=base_data_dir,
        output_dir=output_dir,
        combined_numbering=combined_numbering
    )
    
    # Save mapping file
    save_mapping_file(mapping_records, output_dir)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FILE ORGANIZATION SUMMARY")
    logger.info("="*60)
    
    if mapping_records:
        uc_count = len([r for r in mapping_records if r['Patient_Type'] == 'UC'])
        cd_count = len([r for r in mapping_records if r['Patient_Type'] == 'CD'])
        ibdu_count = len([r for r in mapping_records if r['Patient_Type'] == 'IBDU'])
        
        logger.info(f"UC cases organized: {uc_count}")
        logger.info(f"CD cases organized: {cd_count}")
        logger.info(f"IBDU cases organized: {ibdu_count}")
        logger.info(f"Total cases organized: {len(mapping_records)}")
        
        # Phase distribution
        single_phase = len([r for r in mapping_records if 'Single' in str(r.get('Phase_Classification', ''))])
        multi_phase = len([r for r in mapping_records if 'Multi' in str(r.get('Phase_Classification', ''))])
        
        logger.info(f"\nPhase Distribution:")
        logger.info(f"  Single-Phase: {single_phase}")
        logger.info(f"  Multi-Phase: {multi_phase}")
        
        # Show directory structure
        logger.info(f"\nOrganized structure:")
        logger.info(f"{output_dir}/")
        logger.info(f"├── cd/     ({cd_count} cases)")
        logger.info(f"├── uc/     ({uc_count} cases)")
        logger.info(f"├── ibdu/   ({ibdu_count} cases)")
        logger.info(f"└── IBD_file_mapping.csv")
        
        # Show sample mappings
        logger.info(f"\nSample file mappings:")
        for i, record in enumerate(mapping_records[:3]):
            phase = record.get('Phase_Type', 'Unknown')
            logger.info(f"  {record['Original_File']} -> {record['IBD_ID']}.nii.gz ({phase})")
        
        if len(mapping_records) > 3:
            logger.info(f"  ... and {len(mapping_records) - 3} more")
    
    logger.info("\nFile organization complete!")


if __name__ == "__main__":
    main()