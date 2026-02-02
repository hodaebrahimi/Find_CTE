#!/usr/bin/env python3
"""
IBD CT Enterography Data Processor with Comprehensive Metadata Search

This script processes IBD patient imaging data to identify and organize
CT Enterography scans using comprehensive search across ALL metadata fields.

Two CSV sets are generated:
1. CTE scans (comprehensive identification across all fields)
2. Combined metadata (all CT scans with phase classification)

Author: Hoda
"""

from pathlib import Path
import argparse
import pandas as pd
import json
import nibabel as nib
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DEFAULT_CTE_THRESHOLD = 3.0
DEFAULT_UNCERTAIN_THRESHOLD = 2.0
DEFAULT_PROCESS_ALL_SERIES = True
DEFAULT_EXCLUDE_NIFTI_NAME_SUBSTRINGS = [
    'seg', 'mask', 'label', 'roi', 'annotation', 'annot', 'pred', 'prediction'
]


def contains_any(text: str, keywords: List[str]) -> bool:
    """
    Check if any keyword appears in text (case-insensitive).
    
    Args:
        text: String to search
        keywords: List of keywords to look for
        
    Returns:
        True if any keyword found
    """
    if not text:
        return False
    text_lower = str(text).lower()
    return any(k.lower() in text_lower for k in keywords)


def calculate_cte_score(json_metadata: Dict, nifti_metadata: Dict) -> Tuple[float, Dict]:
    """
    Calculate robust CTE score using tiered classification approach.
    
    Based on ChatGPT recommendations for recovering CTE from NIfTI+JSON:
    - Tier 1: Modality, Series/Protocol/Study descriptions (2 points)
    - Tier 2: Contrast evidence (1 point)
    - Tier 3: Acquisition phase hints (0.5 points)
    - Tier 4: Anatomy sanity checks (1 point)
    
    Threshold: score >= 3 indicates CTE
    
    Args:
        json_metadata: Dictionary with json_ prefixed fields
        nifti_metadata: Dictionary with nifti_ prefixed fields
        
    Returns:
        Tuple of (score, details_dict)
    """
    score = 0.0
    details = {
        'tier1_modality': False,
        'tier1_enterography': False,
        'tier2_contrast': False,
        'tier3_phase': False,
        'tier4_anatomy': False,
        'reasons': []
    }
    
    # TIER 1: Modality check
    modality = json_metadata.get('json_Modality', nifti_metadata.get('nifti_Modality', ''))
    if str(modality).upper() == 'CT':
        score += 1
        details['tier1_modality'] = True
        details['reasons'].append('Modality=CT')
    
    # TIER 1: Enterography keywords in descriptions
    entero_keywords = ['entero', 'enterography', 'cte', 'small bowel', 'small-bowel', 'sb']
    
    # Check SeriesDescription
    series_desc = json_metadata.get('json_SeriesDescription', '')
    if contains_any(series_desc, entero_keywords):
        score += 2
        details['tier1_enterography'] = True
        details['reasons'].append(f'SeriesDescription contains enterography')
    
    # Check ProtocolName
    protocol_name = json_metadata.get('json_ProtocolName', '')
    if contains_any(protocol_name, entero_keywords):
        if not details['tier1_enterography']:  # Don't double count
            score += 1
        details['tier1_enterography'] = True
        details['reasons'].append(f'ProtocolName contains enterography')
    
    # Check StudyDescription
    study_desc = json_metadata.get('json_StudyDescription', '')
    if contains_any(study_desc, entero_keywords):
        if not details['tier1_enterography']:  # Don't double count
            score += 1
        details['tier1_enterography'] = True
        details['reasons'].append(f'StudyDescription contains enterography')
    
    # Check ProcedureStepDescription
    proc_desc = json_metadata.get('json_ProcedureStepDescription', '')
    if contains_any(proc_desc, entero_keywords):
        if not details['tier1_enterography']:  # Don't double count
            score += 1
        details['tier1_enterography'] = True
        details['reasons'].append(f'ProcedureStepDescription contains enterography')
    
    # TIER 2: Contrast evidence (CRITICAL for CTE)
    has_contrast = False
    
    # Check ContrastBolusAgent
    contrast_agent = json_metadata.get('json_ContrastBolusAgent', '')
    if contrast_agent and contrast_agent != 'N/A':
        has_contrast = True
        details['reasons'].append(f'ContrastBolusAgent={contrast_agent}')
    
    # Check ContrastBolusRoute for IV
    contrast_route = json_metadata.get('json_ContrastBolusRoute', '')
    if contains_any(contrast_route, ['iv', 'intravenous']):
        has_contrast = True
        details['reasons'].append(f'ContrastBolusRoute={contrast_route}')
    
    # Check ImageType for POST_CONTRAST
    image_type = json_metadata.get('json_ImageType', [])
    if isinstance(image_type, list):
        if any('POST_CONTRAST' in str(t).upper() or 'POST CONTRAST' in str(t).upper() for t in image_type):
            has_contrast = True
            details['reasons'].append('ImageType contains POST_CONTRAST')
    elif isinstance(image_type, str):
        if 'POST_CONTRAST' in image_type.upper() or 'POST CONTRAST' in image_type.upper():
            has_contrast = True
            details['reasons'].append('ImageType contains POST_CONTRAST')
    
    if has_contrast:
        score += 1
        details['tier2_contrast'] = True
    
    # TIER 3: Acquisition phase hints
    phase_keywords = ['venous', 'portal', 'enteric', 'portal venous']
    
    if contains_any(series_desc, phase_keywords):
        score += 0.5
        details['tier3_phase'] = True
        details['reasons'].append('Phase keywords in SeriesDescription')
    elif contains_any(protocol_name, phase_keywords):
        score += 0.5
        details['tier3_phase'] = True
        details['reasons'].append('Phase keywords in ProtocolName')
    
    # TIER 4: Anatomy sanity checks
    body_part = json_metadata.get('json_BodyPartExamined', '')
    anatomy_keywords = ['abdomen', 'abdomenpelvis', 'abdpel', 'abd', 'pelvis']
    
    if contains_any(body_part, anatomy_keywords):
        score += 1
        details['tier4_anatomy'] = True
        details['reasons'].append(f'BodyPartExamined={body_part}')
    
    # Also check image dimensions for abdomen coverage (from NIfTI)
    image_shape = nifti_metadata.get('nifti_image_shape', [])
    if len(image_shape) >= 3:
        z_slices = image_shape[2]
        if z_slices > 100:  # Typical abdomen coverage
            if not details['tier4_anatomy']:  # Don't double count
                score += 0.5
            details['tier4_anatomy'] = True
            details['reasons'].append(f'Large z-coverage ({z_slices} slices)')
    
    return int(score * 10) / 10, details  # Round to 1 decimal


def extract_all_nifti_header_fields(nifti_path: Path) -> Dict:
    """
    Dynamically extract ALL available fields from NIfTI header.
    
    This function extracts every possible field from the NIfTI header,
    including standard NIfTI fields, extensions, and any text fields.
    
    Args:
        nifti_path: Path to NIfTI file
        
    Returns:
        Dictionary containing all extracted NIfTI header fields
    """
    all_metadata = {}
    
    try:
        nii_img = nib.load(nifti_path)
        header = nii_img.header
        
        # Extract ALL standard header fields dynamically
        if hasattr(header, 'keys'):
            for key in header.keys():
                try:
                    value = header[key]
                    if isinstance(value, bytes):
                        decoded = value.decode('utf-8', errors='ignore').strip('\x00')
                        if decoded:
                            all_metadata[key] = decoded
                    elif isinstance(value, (int, float, str)):
                        all_metadata[key] = value
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        all_metadata[key] = value
                except Exception as e:
                    logger.debug(f"Could not extract header field {key}: {e}")
        
        # Extract specific known fields
        known_fields = ['descrip', 'aux_file', 'intent_name', 'magic', 'db_name']
        for field in known_fields:
            try:
                value = header.get(field, None)
                if value is not None:
                    if isinstance(value, bytes):
                        decoded = value.decode('utf-8', errors='ignore').strip('\x00')
                        if decoded:
                            all_metadata[field] = decoded
                    elif value:
                        all_metadata[field] = value
            except Exception as e:
                logger.debug(f"Could not extract {field}: {e}")
        
        # Extract from extensions (DICOM metadata from dcm2niix)
        if hasattr(header, 'extensions') and header.extensions:
            extension_count = 0
            for ext in header.extensions:
                try:
                    ext_code = None
                    content = None
                    
                    if hasattr(ext, 'get_code'):
                        ext_code = ext.get_code()
                    elif isinstance(ext, tuple):
                        ext_code = ext[0]
                    
                    if hasattr(ext, 'get_content'):
                        content = ext.get_content()
                    elif isinstance(ext, tuple) and len(ext) > 1:
                        content = ext[1]
                    
                    if content and isinstance(content, bytes):
                        try:
                            text_content = content.decode('utf-8', errors='ignore').strip('\x00')
                            if text_content:
                                json_data = json.loads(text_content)
                                for json_key, json_value in json_data.items():
                                    all_metadata[json_key] = json_value
                                logger.debug(f"Extracted {len(json_data)} fields from extension code {ext_code}")
                        except json.JSONDecodeError:
                            text_content = content.decode('utf-8', errors='ignore').strip('\x00')
                            if text_content and len(text_content) > 5:
                                all_metadata[f'extension_{extension_count}_text'] = text_content
                                if ext_code is not None:
                                    all_metadata[f'extension_{extension_count}_code'] = ext_code
                    
                    extension_count += 1
                except Exception as e:
                    logger.debug(f"Could not parse extension: {e}")
        
        # Try to get any other header attributes dynamically
        if hasattr(header, '__dict__'):
            for attr_name, attr_value in header.__dict__.items():
                if not attr_name.startswith('_') and attr_name not in all_metadata:
                    try:
                        if isinstance(attr_value, bytes):
                            decoded = attr_value.decode('utf-8', errors='ignore').strip('\x00')
                            if decoded:
                                all_metadata[attr_name] = decoded
                        elif isinstance(attr_value, (int, float, str, list, tuple)):
                            all_metadata[attr_name] = attr_value
                    except Exception as e:
                        logger.debug(f"Could not extract attribute {attr_name}: {e}")
        
    except Exception as e:
        logger.warning(f"Error extracting NIfTI header from {nifti_path}: {e}")
    
    return all_metadata


def classify_ct_scan(all_metadata: Dict, json_metadata: Dict, nifti_metadata: Dict,
                     json_modality: Optional[str] = None,
                     cte_threshold: float = DEFAULT_CTE_THRESHOLD,
                     uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD) -> Tuple[bool, Dict]:
    """
    Classify CT scan using robust CTE scoring system.
    
    Uses tiered approach based on ChatGPT recommendations for recovering CTE from NIfTI+JSON:
    - Tier 1: Modality (1 pt) + Enterography keywords in descriptions (1-2 pts)
    - Tier 2: Contrast evidence (1 pt) - CRITICAL for CTE
    - Tier 3: Phase hints (0.5 pts)
    - Tier 4: Anatomy checks (1 pt)
    
    Threshold: score >= cte_threshold indicates CTE
    
    This is more robust than keyword-only matching because:
    1. Requires multiple evidence sources
    2. Weighs contrast presence heavily (CTE must have IV contrast)
    3. Handles multi-site data with varying naming conventions
    4. Better false positive/negative balance
    
    Args:
        all_metadata: Combined metadata dictionary with fields from all sources
        json_metadata: JSON sidecar metadata (json_ prefixed)
        nifti_metadata: NIfTI header metadata (nifti_ prefixed)
        json_modality: Modality from JSON file (preferred source for modality)
        
    Returns:
        Tuple of (is_cte, classification_dict)
    """
    # Calculate robust CTE score
    cte_score, score_details = calculate_cte_score(json_metadata, nifti_metadata)
    
    # Determine if CTE / uncertain
    is_cte = cte_score >= cte_threshold
    if cte_score >= cte_threshold:
        cte_label = 'CTE'
    elif cte_score >= uncertain_threshold:
        cte_label = 'UNCERTAIN'
    else:
        cte_label = 'NOT_CTE'
    
    # Build classification dictionary with both new and legacy fields
    classification = {
        # New scoring fields
        'Is_CTE': is_cte,
        'CTE_Label': cte_label,
        'CTE_Threshold': cte_threshold,
        'Uncertain_Threshold': uncertain_threshold,
        'CTE_Score': cte_score,
        'CTE_Score_Reasons': '; '.join(score_details['reasons']),
        'Has_Enterography': score_details['tier1_enterography'],
        'Has_Contrast': score_details['tier2_contrast'],
        'Has_Valid_Anatomy': score_details['tier4_anatomy'],
        
        # Legacy fields for compatibility
        'Enterography_Found_In': [],
        'Anatomy_Found': [],
        'Anatomy_Found_In': [],
        'Phase_Classification': '',
        'Phase_Type': None,
        'Phase_Keywords_Found': [],
        'Phase_Found_In': [],
        'Scout_Excluded': False,
        'Localizer_Excluded': False
    }
    
    # Check modality - must be CT
    modality = json_modality if json_modality else all_metadata.get('nifti_Modality', '')
    if str(modality).upper() != 'CT':
        return False, classification
    
    # Check for Scout/Localizer exclusions (preserve legacy logic)
    for key, value in all_metadata.items():
        if value is None:
            continue
        value_upper = str(value).upper()
        
        if 'SCOUT' in value_upper:
            classification['Scout_Excluded'] = True
            classification['Is_CTE'] = False
            return False, classification
        
        if 'LOCALIZER' in value_upper:
            classification['Localizer_Excluded'] = True
            classification['Is_CTE'] = False
            return False, classification
    
    # Fill in legacy enterography fields for backward compatibility
    if score_details['tier1_enterography']:
        # Check which fields had enterography keywords
        series_desc = json_metadata.get('json_SeriesDescription', '')
        protocol = json_metadata.get('json_ProtocolName', '')
        study_desc = json_metadata.get('json_StudyDescription', '')
        
        entero_keywords = ['entero', 'enterography', 'cte', 'small bowel', 'sb']
        
        if contains_any(series_desc, entero_keywords):
            classification['Enterography_Found_In'].append('json_SeriesDescription')
        if contains_any(protocol, entero_keywords):
            classification['Enterography_Found_In'].append('json_ProtocolName')
        if contains_any(study_desc, entero_keywords):
            classification['Enterography_Found_In'].append('json_StudyDescription')
    
    # Fill in legacy anatomy fields
    if score_details['tier4_anatomy']:
        body_part = json_metadata.get('json_BodyPartExamined', '')
        if body_part:
            classification['Anatomy_Found'].append('abdomen')
            classification['Anatomy_Found_In'].append(f'json_BodyPartExamined:{body_part}')
    
    # Fill in phase classification
    single_phase_keywords = ['SINGLE PHASE', 'SINGLEPHASE', 'SINGLE-PHASE', 'SINGLE', 'MONOPHASIC']
    multi_phase_keywords = ['MULTI PHASE', 'MULTIPHASE', 'BIPHASIC', 'TRIPHASIC', 
                            'ARTERIAL', 'VENOUS', 'PORTAL', 'DELAYED', 'ENTERIC',
                            'PRE CONTRAST', 'POST CONTRAST']
    
    # Search for phase keywords
    for field_name, field_value in all_metadata.items():
        if field_value is None or field_value == '':
            continue
        
        value_str = str(field_value).upper()
        
        for keyword in single_phase_keywords:
            if keyword in value_str:
                classification['Phase_Keywords_Found'].append(f"SINGLE:{keyword}")
                classification['Phase_Found_In'].append(f"{field_name}:{keyword}")
        
        for keyword in multi_phase_keywords:
            if keyword in value_str:
                classification['Phase_Keywords_Found'].append(f"MULTI:{keyword}")
                classification['Phase_Found_In'].append(f"{field_name}:{keyword}")
    
    # Determine phase classification
    has_single = any('SINGLE:' in k for k in classification['Phase_Keywords_Found'])
    has_multi = any('MULTI:' in k for k in classification['Phase_Keywords_Found'])
    
    if has_single:
        classification['Phase_Classification'] = 'Single-Phase'
    else:
        classification['Phase_Classification'] = 'Multi-Phase (assumed)'
    
    return is_cte, classification


def extract_separated_metadata(nifti_path: Path, json_path: Optional[Path], 
                               folder_name: str, patient_id: str) -> Tuple[Dict, Dict, Dict]:
    """
    Extract metadata from NIfTI header and JSON sidecar SEPARATELY.
    Dynamically extracts ALL available fields from NIfTI header.
    
    Returns three dictionaries:
    1. Base metadata (patient info, file info)
    2. NIfTI header metadata (with nifti_ prefix) - ALL fields dynamically extracted
    3. JSON sidecar metadata (with json_ prefix)
    
    Args:
        nifti_path: Path to NIfTI file
        json_path: Path to JSON sidecar file (optional)
        folder_name: Date folder name for extracting deid/date
        patient_id: Patient ID from folder structure
        
    Returns:
        Tuple of (base_metadata, nifti_metadata, json_metadata)
    """
    deid, date = folder_name.split('_') if '_' in folder_name else ('', '')
    
    # Base metadata (source-agnostic)
    base_metadata = {
        'patid': patient_id,
        'accession_deid': deid,
        'SVC_DT': date,
        'has_nifti_header_metadata': False,
        'has_json_sidecar': False
    }
    
    # NIfTI header metadata (all prefixed with nifti_)
    nifti_metadata = {}
    
    # Extract from NIfTI header - DYNAMICALLY extract ALL fields
    try:
        nii_img = nib.load(nifti_path)
        header = nii_img.header
        
        # Standard NIfTI structural fields
        nifti_metadata.update({
            'nifti_pixdim': header.get('pixdim')[:4].tolist() if 'pixdim' in header else None,
            'nifti_voxel_size': list(header.get_zooms()[:3]) if hasattr(header, 'get_zooms') else None,
            'nifti_image_shape': list(nii_img.shape),
            'nifti_data_type': str(header.get_data_dtype()),
            'nifti_qform_code': int(header['qform_code']) if 'qform_code' in header else None,
            'nifti_sform_code': int(header['sform_code']) if 'sform_code' in header else None
        })
        
        # Extract slice thickness from voxel size
        if hasattr(header, 'get_zooms'):
            zooms = header.get_zooms()
            if len(zooms) >= 3:
                nifti_metadata['nifti_SliceThickness'] = float(zooms[2])
        
        # DYNAMIC extraction - get ALL available fields
        all_header_fields = extract_all_nifti_header_fields(nifti_path)
        if all_header_fields:
            base_metadata['has_nifti_header_metadata'] = True
            
            # Add all dynamically extracted fields with nifti_ prefix
            for key, value in all_header_fields.items():
                prefixed_key = f'nifti_{key}'
                if prefixed_key not in nifti_metadata:
                    nifti_metadata[prefixed_key] = value
        
    except Exception as e:
        logger.warning(f"Error reading NIfTI file {nifti_path}: {e}")
    
    # JSON sidecar metadata (all prefixed with json_)
    json_metadata = {}
    
    # Extract from JSON sidecar
    if json_path and json_path.exists():
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            base_metadata['has_json_sidecar'] = True
            
            # Add all JSON fields with json_ prefix
            for key, value in json_data.items():
                json_metadata[f'json_{key}'] = value
                
        except Exception as e:
            logger.warning(f"Error reading JSON file {json_path}: {e}")
    
    return base_metadata, nifti_metadata, json_metadata


def extract_patient_info(folder_name: str) -> Tuple[str, str, str]:
    """Extract patient information from folder name."""
    parts = folder_name.split('_')
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    else:
        raise ValueError(f"Invalid folder format: {folder_name}")


def _is_excluded_nifti(nifti_path: Path, exclude_name_substrings: List[str]) -> bool:
    name = nifti_path.name.lower()
    return any(substr.lower() in name for substr in exclude_name_substrings)


def get_nifti_candidates(
    date_folder: Path,
    process_all_series: bool = DEFAULT_PROCESS_ALL_SERIES,
    exclude_name_substrings: Optional[List[str]] = None
) -> List[Tuple[Path, Optional[Path]]]:
    """Return candidate NIfTI files (and their JSON sidecars) from a date folder.

    Key change vs old logic: by default we process *all* NIfTI files, not just the
    largest, to avoid missing the true CTE series in "uncleaned" folders.
    """
    exclude_name_substrings = exclude_name_substrings or DEFAULT_EXCLUDE_NIFTI_NAME_SUBSTRINGS

    nifti_files: List[Path] = []
    for pattern in ('*.nii.gz', '*.nii'):
        nifti_files.extend([p for p in date_folder.glob(pattern) if p.is_file()])

    nifti_files = [p for p in nifti_files if not _is_excluded_nifti(p, exclude_name_substrings)]
    if not nifti_files:
        return []

    nifti_files.sort(key=lambda f: f.stat().st_size, reverse=True)
    if not process_all_series:
        nifti_files = [nifti_files[0]]

    candidates: List[Tuple[Path, Optional[Path]]] = []
    for nifti_path in nifti_files:
        if nifti_path.name.endswith('.nii.gz'):
            json_name = nifti_path.name.replace('.nii.gz', '.json')
        else:
            json_name = nifti_path.with_suffix('.json').name

        json_path = nifti_path.parent / json_name
        candidates.append((nifti_path, json_path if json_path.exists() else None))

    return candidates


def process_patient_folder(
    patient_folder: Path,
    process_all_series: bool = DEFAULT_PROCESS_ALL_SERIES,
    exclude_name_substrings: Optional[List[str]] = None,
    cte_threshold: float = DEFAULT_CTE_THRESHOLD,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process patient folder with comprehensive metadata search.
    
    Returns:
    1. All CT scans with combined metadata
    2. CTE scans identified using comprehensive search
    
    Args:
        patient_folder: Path to patient folder
        
    Returns:
        Tuple of (all_ct_records, cte_records)
    """
    try:
        patient_type, patient_id, year = extract_patient_info(patient_folder.name)
    except ValueError as e:
        logger.warning(f"Skipping folder {patient_folder.name}: {e}")
        return [], []
    
    all_ct_records = []
    cte_records = []
    
    for date_folder in patient_folder.glob('*_*'):
        if not date_folder.is_dir():
            continue
            
        try:
            candidates = get_nifti_candidates(
                date_folder,
                process_all_series=process_all_series,
                exclude_name_substrings=exclude_name_substrings
            )
            if not candidates:
                continue

            for nifti_path, json_file in candidates:
                # Extract SEPARATED metadata
                base_metadata, nifti_metadata, json_metadata = extract_separated_metadata(
                    nifti_path, json_file, date_folder.name, patient_id
                )

                # Determine modality (prefer JSON, fallback to NIfTI)
                modality = None
                if 'json_Modality' in json_metadata:
                    modality = json_metadata['json_Modality']
                elif 'nifti_Modality' in nifti_metadata:
                    modality = nifti_metadata['nifti_Modality']

                if not modality or str(modality).upper() != 'CT':
                    logger.debug(f"Skipping non-CT scan: {nifti_path.name} (modality: {modality})")
                    continue

                # Prefer metadata-derived date; avoid brittle folder parsing
                date_value = base_metadata.get('SVC_DT', '')

                # Create combined record with ALL metadata
                combined_record = {
                    **base_metadata,
                    **nifti_metadata,
                    **json_metadata,
                    'Patient_Type': patient_type,
                    'Patient_ID': str(patient_id),
                    'Year': year,
                    'Date': date_value,
                    'CT_Scan_File': nifti_path.name,
                    'File_Size_MB': round(nifti_path.stat().st_size / 1024**2, 2),
                    'Date_Folder': date_folder.name,
                    'Date_Folder_Path': str(date_folder),
                    'NIfTI_Full_Path': str(nifti_path),
                    'JSON_Full_Path': str(json_file) if json_file else ''
                }

                # Classify the scan using ALL metadata (both JSON and NIfTI)
                all_combined_metadata = {**nifti_metadata, **json_metadata}
                json_modality = json_metadata.get('json_Modality') if json_metadata else None

                is_cte, classification_info = classify_ct_scan(
                    all_combined_metadata,
                    json_metadata,
                    nifti_metadata,
                    json_modality,
                    cte_threshold=cte_threshold,
                    uncertain_threshold=uncertain_threshold
                )

                combined_record.update(classification_info)
                all_ct_records.append(combined_record)

                if is_cte:
                    cte_record = {
                        **combined_record,
                        'identification_source': 'CTE_Score_Threshold'
                    }
                    cte_records.append(cte_record)
                    logger.debug(f"CTE identified: {nifti_path.name}")
        
        except Exception as e:
            logger.warning(f"Error processing {date_folder}: {e}")
            continue
    
    return all_ct_records, cte_records


def save_datasets(all_ct_records: List[Dict], 
                 cte_records: List[Dict],
                 output_dir: Path):
    """
    Save two separate sets of datasets:
    1. CTE scans (comprehensive search across all fields)
    2. Combined metadata (all CT scans with phase classification)
    
    Args:
        all_ct_records: All CT scan records
        cte_records: CTE records identified using comprehensive search
        output_dir: Output directory
    """
    
    # ===== SET 1: CTE scans (comprehensive identification) =====
    logger.info("\n" + "="*80)
    logger.info("SET 1: CTE SCANS (COMPREHENSIVE SEARCH)")
    logger.info("="*80)
    
    if cte_records:
        logger.info(f"Found {len(cte_records)} CTE scans")
        
        df_cte = pd.DataFrame(cte_records)
        df_cte.to_csv(output_dir / 'cte_all_data.csv', index=False)
        logger.info(f"CTE scans saved to: cte_all_data.csv")
        
        # Separate by patient type
        uc_cte = [r for r in cte_records if r['Patient_Type'] == 'UC']
        cd_cte = [r for r in cte_records if r['Patient_Type'] == 'CD']
        ibdu_cte = [r for r in cte_records if r['Patient_Type'] == 'IBDU']
        
        if uc_cte:
            pd.DataFrame(uc_cte).to_csv(output_dir / 'cte_uc_data.csv', index=False)
            logger.info(f"UC CTE scans saved ({len(uc_cte)} records)")
        if cd_cte:
            pd.DataFrame(cd_cte).to_csv(output_dir / 'cte_cd_data.csv', index=False)
            logger.info(f"CD CTE scans saved ({len(cd_cte)} records)")
        if ibdu_cte:
            pd.DataFrame(ibdu_cte).to_csv(output_dir / 'cte_ibdu_data.csv', index=False)
            logger.info(f"IBDU CTE scans saved ({len(ibdu_cte)} records)")
    
    # ===== SET 2: Combined metadata (all CT scans) =====
    logger.info("\n" + "="*80)
    logger.info("SET 2: ALL CT SCANS WITH COMBINED METADATA")
    logger.info("="*80)
    
    if all_ct_records:
        logger.info(f"Found {len(all_ct_records)} total CT scans")
        
        df_combined = pd.DataFrame(all_ct_records)
        df_combined.to_csv(output_dir / 'ct_all_scans_combined.csv', index=False)
        logger.info(f"Combined CT scans saved to: ct_all_scans_combined.csv")
        
        # Separate by patient type
        uc_combined = [r for r in all_ct_records if r['Patient_Type'] == 'UC']
        cd_combined = [r for r in all_ct_records if r['Patient_Type'] == 'CD']
        ibdu_combined = [r for r in all_ct_records if r['Patient_Type'] == 'IBDU']
        
        if uc_combined:
            pd.DataFrame(uc_combined).to_csv(output_dir / 'ct_uc_all_scans_combined.csv', index=False)
            logger.info(f"UC combined CT scans saved ({len(uc_combined)} records)")
        if cd_combined:
            pd.DataFrame(cd_combined).to_csv(output_dir / 'ct_cd_all_scans_combined.csv', index=False)
            logger.info(f"CD combined CT scans saved ({len(cd_combined)} records)")
        if ibdu_combined:
            pd.DataFrame(ibdu_combined).to_csv(output_dir / 'ct_ibdu_all_scans_combined.csv', index=False)
            logger.info(f"IBDU combined CT scans saved ({len(ibdu_combined)} records)")


def main():
    """Main function to process IBD patient data with comprehensive metadata search."""
    parser = argparse.ArgumentParser(description='Scan uncleaned dataset and extract CT + CTE metadata.')
    parser.add_argument('--root-dir', type=Path, default=Path('/data/ibd/data/patients'))
    parser.add_argument('--output-dir', type=Path, default=Path('.'))
    parser.add_argument('--process-all-series', action='store_true', default=DEFAULT_PROCESS_ALL_SERIES,
                        help='Process all NIfTI series per date folder (recommended for uncleaned data).')
    parser.add_argument('--only-largest-series', action='store_true',
                        help='Backwards-compatible behavior: only process largest NIfTI per folder.')
    parser.add_argument('--cte-threshold', type=float, default=DEFAULT_CTE_THRESHOLD)
    parser.add_argument('--uncertain-threshold', type=float, default=DEFAULT_UNCERTAIN_THRESHOLD)
    parser.add_argument('--exclude-substrings', nargs='*', default=DEFAULT_EXCLUDE_NIFTI_NAME_SUBSTRINGS,
                        help='Exclude NIfTI files whose names contain any of these substrings.')
    args = parser.parse_args()

    root_dir = args.root_dir
    output_dir = args.output_dir
    process_all_series = args.process_all_series and (not args.only_largest_series)
    
    logger.info(f"Starting IBD CT data processing with comprehensive metadata search...")
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Output: 2 CSV sets (CTE and Combined)")
    
    if not root_dir.exists():
        logger.error(f"Root directory does not exist: {root_dir}")
        return
    
    # Get all patient folders
    patient_folders = [f for f in root_dir.glob('*_*_*') if f.is_dir()]
    logger.info(f"Found {len(patient_folders)} patient folders to process")
    
    # Process all folders
    all_ct_records = []
    cte_records = []
    
    for patient_folder in tqdm(patient_folders, desc="Processing patient folders"):
        ct_recs, cte_recs = process_patient_folder(
            patient_folder,
            process_all_series=process_all_series,
            exclude_name_substrings=args.exclude_substrings,
            cte_threshold=args.cte_threshold,
            uncertain_threshold=args.uncertain_threshold
        )
        all_ct_records.extend(ct_recs)
        cte_records.extend(cte_recs)
    
    # Save all datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    save_datasets(all_ct_records, cte_records, output_dir)
    
    # ===== SUMMARY =====
    logger.info("\n" + "="*80)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total CT scans processed: {len(all_ct_records)}")
    logger.info(f"CTE scans identified: {len(cte_records)}")
    
    # CTE Score distribution
    if cte_records:
        cte_scores = [r.get('CTE_Score', 0) for r in cte_records]
        logger.info(f"\nCTE Score Distribution:")
        logger.info(f"  Mean: {sum(cte_scores)/len(cte_scores):.2f}")
        logger.info(f"  Min: {min(cte_scores):.1f}, Max: {max(cte_scores):.1f}")
        
        # Show score breakdown
        score_ranges = {
            '3.0-3.9': len([s for s in cte_scores if 3.0 <= s < 4.0]),
            '4.0-4.9': len([s for s in cte_scores if 4.0 <= s < 5.0]),
            '5.0+': len([s for s in cte_scores if s >= 5.0])
        }
        logger.info(f"  Score Ranges:")
        for range_name, count in score_ranges.items():
            logger.info(f"    {range_name}: {count} scans")
        
        # Contrast statistics
        with_contrast = len([r for r in cte_records if r.get('Has_Contrast', False)])
        logger.info(f"\nContrast Detection:")
        logger.info(f"  With contrast: {with_contrast}/{len(cte_records)} ({with_contrast/len(cte_records)*100:.1f}%)")
    
    # Phase classification summary
    if all_ct_records:
        single_phase = len([r for r in all_ct_records if r.get('Phase_Classification', '').startswith('Single')])
        multi_phase = len([r for r in all_ct_records if r.get('Phase_Classification') == 'Multi-Phase'])
        
        logger.info(f"\nPhase Classification (all CT scans):")
        logger.info(f"  Single-Phase: {single_phase} ({single_phase/len(all_ct_records)*100:.1f}%)")
        logger.info(f"  Multi-Phase: {multi_phase} ({multi_phase/len(all_ct_records)*100:.1f}%)")
    
    # CTE identification analysis
    if cte_records:
        has_entero = len([r for r in cte_records if r.get('Has_Enterography')])
        has_anatomy = len([r for r in cte_records if r.get('Has_Valid_Anatomy')])
        
        logger.info(f"\nCTE Identification Criteria:")
        logger.info(f"  Has 'enterography' keyword: {has_entero}")
        logger.info(f"  Has valid anatomy: {has_anatomy}")
        
        # Anatomy distribution
        anatomy_counts = {}
        for record in cte_records:
            for anatomy in record.get('Anatomy_Found', []):
                anatomy_counts[anatomy] = anatomy_counts.get(anatomy, 0) + 1
        
        if anatomy_counts:
            logger.info(f"\n  Anatomy Distribution (CTE scans):")
            for anatomy, count in sorted(anatomy_counts.items()):
                logger.info(f"    {anatomy}: {count}")
    
    # Metadata source analysis
    if all_ct_records:
        logger.info("\n" + "="*80)
        logger.info("METADATA AVAILABILITY")
        logger.info("="*80)
        
        has_nifti = len([r for r in all_ct_records if r.get('has_nifti_header_metadata')])
        has_json = len([r for r in all_ct_records if r.get('has_json_sidecar')])
        has_both = len([r for r in all_ct_records if r.get('has_nifti_header_metadata') and r.get('has_json_sidecar')])
        
        logger.info(f"Scans with NIfTI header metadata: {has_nifti} ({has_nifti/len(all_ct_records)*100:.1f}%)")
        logger.info(f"Scans with JSON sidecar: {has_json} ({has_json/len(all_ct_records)*100:.1f}%)")
        logger.info(f"Scans with both sources: {has_both} ({has_both/len(all_ct_records)*100:.1f}%)")
        
        # Show sample of what NIfTI fields were found
        logger.info("\n" + "="*80)
        logger.info("SAMPLE NIFTI HEADER FIELDS (from first record)")
        logger.info("="*80)
        sample = all_ct_records[0]
        nifti_fields = [k for k in sample.keys() if k.startswith('nifti_')]
        logger.info(f"Total NIfTI fields extracted: {len(nifti_fields)}")
        logger.info("Field names:")
        for field in sorted(nifti_fields)[:20]:
            value = sample.get(field)
            if isinstance(value, (list, tuple)) and len(str(value)) > 100:
                value_str = str(value)[:100] + "..."
            else:
                value_str = str(value)
            logger.info(f"  {field}: {value_str}")
        if len(nifti_fields) > 20:
            logger.info(f"  ... and {len(nifti_fields) - 20} more fields")
    
    logger.info("\nProcessing complete!")
    logger.info(f"\nGenerated CSV sets:")
    logger.info(f"  1. CTE scans: cte_*.csv (comprehensive search across all fields)")
    logger.info(f"  2. All CT scans: ct_*_combined.csv (with phase classification)")


if __name__ == "__main__":
    main()