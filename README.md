# IBD CT / CTE Dataset Arrangement

Tools to **scan an uncleaned IBD imaging dataset**, identify **CT** and
**CT Enterography (CTE)** studies from **NIfTI + JSON sidecars**, and **export**
organized folders for downstream ML pipelines.

Outputs:

| Folder | Contents |
|--------|----------|
| `CT_all/` | Every CT scan (cd / uc / ibdu) |
| `CTE_only/` | CTE subset (optionally includes UNCERTAIN) |
| `CTEs_cd_patients/` | Quick CD-only CTE collection for server transfer |

> **Important:** this repo publishes **code only** — no NIfTI, no DICOM, no patient metadata.

---

## Repository structure

```text
.
├── list_ct_data.py                          # Step 1: scan & classify
├── create_cte_dataset_2.0.py                # Step 2: organize CT + CTE
├── cd_cte_collection.py                     # Step 3 (optional): collect CD CTEs
├── create_cte_dataset_1.0.py                # (legacy v1 organizer)
├── extract_ct_and_cte_from_combined_csv.py  # (redundant — kept for reference)
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
└── .gitignore
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

---

## Data and privacy

- Do **not** commit imaging data or generated CSVs to GitHub.
- The [.gitignore](.gitignore) is configured to exclude these by default.
- Before publishing, double-check that no PHI/PII is present in tracked files.

---

## Data assumptions

Raw dataset folder structure:

```text
/data/ibd/data/patients/
  {patient_type}_{patient_id}_{year}/
    {accession_deid}_{date}/
      *.nii.gz        (NIfTI images)
      *.json           (dcm2niix JSON sidecars, when available)
```

`patient_type` is one of `cd`, `uc`, or `ibdu`.

---

## Step-by-step workflow

> **TL;DR:** run Step 1, then Step 2 or 3 depending on what you need.

### Step 1 — Scan the dataset and build CSVs

**Script:** `list_ct_data.py` — always run **first**.

Walks every patient folder, reads NIfTI headers + JSON sidecars, computes a
CTE score for each series, and writes CSVs.

```bash
python list_ct_data.py \
  --root-dir /data/ibd/data/patients \
  --output-dir . \
  --process-all-series
```

**Outputs** (written to `--output-dir`):

| File | Description |
|------|-------------|
| `ct_all_scans_combined.csv` | Every CT series found (with CTE scoring columns) |
| `cte_all_data.csv` | CTE subset (CTE_Score >= threshold) |
| `cte_cd_data.csv` | CD-only CTEs |
| `cte_uc_data.csv` | UC-only CTEs |
| `cte_ibdu_data.csv` | IBDU-only CTEs |
| `ct_cd_all_scans_combined.csv` | All CD CT scans |
| `ct_uc_all_scans_combined.csv` | All UC CT scans |
| `ct_ibdu_all_scans_combined.csv` | All IBDU CT scans |

**Key columns produced:**

| Column | Meaning |
|--------|---------|
| `CTE_Score` | Numerical confidence (higher = more likely CTE) |
| `CTE_Label` | `CTE`, `UNCERTAIN`, or `NOT_CTE` |
| `CTE_Score_Reasons` | Human-readable explanation of the score |
| `Has_Contrast` | IV contrast detected in metadata |
| `Has_Enterography` | Enterography keywords found |
| `Has_Valid_Anatomy` | Abdomen/pelvis body part confirmed |
| `NIfTI_Full_Path` | Absolute path to the NIfTI file |
| `JSON_Full_Path` | Absolute path to the JSON sidecar |

#### Tuning recall vs precision

To catch more borderline CTEs, lower the threshold:

```bash
python list_ct_data.py \
  --process-all-series \
  --cte-threshold 2.5 \
  --uncertain-threshold 2.0
```

> **Tip:** `cte_all_data.csv` only includes rows labelled `CTE`.
> To see `UNCERTAIN` rows, filter `ct_all_scans_combined.csv` where
> `CTE_Label == UNCERTAIN`.

#### Excluding derived files

Files whose names contain `seg`, `mask`, `label`, etc. are skipped by default.
Override with:

```bash
python list_ct_data.py --exclude-substrings seg mask label
```

---

### Step 2 — Export organized CT + CTE folders

**Script:** `create_cte_dataset_2.0.py` — run **after** Step 1.

Reads the combined CSV and copies (or symlinks) NIfTI + JSON files into clean folders.

```bash
python create_cte_dataset_2.0.py from-combined \
  --combined-csv ct_all_scans_combined.csv \
  --base-data-dir /data/ibd/data/patients \
  --output-dir organized_CT_and_CTE \
  --mode copy \
  --include-uncertain
```

**Output:**

```text
organized_CT_and_CTE/
├── CT_all/
│   ├── cd/       CT_000001.nii.gz, CT_000001.json, ...
│   ├── uc/
│   └── ibdu/
├── CTE_only/
│   ├── cd/       CTE_000001.nii.gz, CTE_000001.json, ...
│   ├── uc/
│   └── ibdu/
├── CT_all_mapping.csv
└── CTE_only_mapping.csv
```

Use `--mode symlink` instead of `--mode copy` if storage is tight.

There is also a **legacy mode** that reads the older per-type CSVs:

```bash
python create_cte_dataset_2.0.py legacy-cte-only
```

---

### Step 3 (optional) — Collect CD CTEs for server transfer

**Script:** `cd_cte_collection.py` — run **after** Step 1.

Copies only the CD CTE scans (from `cte_cd_data.csv`) into a flat folder for
easy transfer to another machine.

```bash
python cd_cte_collection.py
```

**Output:** `CTEs_cd_patients/CTE_CD_000001.nii.gz`, `CTE_CD_000001.json`, ...

Edit `CSV_PATH` and `OUT_DIR` at the top of the script to customize.

---

### Step 4 — Verify outputs

```bash
# Organized folders
find organized_CT_and_CTE/CT_all  -name "*.nii*" | wc -l
find organized_CT_and_CTE/CTE_only -name "*.nii*" | wc -l

# CD collection
ls CTEs_cd_patients/*.nii* 2>/dev/null | wc -l
```

---

## CTE scoring explained

The scoring system uses a **tiered approach** based on JSON sidecar metadata:

| Tier | What it checks | Max points |
|------|---------------|------------|
| **1 — Modality** | `Modality == CT` | 1 |
| **1 — Enterography** | Keywords (`entero`, `cte`, `small bowel`) in SeriesDescription / ProtocolName / StudyDescription | 2 |
| **2 — Contrast** | `ContrastBolusAgent`, `ContrastBolusRoute`, `ImageType` containing `POST_CONTRAST` | 1 |
| **3 — Phase** | Phase keywords (`venous`, `portal`, `enteric`) | 0.5 |
| **4 — Anatomy** | `BodyPartExamined` = abdomen/pelvis, or large z-coverage | 1.5 |

**Thresholds** (configurable via CLI):

- `CTE_Score >= 3.0` -> labelled **CTE** (default)
- `CTE_Score >= 2.0` -> labelled **UNCERTAIN** (default)
- Below -> **NOT_CTE**

---

## Common pitfalls / troubleshooting

| Problem | Fix |
|---------|-----|
| Missing CTEs | Run with `--process-all-series` and lower `--cte-threshold` (e.g. `2.5`) |
| No JSON sidecars | CTE detection is less reliable with only NIfTI headers; consider re-running dcm2niix |
| Slow scan | Use `--only-largest-series` for speed (sacrifices recall in messy folders) |

---

## Optional / legacy scripts

| Script | Status |
|--------|--------|
| `extract_ct_and_cte_from_combined_csv.py` | Redundant — same functionality is in `create_cte_dataset_2.0.py from-combined` |
| `create_cte_dataset_1.0.py` | Legacy v1 organizer — superseded by v2.0 |

Both can be deleted if you prefer fewer scripts.

---

## License / citation

- License: see [LICENSE](LICENSE).
- Citation: see [CITATION.cff](CITATION.cff).
