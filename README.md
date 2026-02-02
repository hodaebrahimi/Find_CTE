# IBD CT / CTE dataset arrangement

This folder contains scripts to **scan an uncleaned imaging dataset**, identify **CT** and **CT Enterography (CTE)** studies using **NIfTI + JSON sidecars**, and then **export** organized folders containing:

- all CT scans (`CT_all/`)
- CTE subset (`CTE_only/`)

The intended use is to maximize **CTE recall** (don’t miss true CTEs), while still keeping traceability for review.

Important: this repo is meant to publish **code only** (no NIfTI, no DICOM, no patient metadata exports).

---

## Requirements

Python packages:

```bash
pip install -r requirements.txt
```

Recommended:

- Python 3.9+
- Use a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data and privacy

- Do **not** commit imaging data (`*.nii*`, `*.dcm`) or generated CSVs to GitHub.
- This repo’s [.gitignore](.gitignore) is configured to ignore common outputs and imaging formats.
- Before publishing, double-check that no PHI/PII is present in tracked files.

---

## Data assumptions

Your raw dataset is expected to follow this folder structure (or similar):

```
/data/ibd/data/patients/
  {patient_type}_{patient_id}_{year}/
    {accession_deid}_{date}/
      *.nii.gz  (and possibly *.nii)
      *.json    (dcm2niix sidecars, when available)
```

Where `patient_type` is typically one of `cd`, `uc`, `ibdu`.

---

## Quick start (recommended workflow)

### Step 1 — scan the uncleaned dataset and build CSVs

Run the scanner:

```bash
python list_ct_data.py \
  --root-dir /data/ibd/data/patients \
  --output-dir . \
  --process-all-series
```

This generates (in `--output-dir`):

- `ct_all_scans_combined.csv` (all CT series found)
- `cte_all_data.csv` (CTE subset where `CTE_Score >= --cte-threshold`)
- `cte_uc_data.csv`, `cte_cd_data.csv`, `cte_ibdu_data.csv`
- `ct_uc_all_scans_combined.csv`, `ct_cd_all_scans_combined.csv`, `ct_ibdu_all_scans_combined.csv`

#### Tune recall vs precision

If you want to **catch more borderline CTEs**, lower the threshold and/or keep an “uncertain” bucket:

```bash
python list_ct_data.py \
  --process-all-series \
  --cte-threshold 2.5 \
  --uncertain-threshold 2.0
```

Notes:
- `CTE_Label` is `CTE`, `UNCERTAIN`, or `NOT_CTE`.
- `CTE_Score_Reasons` tells you *why* a scan scored.

Tip: `cte_all_data.csv` includes only `CTE` (score >= threshold). If you want the `UNCERTAIN` rows, use `ct_all_scans_combined.csv` and filter by `CTE_Label == UNCERTAIN`.

#### Excluding derived files

By default, files with names containing `seg`, `mask`, `label`, etc. are skipped.
You can override that list:

```bash
python list_ct_data.py --exclude-substrings seg mask label
```

---

### Step 2 — export organized folders for CT and CTE

Use the organizer (preferred mode is `from-combined`):

```bash
python create_cte_dataset_2.0.py from-combined \
  --combined-csv ct_all_scans_combined.csv \
  --base-data-dir /data/ibd/data/patients \
  --output-dir organized_CT_and_CTE \
  --mode copy \
  --include-uncertain
```

Output:

```
organized_CT_and_CTE/
  CT_all/
    cd/
    uc/
    ibdu/
  CTE_only/
    cd/
    uc/
    ibdu/
  CT_all_mapping.csv
  CTE_only_mapping.csv
```

If storage is tight, use symlinks instead of copies:

```bash
python create_cte_dataset_2.0.py from-combined \
  --output-dir organized_CT_and_CTE \
  --mode symlink \
  --include-uncertain
```

---

## What each script does

### `list_ct_data.py`

Purpose:
- scans the raw patient folders
- processes **all NIfTI series per date folder** (recommended for “uncleaned” data)
- extracts metadata from:
  - NIfTI headers
  - JSON sidecars (best source of DICOM-derived tags)
- computes `CTE_Score` and assigns `CTE_Label`

Important output columns:
- `CTE_Score`, `CTE_Label`, `CTE_Score_Reasons`
- `Has_Contrast`, `Has_Enterography`, `Has_Valid_Anatomy`
- `NIfTI_Full_Path`, `JSON_Full_Path` (used by the organizer)

### `create_cte_dataset_2.0.py`

Purpose:
- organizes files into `CT_all/` and `CTE_only/` based on the combined CSV
- copies/symlinks both the NIfTI and JSON sidecar when available

Modes:
- `from-combined` (recommended): uses `ct_all_scans_combined.csv`
- `legacy-cte-only`: uses the older `cte_uc_data.csv`, `cte_cd_data.csv`, `cte_ibdu_data.csv`

---

## Common pitfalls / troubleshooting

### 1) “I’m missing CTEs”

Try:
- make sure you ran with `--process-all-series`
- lower `--cte-threshold` (e.g. `2.5`) and include uncertain in export

### 2) “No JSON sidecars”

If the conversion didn’t produce JSON files, your ability to detect CTE drops.
The code will still run using NIfTI headers, but CTE detection will be less reliable.

### 3) Slow run

Scanning every series is slower but improves recall. If you need speed:

```bash
python list_ct_data.py --only-largest-series
```

(Expect more missed CTEs in messy folders.)

---

## Optional / legacy scripts

- `extract_ct_and_cte_from_combined_csv.py` is now redundant with `create_cte_dataset_2.0.py from-combined`.
  You can delete it if you prefer fewer scripts.

---

## Publishing to GitHub (checklist)

1) Confirm that outputs are ignored

- `*.csv`, imaging files, and dataset folders should be ignored by default.
- If you previously ran `git add` on outputs, untrack them before committing:

```bash
git rm -r --cached .
git add .
```

2) Initialize git

```bash
git init
git add .
git status
```

3) Create a GitHub repo and push

Option A (GitHub CLI):

```bash
gh repo create <your-repo-name> --public --source . --remote origin --push
```

Option B (manual remote):

```bash
git remote add origin git@github.com:<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

4) Add a release (optional)

- Use GitHub “Releases” to tag versions (e.g., `v2.0.0`).

---

## License / citation

- License: see [LICENSE](LICENSE).
- Citation: see [CITATION.cff](CITATION.cff).
