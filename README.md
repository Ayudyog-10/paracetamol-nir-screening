# Paracetamol NIR screening (METASPEQ)

Screening tool for **paracetamol + talc powder blends** (80–120 % label claim) scanned on the METASPEQ
892–1710 nm NIR module, plus the consolidated master spectra dataset from 2024–2026.

> **Screening only.** The model sorts samples into LOW / WITHIN RANGE / HIGH bands and flags anything
> unlike its calibration set. It does not replace the validated HPLC assay and must not be used for batch release.

## Performance (5-fold, all scans of a sample held out together)

| Metric | Value |
|---|---|
| R² | 0.83 |
| RMSE | 4.8 % label claim |
| RPD | 2.48 |
| Re-scanned 2026 powders (13) | RMSE 3.5 % label claim |

Bands: LOW < 85 · BORDERLINE 85–95 / 105–115 (confirm by HPLC) · WITHIN RANGE 95–105 · HIGH > 115.
Formulation tablets, pure API, pure talc and the IPRS standard are flagged **out of domain**.
Full details: `docs/Paracetamol_NIR_White_Paper.docx` and `models/model_card.json`.

## Repository layout

```
app/app.py                      Streamlit web app
src/paracetamol_nir/            preprocessing, model class, file readers
scripts/build_master_dataset.py rebuild data/ from the raw "Paracetamol" project folder
scripts/benchmark.py            random-split vs sample-grouped benchmarking
scripts/train.py                train the final model -> models/
scripts/predict.py              command-line screening
scripts/make_figures.py         white-paper figures
scripts/export_master_excel.py  Drive-ready master workbook
data/                           master spectra (parquet), sample register, campaigns, wavelength axis
models/                         trained model + model card
examples/                       files to try in the app
reports/                        figures and result tables
docs/                           white paper
```

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
python scripts/predict.py examples/rescan_label101.5_absorbance.csv
python tests/test_smoke.py
```

Accepted inputs: instrument raw CSV (`Pixel,WaveLength,dark,ref,sample`, one file per scan — upload several),
a CSV/XLSX with one scan per row and wavelength headers, or a Spectrafind export (`Absorbance_1…256`).

## Rebuild everything from the raw folder

```bash
python scripts/build_master_dataset.py /path/to/Paracetamol data
python scripts/train.py data/paracetamol_nir_master.parquet models
python scripts/export_master_excel.py Paracetamol_NIR_Master_Raw_Spectra.xlsx
```

Excluded on purpose: the Sept 2024 formulation campaign (not in the folder) and the 25–28 Mar 2025
low-dose set (formulation undocumented; code C04 is skipped).

## Deploy on Streamlit Community Cloud

1. Push this folder to a repository under the **Ayudyog-10** account (sign in to GitHub as Ayudyog-10, or add your
   personal account as a collaborator first — otherwise the push fails with a 403).
2. At share.streamlit.io, sign in as Ayudyog-10 → *Create app* → pick the repo, branch `main`, main file `app/app.py`.
3. Under *Advanced settings* choose **Python 3.12** (the model was saved with scikit-learn 1.8.0).
4. Keep the app private and invite collaborators with *Share*.

Docker alternative: `docker build -t paracetamol-nir . && docker run -p 8501:8501 paracetamol-nir`

---
METASPEQ · Ayudyog Private Limited
