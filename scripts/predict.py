"""Screen one sample from the command line.

python scripts/predict.py my_scans.csv
Input: an instrument CSV (Pixel,WaveLength,dark,ref,sample) OR a table with one scan per row
and wavelength column headers (e.g. 892.434, 896.159, ...).
"""
import sys, os
import joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paracetamol_nir.io import read_any

m = joblib.load(os.path.join(os.path.dirname(__file__), "..", "models", "paracetamol_blend_screening_model.joblib"))
for f in sys.argv[1:]:
    X, wl, kind = read_any(f)
    r = m.screen(X, wl)
    print(f"{os.path.basename(f)}  [{kind}, {r['n_scans']} scan(s)]")
    print(f"  VERDICT: {r['verdict']}")
    print(f"  indicative label claim: {r['median_label_claim']:.1f} %  (IQR {r['iqr'][0]:.1f}-{r['iqr'][1]:.1f})")
    print(f"  scans in domain: {r['fraction_in_domain']:.0%}   similarity to calibration blends: {r['blend_similarity']:.4f}")
    print("  Screening result only - confirm by the validated HPLC assay before any release decision.")
