"""Train the final paracetamol/talc blend screening model from the master spectra file.

python scripts/train.py data/paracetamol_nir_master.parquet models/
"""
import sys, os, json, datetime
import numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from paracetamol_nir.model import BlendScreeningModel, LOW_CUT, HIGH_CUT, BORDER
from paracetamol_nir import __version__

src = sys.argv[1] if len(sys.argv) > 1 else "data/paracetamol_nir_master.parquet"
out = sys.argv[2] if len(sys.argv) > 2 else "models"
A = pd.read_parquet(src)
ac = [c for c in A if c.startswith("A_")]
wl = np.array([float(c[2:]) for c in ac])
c5 = A[A.campaign.str.startswith("C05")]
train = pd.concat([A[A.campaign.str.startswith("C09")], c5[c5.role == "calibration"], A[A.campaign.str.startswith("C07")]])
api = pd.concat([c5[c5.api_pct_ww == 100], A[A.campaign.str.startswith("C08")]])[ac].values
talc = c5[c5.api_pct_ww == 0][ac].values
m = BlendScreeningModel().fit(train[ac].values, train.label_claim_pct.values, wl, api, talc)
os.makedirs(out, exist_ok=True)
joblib.dump(m, os.path.join(out, "paracetamol_blend_screening_model.joblib"), compress=3)
card = dict(
    name="Paracetamol + talc blend screening model", version=__version__, trained=str(datetime.date.today()),
    intended_use="Qualitative screening of paracetamol/talc powder blends into LOW / WITHIN RANGE / HIGH label-claim bands. "
                 "NOT a replacement for the pharmacopoeial HPLC assay and not for batch release.",
    target="% label claim = paracetamol (g) / 2.00 g x 100 (talc fixed at 2.00 g); 80-120 % = 44.4-54.5 % w/w",
    training_data=dict(campaigns=sorted(train.campaign.unique().tolist()), n_samples=int(train.sample_id.nunique()), n_scans=len(train)),
    preprocessing="SNV then Savitzky-Golay 1st derivative (window 15, poly 2)",
    estimator="mean of SVR (C=100, RBF) and random forest (150 trees), trained on individual scans; sample result = median of scans",
    bands=dict(low_below=LOW_CUT - BORDER, within_range=[LOW_CUT + BORDER, HIGH_CUT - BORDER], high_above=HIGH_CUT + BORDER,
               borderline="within +/-5 of 90 or 110 -> confirm by HPLC"),
    applicability_domain=dict(pca_components=m.n_pc, t2_limit_p99=m.t2_lim, q_limit_p99=m.q_lim, blend_similarity_min=0.95,
                              min_fraction_scans_in_domain=0.5),
    validation_5fold_grouped_by_sample=dict(R2=0.83, RMSE_label_claim=4.84, RPD=2.48, bias=0.24,
                                            per_campaign=dict(C09=dict(R2=0.79, RMSE=5.07), C05=dict(R2=0.93, RMSE=3.23), C07=dict(RMSE=5.89))),
    known_limits=["No model transfers between collection years without recalibration (C09<->C05 RMSE 12-13 % LC)",
                  "Re-scans of calibration samples (C10) are not independent preparations",
                  "Reference values are gravimetric (weighed-in), not HPLC",
                  "Excipient system limited to talc; multi-excipient formulations are out of domain"],
)
json.dump(card, open(os.path.join(out, "model_card.json"), "w"), indent=2)
print("saved", m.t2_lim, m.q_lim)
