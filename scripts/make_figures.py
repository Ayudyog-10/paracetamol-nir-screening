"""Figures for the white paper (run from repo root after the analysis CSVs exist in reports/tables)."""
import sys, os
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from paracetamol_nir.preprocess import snv
from paracetamol_nir.model import band, prep
T, F = "reports/tables", "reports/figures"; os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
BLUE, ORG, GRN, RED, GRY = "#2F6F8F", "#D98C2B", "#4E9A5B", "#B84A4A", "#888888"
A = pd.read_parquet("data/paracetamol_nir_master.parquet"); ac = [c for c in A if c.startswith("A_")]; wl = np.array([float(c[2:]) for c in ac])
C = lambda s: A[A.campaign.str.startswith(s)]

# F1 campaign overview
cm = pd.read_csv("data/campaigns.csv")
fig, ax = plt.subplots(figsize=(7, 3))
lab = [c.replace("_", " ", 2) for c in cm.campaign]
ax.barh(lab, cm.n_spectra, color=BLUE); ax.invert_yaxis()
for i, (s, n) in enumerate(zip(cm.n_samples, cm.n_spectra)): ax.text(n + 10, i, f"{s} samples", va="center", fontsize=8)
ax.set_xlabel("number of spectra"); ax.set_xlim(0, 1150); ax.set_title("Collections in the master spectra file")
plt.tight_layout(); plt.savefig(f"{F}/f1_campaigns.png"); plt.close()

# F2 spectra
c5 = C("C05"); c9 = C("C09")
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
for d, l, col in [(c5[c5.api_pct_ww == 100], "pure paracetamol", RED), (c5[c5.api_pct_ww == 0], "pure talc", GRY), (C("C08"), "IPRS standard", ORG),
                  (c9[c9.label_claim_pct == 80], "blend 80 % LC", BLUE), (c9[c9.label_claim_pct == 120], "blend 120 % LC", GRN)]:
    ax[0].plot(wl, d[ac].values.mean(0), color=col, label=l, lw=1.2)
    ax[1].plot(wl, snv(d[ac].values).mean(0), color=col, lw=1.2)
ax[0].set(xlabel="wavelength (nm)", ylabel="absorbance", title="Mean raw spectra"); ax[0].legend(fontsize=7, frameon=False)
ax[1].set(xlabel="wavelength (nm)", ylabel="SNV absorbance", title="After SNV")
plt.tight_layout(); plt.savefig(f"{F}/f2_spectra.png"); plt.close()

# F3 leakage
r = pd.read_csv(f"{T}/benchmark_c09.csv"); r = r[r.pre == "SNV"]
keep = ["PLSR-8", "Ridge", "SVR", "KNN-3", "RF"]
fig, ax = plt.subplots(figsize=(6.5, 3)); x = np.arange(len(keep))
for off, sch, col, l in [(-0.2, "random", ORG, "random scan split (as in earlier report)"), (0.2, "grouped", BLUE, "sample-grouped CV (honest)")]:
    v = [r[(r.scheme == sch) & (r.model == k)].sample_RMSE.iloc[0] for k in keep]
    ax.bar(x + off, v, 0.4, color=col, label=l)
    for xi, vi in zip(x + off, v): ax.text(xi, vi + 0.1, f"{vi:.1f}", ha="center", fontsize=7)
ax.set_xticks(x, keep); ax.set_ylabel("RMSE (% label claim)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("2026 blends: the same models under two validation schemes")
plt.tight_layout(); plt.savefig(f"{F}/f3_leakage.png"); plt.close()

# F4 pred vs obs
cv = pd.read_csv(f"{T}/final_model_cv.csv")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.axhspan(95, 105, color=GRN, alpha=.08); ax.axhspan(85, 95, color=ORG, alpha=.08); ax.axhspan(105, 115, color=ORG, alpha=.08)
for c, col in [("C09", BLUE), ("C05", RED), ("C07", GRN)]:
    d = cv[cv.camp == c]; ax.scatter(d.y, d.p, s=18, color=col, label=c, alpha=.85)
ax.plot([78, 122], [78, 122], color=GRY, lw=1); ax.plot([78, 122], [83, 127], ":", color=GRY, lw=.8); ax.plot([78, 122], [73, 117], ":", color=GRY, lw=.8)
ax.set(xlim=(78, 122), ylim=(70, 125), xlabel="weighed-in label claim (%)", ylabel="predicted (%, median of scans)", title="Final model, 5-fold sample-grouped CV")
ax.legend(frameon=False, fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig(f"{F}/f4_pred_obs.png"); plt.close()

# F5 transfer
t = pd.read_csv(f"{T}/transfer.csv"); t = t[t.model.isin(["PLSR-8", "SVR", "RF"])]
tests = t.test.unique(); fig, ax = plt.subplots(figsize=(7, 3)); x = np.arange(len(tests))
for i, (mname, col) in enumerate([("PLSR-8", GRY), ("SVR", BLUE), ("RF", GRN)]):
    v = [t[(t.test == s) & (t.model == mname)].RMSE.iloc[0] for s in tests]
    ax.bar(x + (i - 1) * 0.27, np.minimum(v, 30), 0.27, color=col, label=mname)
short = {"C09 -> C10 rescans": "C09 -> C10\nrescans", "C09 (2026) -> C05 (2025 Mendine)": "C09 2026 ->\nC05 2025", "C09 (2026) -> C07 (2025 sub-samples)": "C09 2026 ->\nC07 2025", "C05 (2025) -> C09 (2026)": "C05 2025 ->\nC09 2026", "C01 (Jan 24) -> C02 (Apr 24)": "C01 Jan 24 ->\nC02 Apr 24 (mg)"}
ax.set_xticks(x, [short.get(s, s) for s in tests], fontsize=7); ax.set_ylabel("RMSE (clipped at 30)")
ax.set_title("Train on one collection, predict another (units: % LC; mg for C01->C02)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{F}/f5_transfer.png"); plt.close()

# F6 domain map
m = joblib.load("models/paracetamol_blend_screening_model.joblib")
fig, ax = plt.subplots(figsize=(5, 4))
for s, col, l in [("C09", BLUE, "C09 blends (train)"), ("C10", GRN, "C10 rescans"), ("C06", ORG, "C06 talc-varied"), ("C01", RED, "C01 formulation"), ("C02", "#7A4EA0", "C02 formulation")]:
    d = C(s); rr = m.predict_scans(d[ac].values)
    ax.scatter(rr["t2"], rr["q"], s=5, color=col, alpha=.5, label=l)
ax.axvline(m.t2_lim, color="k", lw=.8, ls="--"); ax.axhline(m.q_lim, color="k", lw=.8, ls="--")
ax.set(xscale="log", yscale="log", xlabel="Hotelling T²", ylabel="Q residual", title="Applicability-domain check (99 % limits)")
ax.legend(frameon=False, fontsize=7, markerscale=3)
plt.tight_layout(); plt.savefig(f"{F}/f6_domain.png"); plt.close()

# F7 confusion
cv["tb"] = cv.y.apply(band); cv["pb"] = cv.verdict.str.replace(" - confirm by HPLC", "").str.replace(" - not a paracetamol/talc blend like the calibration set", "")
order = ["LOW", "BORDERLINE", "WITHIN RANGE", "HIGH", "OUT OF DOMAIN"]
cv["tb"] = cv.tb.str.replace(" - confirm by HPLC", "")
M = pd.crosstab(cv.tb, cv.pb).reindex(index=order[:4], columns=order, fill_value=0)
fig, ax = plt.subplots(figsize=(5.5, 3)); ax.imshow(M.values, cmap="Blues")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, M.values[i, j], ha="center", va="center", fontsize=9)
ax.set_xticks(range(5), order, fontsize=7, rotation=20); ax.set_yticks(range(4), order[:4], fontsize=7)
ax.set_xlabel("screening verdict"); ax.set_ylabel("true band (weighed-in)"); ax.set_title("Screening outcome, 82 held-out samples")
plt.tight_layout(); plt.savefig(f"{F}/f7_confusion.png"); plt.close()
print("figures done")
