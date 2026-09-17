"""Build the consolidated Paracetamol NIR master spectra file from the raw project folder.

Usage:  python scripts/build_master_dataset.py <path-to-Paracetamol-folder> <out-dir>
Every scan in every usable campaign becomes one row, with metadata columns followed by
256 absorbance columns (892-1710 nm, Hamamatsu C14486GA axis).
"""
import sys, os, re, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Paracetamol"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data"
os.makedirs(OUT, exist_ok=True)
TALC_G = 2.00
EXCIP_G = 0.471 + 0.282 + 0.188 + 0.113 + 0.056  # 1.110 g per formulation batch

def j(*p): return os.path.join(ROOT, *p)

# ---------- wavelength axis (from the instrument CSV header) ----------
def read_csv_scan(f):
    lines = open(f, encoding="latin1").read().splitlines()
    meta = {}
    for i, l in enumerate(lines):
        if l.startswith("Pixel,"): hi = i; break
        if "=" in l:
            k, v = l.split("=", 1); meta[k.strip()] = v.strip()
    arr = np.array([[float(x) for x in l.split(",")[:5]] for l in lines[hi + 2:] if l[:1].isdigit()])
    return meta, arr

f0 = sorted(glob.glob(j("Paracetamol Data 400 to 1200 mg ", "Spectral Data", "*.csv")))[0]
_, a0 = read_csv_scan(f0)
WL = a0[:, 1]
ACOLS = [f"A_{w:.3f}" for w in WL]
CCOLS = [f"C_{w:.3f}" for w in WL]

records, counts, darkref = [], [], []

def add(meta, absorb, cnt=None, dark=None, ref=None):
    rid = f"{meta['campaign']}__{len(records):05d}"
    meta = dict(spectrum_id=rid, **meta)
    records.append({**meta, **dict(zip(ACOLS, absorb))})
    if cnt is not None:
        counts.append({"spectrum_id": rid, **dict(zip(CCOLS, cnt))})
    if dark is not None:
        darkref.append({"spectrum_id": rid, "kind": "dark", **dict(zip(CCOLS, dark))})
        darkref.append({"spectrum_id": rid, "kind": "white_ref", **dict(zip(CCOLS, ref))})

def blend_from_label_claim(lc):
    p = TALC_G * lc / 100.0
    return p, 100 * p / (p + TALC_G)

def blend_from_ww(w):
    p = TALC_G * w / (100.0 - w)
    return p, 100 * p / TALC_G

# ---------- C01 / C02: 2024 formulation campaigns (raw instrument CSV) ----------
for camp, folder, note in [
    ("C01_2024-01_FORMULATION", j("Paracetamol Data 400 to 1200 mg ", "Spectral Data"), "Jan 2024, 41 doses 400-1200 mg"),
    ("C02_2024-04_FORMULATION", j("Paracetamol 550 to 750 mg with 5 minutes rest of Lamp"), "Apr 2024, 21 doses 550-750 mg, 5-min lamp rest"),
]:
    for f in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        meta, a = read_csv_scan(f)
        m = re.match(r"S(\d+)_(\d+)_P(\d+)_(\d+)\.csv", os.path.basename(f))
        dose = int(m[2]); para_g = 12 * dose / 1000
        d, r, s = a[:, 2], a[:, 3], a[:, 4]
        refl = (s - d) / (r - d)
        A = -np.log10(np.clip(refl, 1e-6, None))
        add(dict(campaign=camp, acq_date=pd.to_datetime(meta["Date"]).strftime("%Y-%m-%d %H:%M"),
                 source_file=os.path.relpath(f, ROOT), sample_id=f"{camp[:3]}_S{int(m[1]):02d}_{dose}mg",
                 portion=int(m[3]), scan=int(m[4]), material_system="formulation (API + MCC/PVP/CaSt/talc/MgSt)",
                 role="calibration", dose_mg=dose, label_claim_pct=np.nan, para_g=para_g, diluent_g=EXCIP_G,
                 api_pct_ww=100 * para_g / (para_g + EXCIP_G), exposure_us=int(meta["ExposureTime"]),
                 avg_count=int(meta["AverageCount"]), absorbance_source="computed -log10((S-D)/(R-D))",
                 temperature_c=np.nan, humidity_pct=np.nan, qc_flag="", notes=note), A, s, d, r)

# ---------- helper for header-less ab/ad workbooks (id, label, 256 values) ----------
def read_plain(f):
    t = pd.read_excel(f, header=None)
    if isinstance(t.iloc[0, 2], str) or not np.issubdtype(type(t.iloc[0, 2]), np.number):
        t = pd.read_excel(f)
        t.columns = range(t.shape[1])
    return t.dropna(subset=[2])

def read_headed(f):
    t = pd.read_excel(f)
    t.columns = ["id", "label"] + list(range(256))
    return t

# ---------- C03: 5 Mar 2025 Spectrafind field-test logs ----------
for fn_ab, fn_ad in [("05-03-2025-15-12.xlsx", "05-03-2025-15-12-AD.xlsx"), ("05-03-2025-16-44-ab.xlsx", "05-03-2025-16-44-ad.xlsx")]:
    ab = pd.read_excel(j("Paracetamol data test", "05-03-2025", fn_ab))
    ad = pd.read_excel(j("Paracetamol data test", "05-03-2025", fn_ad))
    ac = [c for c in ad.columns if str(c).startswith(("Imagedata", "Absorbance"))]
    ad_is_counts = ad[ac].values.max() > 50
    for i, row in ab.iterrows():
        A = row[[f"Absorbance_{k}" for k in range(1, 257)]].values.astype(float)
        cnt = ad.loc[i, ac].values.astype(float) if ad_is_counts else None
        add(dict(campaign="C03_2025-03-05_FIELD_TEST", acq_date=f"{row['Date']} {row['Time']}", source_file=f"Paracetamol data test/05-03-2025/{fn_ab}",
                 sample_id=f"C03_{int(row['Ref.'])}mg", portion=np.nan, scan=i + 1, material_system="formulation (assumed; not documented)",
                 role="field_test", dose_mg=float(row["Ref."]), label_claim_pct=np.nan, para_g=np.nan, diluent_g=np.nan, api_pct_ww=np.nan,
                 exposure_us=np.nan, avg_count=np.nan, absorbance_source="Spectrafind export", temperature_c=row["Temperature"],
                 humidity_pct=row["Humidity"], qc_flag="composition undocumented; logged prediction far from reference",
                 notes=f"Spectrafind logged prediction = {row['Predicted']}"), A, cnt)

# C04 (25-28 Mar 2025 low-dose set) intentionally excluded: formulation not documented.

# ---------- C05: Mendine Experiment 1 (paracetamol + fixed 2 g talc) ----------
f = j("000_Mendine", "Data taken", "09052025", "paracetamol_ab_accuracy test.xlsx")
ab = read_plain(f)
adf = read_plain(j("000_Mendine", "Data taken", "09052025", "paracetamol_ad_accuracy test.xlsx")).drop_duplicates(0).set_index(0)
for _, row in ab.iterrows():
    sid = str(row[0]); lab = float(row[1])
    m = re.match(r"(Paracetamol|Talc)_(?:(\d{8})_)?S(\d+)(?:_R(\d))?_P(\d+)_(\d+)", sid)
    day = m[2] or "09052025"; day = f"{day[4:]}-{day[2:4]}-{day[:2]}"
    flag, role = "", "calibration"
    if m[1] == "Talc":
        ww = 0.0; role = "pure_component"; lc = np.nan; p = 0.0
        if lab != 0: flag = f"label recorded as {lab} for pure talc; corrected to 0"
    elif int(m[3]) == 0:
        ww, lc, p, role = 100.0, np.nan, np.nan, "pure_component"
    else:
        ww = lab; p, lc = blend_from_ww(ww)
        if m[4]: role = "repeat_scan"
    cnt = adf.loc[sid].values[1:].astype(float) if sid in adf.index else None
    add(dict(campaign="C05_2025-05_MENDINE_EXP1", acq_date=day, source_file="000_Mendine/Data taken/09052025/paracetamol_ab_accuracy test.xlsx",
             sample_id=f"C05_{m[1][:4]}_S{int(m[3]):02d}" + (f"_R{m[4]}" if m[4] else ""), portion=int(m[5]), scan=int(m[6]),
             material_system="binary blend paracetamol + talc (2.00 g)" if 0 < ww < 100 else ("pure talc" if ww == 0 else "pure paracetamol (LOBA 98%)"),
             role=role, dose_mg=np.nan, label_claim_pct=lc, para_g=p, diluent_g=TALC_G if 0 < ww < 100 else np.nan, api_pct_ww=ww,
             exposure_us=np.nan, avg_count=np.nan, absorbance_source="exported absorbance", temperature_c=np.nan, humidity_pct=np.nan,
             qc_flag=flag, notes="Mendine QC lab, Kolkata"), row[2:258].values.astype(float), cnt)

auto = pd.read_excel(j("000_Mendine", "Data taken", "09052025", "Accuracy test Mendine_arbab", "Auto prediction", "ab auto.xlsx"))
for i, row in auto.iterrows():
    ww = float(row["Ref."]); p, lc = blend_from_ww(ww)
    add(dict(campaign="C05_2025-05_MENDINE_EXP1", acq_date=f"{row['Date']} {row['Time']}", source_file="000_Mendine/.../Auto prediction/ab auto.xlsx",
             sample_id=f"C05_AUTO_{ww:g}", portion=np.nan, scan=i + 1, material_system="binary blend paracetamol + talc (2.00 g)", role="field_test",
             dose_mg=np.nan, label_claim_pct=lc, para_g=p, diluent_g=TALC_G, api_pct_ww=ww, exposure_us=np.nan, avg_count=np.nan,
             absorbance_source="Spectrafind export", temperature_c=row["Temperature"], humidity_pct=row["Humidity"], qc_flag="",
             notes=f"Spectrafind logged prediction = {row['Predicted']}"), row[[f"Absorbance_{k}" for k in range(1, 257)]].values.astype(float))

# ---------- C06: Mendine Experiment 2 (fixed 2 g paracetamol, talc varied) ----------
design = pd.read_excel(j("000_Mendine", "Experiment 2", "Proposed Table for Experiment-2.xlsx")).dropna(subset=["No. of sample "])
design = design.set_index(design["No. of sample "].astype(int))
ab = read_plain(j("000_Mendine", "Experiment 2", "ab_exp2.xlsx"))
ad = read_plain(j("000_Mendine", "Experiment 2", "ad_exp2.xlsx"))
ab["k"] = ab.groupby(0).cumcount() + 1
for (i, row), (_, rc) in zip(ab.iterrows(), ad.iterrows()):
    n = int(row[0]); talc = float(design.loc[n, "Weight of Talc(gm)"]); p = 2.0
    ww = 100 * p / (p + talc)
    add(dict(campaign="C06_2025_MENDINE_EXP2", acq_date="2025 (undated)", source_file="000_Mendine/Experiment 2/ab_exp2.xlsx",
             sample_id=f"C06_S{n:02d}", portion=(row["k"] - 1) // 4 + 1, scan=(row["k"] - 1) % 4 + 1,
             material_system="binary blend 2.00 g paracetamol + varied talc", role="calibration", dose_mg=np.nan,
             label_claim_pct=np.nan, para_g=p, diluent_g=talc, api_pct_ww=ww, exposure_us=np.nan, avg_count=np.nan,
             absorbance_source="exported absorbance", temperature_c=np.nan, humidity_pct=np.nan,
             qc_flag="label column held purity (98) only; api_pct_ww derived from design table by sample number",
             notes="specificity experiment"), row[2:258].values.astype(float), rc[2:258].values.astype(float))

# ---------- C07: Mendine 0.99 set (sub-samples 15a-d, 16a-d) ----------
ab = read_plain(j("000_Mendine", "Mendine_0.99", "ab new.xlsx"))
ad = read_plain(j("000_Mendine", "Mendine_0.99", "ad new.xlsx")).set_index(0)
for _, row in ab.iterrows():
    sid = str(row[0]); m = re.match(r"(\d+)\((\w)\)_P(\d+)_(\d+)", sid); lc = float(row[1]); p, ww = blend_from_label_claim(lc)
    add(dict(campaign="C07_2025_MENDINE_SUBSAMPLES", acq_date="2025 (undated)", source_file="000_Mendine/Mendine_0.99/ab new.xlsx",
             sample_id=f"C07_{m[1]}{m[2]}", portion=int(m[3]), scan=int(m[4]), material_system="binary blend paracetamol + talc (2.00 g)",
             role="calibration", dose_mg=np.nan, label_claim_pct=lc, para_g=p, diluent_g=TALC_G, api_pct_ww=ww, exposure_us=np.nan,
             avg_count=np.nan, absorbance_source="exported absorbance", temperature_c=np.nan, humidity_pct=np.nan, qc_flag="", notes=""),
        row[2:258].values.astype(float), ad.loc[sid].values[1:].astype(float) if sid in ad.index else None)

# ---------- C08: IPRS reference standard ----------
ab = read_plain(j("000_Mendine", "Identification", "ab_id.xlsx"))
ad = read_plain(j("000_Mendine", "Identification", "ad_new.xlsx"))
for k, ((_, row), (_, rc)) in enumerate(zip(ab.iterrows(), ad.iterrows()), 1):
    add(dict(campaign="C08_2025_IPRS_STANDARD", acq_date="2025 (undated)", source_file="000_Mendine/Identification/ab_id.xlsx",
             sample_id="C08_IPRS_paracetamol", portion=1, scan=k, material_system="IPRS paracetamol reference standard",
             role="identity_reference", dose_mg=np.nan, label_claim_pct=np.nan, para_g=np.nan, diluent_g=np.nan, api_pct_ww=100.0,
             exposure_us=np.nan, avg_count=np.nan, absorbance_source="exported absorbance", temperature_c=np.nan, humidity_pct=np.nan,
             qc_flag="", notes=""), row[2:258].values.astype(float), rc[2:258].values.astype(float))

# ---------- C09 / C10: 2026 binary-blend calibration + re-scan tests ----------
S = j("Paracetamol supervision_sudip da")
t = read_headed(os.path.join(S, "absorbance full data for Paracetamol.xlsx"))
for _, row in t.iterrows():
    _, s, r = str(row["id"]).split("_"); lc = float(row["label"]); p, ww = blend_from_label_claim(lc)
    add(dict(campaign="C09_2026-03_BLEND_CALIBRATION", acq_date="2026-03 (undated)", source_file="Paracetamol supervision_sudip da/absorbance full data for Paracetamol.xlsx",
             sample_id=f"C09_S{int(s):02d}", portion=(int(r) - 1) // 4 + 1, scan=(int(r) - 1) % 4 + 1,
             material_system="binary blend paracetamol + talc (2.00 g)", role="calibration", dose_mg=np.nan, label_claim_pct=lc,
             para_g=p, diluent_g=TALC_G, api_pct_ww=ww, exposure_us=np.nan, avg_count=np.nan, absorbance_source="exported absorbance",
             temperature_c=np.nan, humidity_pct=np.nan, qc_flag="", notes=""), row[list(range(256))].values.astype(float))
# ADC counts where available
t = read_headed(os.path.join(S, "11 & 12 -03-2026", "test_ab.xlsx"))
ads = pd.concat([read_headed(os.path.join(S, "11 & 12 -03-2026", "test_ad.xlsx")), read_headed(os.path.join(S, "16-03-2026", "ad.xlsx")),
                 read_headed(os.path.join(S, "17-03-2026", "ad.xlsx"))])
t["k"] = t.groupby("id").cumcount()
ads["k"] = ads.groupby("id").cumcount()
adk = ads.drop_duplicates(["id", "k"]).set_index(["id", "k"])
for _, row in t.iterrows():
    sid = str(row["id"]); m = re.match(r"[sS]_(\d+)(?:_(\d))?", sid); lc = float(row["label"]); p, ww = blend_from_label_claim(lc)
    key = (sid, row["k"])
    add(dict(campaign="C10_2026-03_BLEND_RESCAN_TEST", acq_date="2026-03-11..17", source_file="Paracetamol supervision_sudip da/11 & 12 -03-2026/test_ab.xlsx",
             sample_id=f"C09_S{int(m[1]):02d}", portion=np.nan, scan=int(m[2]) if m[2] else row["k"] + 1,
             material_system="binary blend paracetamol + talc (2.00 g)", role="rescan_test", dose_mg=np.nan, label_claim_pct=lc,
             para_g=p, diluent_g=TALC_G, api_pct_ww=ww, exposure_us=np.nan, avg_count=np.nan, absorbance_source="exported absorbance",
             temperature_c=np.nan, humidity_pct=np.nan, qc_flag="same physical samples as C09 (not independent preparations)", notes=""),
        row[list(range(256))].values.astype(float), adk.loc[key, list(range(256))].values.astype(float) if key in adk.index else None)

# ---------- assemble ----------
A = pd.DataFrame(records)
C = pd.DataFrame(counts)
DR = pd.DataFrame(darkref)
meta_cols = [c for c in A.columns if not c.startswith("A_")]
A["group_id"] = A["sample_id"]
reg = (A.groupby(["campaign", "sample_id"], sort=False)
         .agg(n_scans=("spectrum_id", "size"), role=("role", "first"), material=("material_system", "first"),
              dose_mg=("dose_mg", "first"), label_claim_pct=("label_claim_pct", "first"), api_pct_ww=("api_pct_ww", "first"),
              first_date=("acq_date", "first"), qc_flag=("qc_flag", "first")).reset_index())
camps = (A.groupby("campaign").agg(n_spectra=("spectrum_id", "size"), n_samples=("sample_id", "nunique"),
                                    dates=("acq_date", lambda s: f"{s.astype(str).min()[:10]} .. {s.astype(str).max()[:10]}"),
                                    material=("material_system", lambda s: "; ".join(sorted(set(s)))),
                                    dose_range=("dose_mg", lambda s: "" if s.isna().all() else f"{s.min():g}-{s.max():g} mg"),
                                    ww_range=("api_pct_ww", lambda s: "" if s.isna().all() else f"{s.min():.2f}-{s.max():.2f} %"),
                                    has_counts=("spectrum_id", lambda s: int(s.isin(C.spectrum_id).sum()) if len(C) else 0)).reset_index())
A = A[meta_cols + ["group_id"] + ACOLS]
A.to_parquet(os.path.join(OUT, "paracetamol_nir_master.parquet"), index=False)
C.to_parquet(os.path.join(OUT, "paracetamol_nir_sample_counts.parquet"), index=False)
DR.to_parquet(os.path.join(OUT, "paracetamol_nir_dark_ref_counts.parquet"), index=False)
reg.to_csv(os.path.join(OUT, "sample_register.csv"), index=False)
camps.to_csv(os.path.join(OUT, "campaigns.csv"), index=False)
pd.DataFrame({"pixel": range(1, 257), "wavelength_nm": WL}).to_csv(os.path.join(OUT, "wavelength_axis.csv"), index=False)
print(camps.to_string())
print("spectra", len(A), "counts", len(C), "dark/ref", len(DR))
