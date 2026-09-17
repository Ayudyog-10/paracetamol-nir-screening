"""Write the Drive-ready master workbook from the parquet files in data/."""
import sys, datetime
import pandas as pd, xlsxwriter

D = "data"; out = sys.argv[1] if len(sys.argv) > 1 else "Paracetamol_NIR_Master_Raw_Spectra.xlsx"
A = pd.read_parquet(f"{D}/paracetamol_nir_master.parquet")
C = pd.read_parquet(f"{D}/paracetamol_nir_sample_counts.parquet")
R = pd.read_parquet(f"{D}/paracetamol_nir_dark_ref_counts.parquet")
camp = pd.read_csv(f"{D}/campaigns.csv"); reg = pd.read_csv(f"{D}/sample_register.csv"); wl = pd.read_csv(f"{D}/wavelength_axis.csv")
A = A.drop(columns=["group_id"])
dictionary = [
    ("spectrum_id", "Unique ID of one scan (campaign code + running number). Links all sheets."),
    ("campaign", "Collection code (see Campaigns sheet)."),
    ("acq_date", "Acquisition date/time as recorded; 'undated' where the source file had none."),
    ("source_file", "Path of the original file inside the Paracetamol project folder."),
    ("sample_id", "Physical sample. C10 re-scans reuse the C09 sample IDs because they are the same powders."),
    ("portion / scan", "Cup fill (P1-P4) and replicate scan number within the fill."),
    ("material_system", "What was scanned."),
    ("role", "calibration / rescan_test / field_test / pure_component / identity_reference / repeat_scan."),
    ("dose_mg", "2024-25 formulation campaigns: nominal dose per tablet (mg). Paracetamol weighed = 12 x dose / 1000 g."),
    ("label_claim_pct", "Binary blends: paracetamol (g) / 2.00 g x 100. 80-120 % label claim."),
    ("para_g / diluent_g", "Weighed paracetamol and diluent (talc, or 1.110 g total excipients for formulations)."),
    ("api_pct_ww", "Paracetamol % w/w = para_g / (para_g + diluent_g) x 100."),
    ("exposure_us / avg_count", "Instrument settings when recorded in the raw file."),
    ("absorbance_source", "How absorbance was obtained. 'computed' = -log10((sample-dark)/(white-dark)) from raw counts."),
    ("temperature_c / humidity_pct", "Logged by Spectrafind where available."),
    ("qc_flag", "Known issues with the record. Read before using the row."),
    ("notes", "Free text, e.g. prediction logged by the software at the time."),
    ("A_<nm>", "Absorbance at that wavelength (256 points, 892.434-1709.864 nm)."),
    ("C_<nm>", "Raw detector counts (ADC) at that wavelength."),
]
wb = xlsxwriter.Workbook(out, {"constant_memory": False, "nan_inf_to_errors": True})
wb.formats[0].set_font_name("Arial"); wb.formats[0].set_font_size(10)
H = wb.add_format({"bold": True, "font_name": "Arial", "font_size": 10, "bg_color": "#2F6F8F", "font_color": "white", "border": 1, "text_wrap": True, "valign": "top"})
T = wb.add_format({"bold": True, "font_name": "Arial", "font_size": 14, "font_color": "#2F6F8F"})
B = wb.add_format({"bold": True, "font_name": "Arial", "font_size": 10})
W = wb.add_format({"font_name": "Arial", "font_size": 10, "text_wrap": True, "valign": "top"})
N = wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "0.000000"})
ws = wb.add_worksheet("README")
ws.set_column(0, 0, 28); ws.set_column(1, 1, 110)
ws.write(0, 0, "Paracetamol NIR - master raw spectra file", T)
info = [
    ("Owner", "METASPEQ / Ayudyog Private Limited"),
    ("Built", f"{datetime.date.today():%d %B %Y} from the 'Paracetamol' project folder (zip dated 17 Sep 2026)"),
    ("Instrument", "Hamamatsu C14486GA mini-spectrometer module (unit P1221018 in 2024 raw files), 256 pixels, 892-1710 nm, diffuse reflection"),
    ("Contents", f"{len(A)} scans from {A.sample_id.nunique()} physical samples in {A.campaign.nunique()} collections; raw counts for {C.spectrum_id.nunique()} scans; dark + white reference for {R.spectrum_id.nunique()} scans"),
    ("Excluded on purpose", "Sept 2024 formulation campaign (not in the folder) and the 25-28 Mar 2025 low-dose set (C04, formulation not documented). The C04 code is therefore skipped."),
    ("Reference values", "All targets are gravimetric (weighed-in) values. No HPLC assay values are attached to any scan."),
    ("How to use", "Filter the Absorbance sheet by campaign and role. Keep all scans of a sample together when splitting data for model validation - splitting scans at random inflates accuracy."),
    ("Known issues", "See qc_flag column. Pure-talc scan S0 of 11 May 2025 was labelled 101 in the source and is stored as 0 % API; Mendine Exp-2 labels were purity only, so % w/w was taken from the design table; C10 are re-scans of C09 powders, not new preparations."),
    ("Companion", "The same data are in data/*.parquet of the GitHub repo 'paracetamol-nir-screening'; scripts/build_master_dataset.py rebuilds everything from the raw folder."),
]
for i, (k, v) in enumerate(info, 2): ws.write(i, 0, k, B); ws.write(i, 1, v, W)
r0 = len(info) + 4
ws.write(r0 - 1, 0, "Column dictionary", T)
ws.write(r0, 0, "Column", H); ws.write(r0, 1, "Meaning", H)
for i, (k, v) in enumerate(dictionary, r0 + 1): ws.write(i, 0, k, B); ws.write(i, 1, v, W)

def dump(name, df, widths=None, numfmt_from=None):
    s = wb.add_worksheet(name)
    for j, c in enumerate(df.columns): s.write(0, j, c, H)
    vals = df.astype(object).where(df.notna(), None).values.tolist()
    for i, row in enumerate(vals, 1):
        for j, v in enumerate(row):
            if v is None: continue
            if numfmt_from is not None and j >= numfmt_from: s.write_number(i, j, v, N)
            else: s.write(i, j, v)
    s.freeze_panes(1, 2 if numfmt_from else 1); s.autofilter(0, 0, len(df), len(df.columns) - 1)
    for j, w in (widths or {}).items(): s.set_column(j, j, w)
    return s

dump("Campaigns", camp, {0: 32, 3: 24, 4: 60, 5: 14, 6: 16, 7: 12})
dump("Sample_Register", reg, {0: 32, 1: 22, 3: 16, 4: 50, 8: 20, 9: 60})
meta = [c for c in A.columns if not c.startswith("A_")]
dump("Absorbance", A, {0: 30, 1: 30, 3: 40, 4: 18, 7: 40, 18: 50}, numfmt_from=len(meta))
dump("Sample_ADC_Counts", C, {0: 30})
dump("Dark_WhiteRef_Counts", R, {0: 30, 1: 12})
dump("Wavelength_Axis", wl, {1: 16})
wb.close(); print("written", out)
