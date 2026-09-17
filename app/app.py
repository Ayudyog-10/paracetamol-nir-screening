"""METASPEQ Paracetamol NIR screening app (Streamlit)."""
import os
import sys
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from paracetamol_nir.io import read_any  # noqa: E402
from paracetamol_nir.preprocess import snv  # noqa: E402

st.set_page_config(page_title="Paracetamol NIR screening", page_icon="💊", layout="wide")


@st.cache_resource
def load():
    m = joblib.load(os.path.join(ROOT, "models", "paracetamol_blend_screening_model.joblib"))
    card = json.load(open(os.path.join(ROOT, "models", "model_card.json")))
    return m, card


model, card = load()
COLORS = {"LOW": "#B84A4A", "HIGH": "#B84A4A", "WITHIN RANGE": "#4E9A5B"}

st.title("Paracetamol + talc blend — NIR screening")
st.warning("**Screening tool only.** Results sort samples into LOW / WITHIN RANGE / HIGH label-claim bands. "
           "They do not replace the validated HPLC assay and must not be used for batch release.")

with st.sidebar:
    st.header("About the model")
    st.write(card["estimator"])
    v = card["validation_5fold_grouped_by_sample"]
    st.metric("Validated error (RMSE)", f"± {v['RMSE_label_claim']:.1f} % label claim")
    st.caption(f"R² {v['R2']:.2f} · RPD {v['RPD']:.2f} · {card['training_data']['n_samples']} samples, "
               f"{card['training_data']['n_scans']} scans")
    st.markdown("**Bands** (% label claim)\n\n- LOW: below 85\n- BORDERLINE: 85–95 or 105–115 → confirm by HPLC\n"
                "- WITHIN RANGE: 95–105\n- HIGH: above 115")
    st.markdown("**Valid only for** paracetamol blended with 2.00 g talc, scanned on the METASPEQ "
                "892–1710 nm module. Other formulations are flagged *out of domain*.")
    with st.expander("Known limits"):
        for k in card["known_limits"]:
            st.write("• " + k)

st.subheader("1 · Upload scans of ONE sample")
st.caption("Accepted: instrument raw CSV (Pixel, WaveLength, dark, ref, sample) — several files allowed — "
           "or a CSV/XLSX with one scan per row and wavelength column headers, or a Spectrafind export "
           "(Absorbance_1…256). Four or more scans are recommended.")
files = st.file_uploader("Spectra files", type=["csv", "xlsx"], accept_multiple_files=True)

demo_dir = os.path.join(ROOT, "examples")
demo = st.selectbox("…or try an example", ["(none)"] + sorted(os.listdir(demo_dir)))

Xs, wls = [], []
try:
    if files:
        for f in files:
            X, wl, _ = read_any(f.getvalue(), f.name)
            Xs.append(X); wls.append(wl)
    elif demo != "(none)":
        X, wl, _ = read_any(os.path.join(demo_dir, demo))
        Xs.append(X); wls.append(wl)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the file: {e}")

if Xs:
    X = np.vstack([model._check_axis(x, w) for x, w in zip(Xs, wls)])
    r = model.screen(X)
    st.subheader("2 · Result")
    base = r["verdict"].split(" - ")[0]
    col = COLORS.get(base, "#D98C2B")
    st.markdown(f"<div style='padding:14px;border-radius:8px;background:{col}22;border-left:6px solid {col}'>"
                f"<span style='font-size:1.6em;font-weight:700;color:{col}'>{r['verdict']}</span></div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indicative label claim", f"{r['median_label_claim']:.1f} %")
    c2.metric("Scan spread (IQR)", f"{r['iqr'][0]:.1f} – {r['iqr'][1]:.1f}")
    c3.metric("Scans inside model domain", f"{r['fraction_in_domain']:.0%} of {r['n_scans']}")
    c4.metric("Similarity to calibration blends", f"{r['blend_similarity']:.4f}")
    if base == "OUT OF DOMAIN":
        st.error("The indicative value is shown for information only — the sample does not look like the calibration set.")
    if r["n_scans"] < 4:
        st.info("Fewer than 4 scans — repack the sample cup and add scans for a steadier result.")

    s = r["scans"]
    tab1, tab2, tab3 = st.tabs(["Spectra", "Per-scan table", "Domain check"])
    with tab1:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        for x in X:
            ax[0].plot(model.wavelengths, x, lw=.8)
        ax[0].set(xlabel="wavelength (nm)", ylabel="absorbance", title="Uploaded scans")
        ax[1].plot(model.wavelengths, model.ref_blend, color="k", lw=1.5, label="calibration blend mean")
        for x in snv(X):
            ax[1].plot(model.wavelengths, x, lw=.8, alpha=.7)
        ax[1].set(xlabel="wavelength (nm)", ylabel="SNV", title="Compared with calibration set")
        ax[1].legend(frameon=False)
        st.pyplot(fig)
    with tab2:
        st.dataframe(pd.DataFrame({"scan": np.arange(1, len(X) + 1), "predicted % LC": s["pred"].round(1),
                                   "T²": s["t2"].round(1), "Q": s["q"].round(1), "in domain": s["in_domain"],
                                   "r (blend)": s["r_blend"].round(4), "r (pure API)": s["r_api"].round(3)}),
                     hide_index=True, width="stretch")
    with tab3:
        st.write(f"A scan is inside the domain when T² ≤ {model.t2_lim:.1f} and Q ≤ {model.q_lim:.1f} "
                 "(99th percentiles of the calibration scans). A sample is screened only if at least half of its "
                 "scans are inside and its spectrum correlates ≥ 0.95 with the calibration blends.")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.scatter(s["t2"], s["q"], color="#2F6F8F")
        ax.axvline(model.t2_lim, ls="--", color="k", lw=.8)
        ax.axhline(model.q_lim, ls="--", color="k", lw=.8)
        ax.set(xscale="log", yscale="log", xlabel="T²", ylabel="Q")
        st.pyplot(fig)
else:
    st.info("Upload scans or pick an example to begin.")

st.caption("METASPEQ · Ayudyog Private Limited · model v" + card["version"])
