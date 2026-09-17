import os, sys, glob
import joblib
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from paracetamol_nir.io import read_any


def test_examples_screen():
    m = joblib.load(os.path.join(ROOT, "models", "paracetamol_blend_screening_model.joblib"))
    out = {}
    for f in glob.glob(os.path.join(ROOT, "examples", "*")):
        X, wl, _ = read_any(f)
        out[os.path.basename(f)] = m.screen(X, wl)["verdict"]
    assert out["formulation_raw_instrument_scan_OUT_OF_DOMAIN.csv"].startswith("OUT OF DOMAIN")
    assert not out["rescan_label101.5_absorbance.csv"].startswith("OUT OF DOMAIN")


if __name__ == "__main__":
    test_examples_screen(); print("ok")
