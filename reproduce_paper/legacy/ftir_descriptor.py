#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpretable geometric band descriptor for biofluid ATR-FTIR spectra, evaluated on two public
salivary datasets with an acquisition audit based on negative-control spectral regions.

DATASETS (prepared in a common format by prepare_datasets.py)
  T2D_saliva              Type 2 diabetes vs. control (1040 subjects, one spectrum each).
  COVID_original          COVID-19 PCR-positive vs. negative as in the original study (104 subjects with
                          replicates; enrolment group is almost completely confounded with collection site).
  COVID_NSW_PCR           Within a single site (NSW): follow-up PCR positive vs. negative on the day of saliva
                          collection (no site confounding).
  COVID_site_NSW_vs_TPCH  Audit: COVID-positive subjects only, NSW vs. TPCH (the label is the collection site).

DESCRIPTOR
  Common core of bands present in any biological sample (proteins, lipids, carbohydrates, phosphates,
  carboxylates, amide A / O-H), measured as the geometry of each band envelope (relative area, position,
  FWHM, sharpness, asymmetry), plus log area ratios, carbohydrate sub-band fractions, second-derivative
  intensities at peaks detected in the pooled mean spectrum (label-free) and band centroids.
  Instrument rule: the diamond multiphonon region (1800-2700 cm-1) is excluded for diamond ATR data.
  Quality-control rule: a band invalid in more than QC_MAX_INVALID of the spectra of any dataset is excluded
  (label-free). The descriptor configuration is shared by all datasets.

PREPROCESSING COMMON TO ALL METHODS
  Resampling to a 1 cm-1 grid and noise harmonization: when the high-frequency noise level separates the
  classes of a dataset (|AUC - 0.5| > 0.1, a fixed dataset-level rule), white Gaussian noise is added to
  each spectrum so that all spectra reach the same noise level (the noisiest spectrum). The per-spectrum
  operation does not use labels.

METHODS
  Primary model: Descriptor_GB (descriptor + histogram gradient boosting).
  Secondary: Descriptor_LR (interpretable, logistic regression), Descriptor_base_LR (ablation: original 46
  variables).
  Full-spectrum references: Full_spectrum, PCA, Second_derivative, PLS-DA.
  Equal-budget compact comparators: Binning, Random_intervals, VIP_topk, SPA_LDA, Literature_panel.
  Acquisition audit (negative controls): Neg_control_raw and Neg_control_d2 (3750-3995 cm-1, no sample
  bands, outside the diamond region) and Diamond_control_d2 (diamond region only).
  Hyperparameters are tuned in inner cross-validation (nested CV, grouped by subject when there are
  replicates). Non-inferiority vs. each full-spectrum reference uses the Nadeau-Bengio corrected t-test.

USAGE
  python prepare_datasets.py        (once)
  python ftir_descriptor.py          (full run: several hours)
  Environment variables:
    FTIR_EXPLORATORY=1   5x3x1 validation, subset of methods, no robustness/permutation tests
    FTIR_QUICK=1         3x3x1, only to check that everything runs
    FTIR_DATASETS=T2D_saliva             nested validation only for these datasets (to split long runs;
                                         all datasets are still loaded for the shared descriptor)
    FTIR_OUT=/content/drive/MyDrive/results   output folder (default: results_ftir_descriptor)
    FTIR_RESUME=0        recompute datasets whose nested validation was already saved in OUT_DIR/cv_parts
                         (by default, finished datasets are reused, so an interrupted run can be resumed)

OUTPUTS (in OUT_DIR)
  descriptors_<dataset>.csv, descriptor_configuration.json, excluded_bands.csv, detected_peaks.csv,
  noise_harmonization_<dataset>.csv, band_quality.csv, global_quality.csv, feature_reliability.csv,
  perturbation_magnitudes.csv, cv_details.csv, cv_summary.csv, paired_comparisons.csv,
  noninferiority_global.csv, selection_stability.csv, permutation_test.csv, region_importance.csv,
  feature_effects.csv, cross_dataset_effects.csv, fig1_spectra_bands.png, fig2_performance_robustness.png,
  fig3_effect_map.png, fig4_noninferiority.png, fig5_region_importance.png, fig6_acquisition_audit.png

Requirements: numpy, scipy, pandas, scikit-learn, matplotlib
"""

import csv
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import mannwhitneyu, chi2
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (StratifiedKFold, StratifiedGroupKFold, GridSearchCV,
                                     StratifiedShuffleSplit)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, log_loss
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

DESCRIPTOR_VERSION = "1.0"

# =============================================================================
# CONFIGURATION
# =============================================================================
# role "disease": diagnostic comparison (enters the cross-dataset effect comparison)
# role "audit":   acquisition/site audit (reported separately)
# n_inner: inner folds for datasets with few subjects in the minority class (default N_INNER)
DATASETS = {
    "T2D_saliva":             {"path": "t2d_saliva.csv", "positive": "T2D", "group": "subject",
                               "role": "disease", "literature_panel": "T2D"},
    "COVID_original":         {"path": "covid_original.csv", "positive": "COVID_POS", "group": "subject",
                               "role": "audit"},
    "COVID_NSW_PCR":          {"path": "covid_nsw_pcr.csv", "positive": "PCR_POS", "group": "subject",
                               "role": "disease", "n_inner": 3},
    "COVID_site_NSW_vs_TPCH": {"path": "covid_site_nsw_vs_tpch.csv", "positive": "TPCH", "group": "subject",
                               "role": "audit", "n_inner": 3},
}
# All datasets are always loaded (the descriptor configuration and the effect analysis use all of them);
# FTIR_DATASETS restricts only the nested validation, which is the slow part.
_sel = os.environ.get("FTIR_DATASETS", "").strip()
RUN_CV = [k for k in DATASETS if not _sel or k in [s.strip() for s in _sel.split(",")]]
if not RUN_CV:
    raise ValueError(f"FTIR_DATASETS={_sel} does not match any dataset name.")

LABEL_COL = "label"
OUT_DIR = Path(os.environ.get("FTIR_OUT", "results_ftir_descriptor"))

WN_MIN, WN_MAX, STEP = 900.0, 3995.0, 1.0
DIAMOND_ATR = True                                       # both datasets were acquired with diamond ATR
DIAMOND_REGION = (1800.0, 2700.0)                        # diamond multiphonon absorption
BIO_WINDOWS = [(900.0, 1800.0), (2800.0, 3700.0)] if DIAMOND_ATR else [(900.0, 3700.0)]
D2_WINDOWS = [(900.0, 1800.0), (2800.0, 3050.0)]         # second-derivative peak search (no water-vapour region)
NEG_CONTROL_WINDOWS = ((3750.0, 3995.0),)                 # no sample bands, outside the diamond region
DIAMOND_WINDOWS = ((1820.0, 1980.0), (2160.0, 2690.0))    # inside the diamond absorption
NOISE_WINDOWS = [(1780.0, 1840.0), (2700.0, 2800.0)]      # band-free regions for noise estimation

SG_WINDOW, SG_POLY = 11, 3      # second-derivative reference method
DESC_SG_WINDOW = 21             # descriptor smoothing (chosen for robustness to noise)
D2F_WINDOW = 13                 # second-derivative features of the descriptor
RELIABILITY_MAX = 0.5           # change due to noise / between-subject SD above which a variable is flagged

HARMONIZE_NOISE = "auto"        # "auto": harmonize only if the noise level separates the classes
                                # (|AUC - 0.5| > NOISE_AUC_TOL); True: always; False: never
NOISE_AUC_TOL = 0.1
NOISE_PERCENTILE = 100          # 100 = level of the noisiest spectrum

# Core bands: (name, lower, upper) in cm-1
BANDS = [
    ("CH3as_lipids",          2945.0, 2990.0),
    ("CH2as_lipids",          2900.0, 2945.0),
    ("CH2s_lipids",           2835.0, 2870.0),
    ("Amide_I",               1600.0, 1700.0),
    ("Amide_II",              1500.0, 1590.0),
    ("COO_sym",               1370.0, 1430.0),
    ("AmideIII_PO2as",        1200.0, 1280.0),
    ("PO2s_CO_carbohydrates",  995.0, 1125.0),
]
RATIOS = [
    ("log_AmideI_AmideII",     ["Amide_I"],                                     ["Amide_II"]),
    ("log_Lipids_AmideI",      ["CH3as_lipids", "CH2as_lipids", "CH2s_lipids"], ["Amide_I"]),
    ("log_CH3_CH2",            ["CH3as_lipids"],                                ["CH2as_lipids"]),
    ("log_COO_AmideI",         ["COO_sym"],                                     ["Amide_I"]),
    ("log_PO2as_AmideI",       ["AmideIII_PO2as"],                              ["Amide_I"]),
    ("log_PO2sCOcarb_AmideI",  ["PO2s_CO_carbohydrates"],                       ["Amide_I"]),
]
# Additional envelope bands (relative areas use the total of the core bands)
BANDS_EXTRA = [
    ("Carbonyl_CO",     1700.0, 1770.0),   # C=O of carboxylic acids (~1715) and esters (~1740)
    ("CH2CH3_bending",  1430.0, 1480.0),   # CH2/CH3 bending of lipids and side chains
    ("AmideA_OH",       3050.0, 3650.0),   # amide A (N-H, ~3290) and water O-H (~3400)
]
SUBBANDS = [
    ("Carbohydrates", 995.0, 1185.0, [
        ("CO_glycogen_1030", 995.0, 1055.0),
        ("PO2s_CO_1080",     1055.0, 1095.0),
        ("CO_COH_1110",      1095.0, 1130.0),
        ("COC_1150",         1130.0, 1185.0),
    ]),
]
# Second-derivative block: peaks detected in the pooled mean spectrum (label-free); fixed positions are
# used only if DETECT_PEAKS is False.
DETECT_PEAKS = True
D2_TOL = 6.0                 # search tolerance for fixed positions (cm-1)
D2_TOL_DET = 3.0             # search tolerance for detected peaks (cm-1)
D2_REF = (1640.0, 1665.0)    # amide I reference for normalization
D2_MIN_DIST = 8.0
D2_PROM_REL = 0.01
D2_MAX_PEAKS = 60
D2_EDGE_MARGIN = 10.0
D2_FIXED = [
    ("d2_glycogen_1030", 1030.0), ("d2_PO2s_1080", 1080.0), ("d2_CO_1110", 1110.0), ("d2_COC_1150", 1150.0),
    ("d2_PO2as_1240", 1240.0), ("d2_COO_1400", 1400.0), ("d2_CH2bend_1455", 1455.0),
    ("d2_tyrosine_1515", 1515.0), ("d2_AmideII_1545", 1545.0), ("d2_beta_sheet_1630", 1630.0),
    ("d2_turns_1685", 1685.0), ("d2_ester_1740", 1740.0), ("d2_CH3as_2958", 2958.0),
    ("d2_CH2as_2922", 2922.0), ("d2_CH2s_2852", 2852.0),
]
ASSIGNMENTS = [
    (900.0, 990.0, "PO4 / ribose, C-C"), (990.0, 1060.0, "C-O carbohydrates (glucose/glycogen)"),
    (1060.0, 1095.0, "PO2- symmetric / C-O"), (1095.0, 1130.0, "C-O / C-O-H carbohydrates"),
    (1130.0, 1185.0, "C-O-C glycogen / glycoproteins"), (1185.0, 1290.0, "PO2- asymmetric / amide III"),
    (1290.0, 1360.0, "amide III / CH2 wagging"), (1360.0, 1430.0, "COO- symmetric / CH3"),
    (1430.0, 1480.0, "CH2/CH3 bending"), (1480.0, 1525.0, "tyrosine / aromatic ring"),
    (1525.0, 1600.0, "amide II"), (1600.0, 1640.0, "amide I: beta sheet"),
    (1640.0, 1665.0, "amide I: alpha helix / random coil"), (1665.0, 1700.0, "amide I: turns / antiparallel beta"),
    (1700.0, 1780.0, "C=O of esters / acids"), (2800.0, 2870.0, "CH2 symmetric (lipids)"),
    (2870.0, 2900.0, "CH3 symmetric"), (2900.0, 2940.0, "CH2 asymmetric (lipids)"),
    (2940.0, 3000.0, "CH3 asymmetric"), (3000.0, 3050.0, "=C-H olefinic"),
]
QC_MAX_INVALID = 0.5
EXTRA_RATIOS = ["log_I1030_I1080", "log_beta_alpha_d2"]
BAND_RATIOS = [
    ("log_Carbonyl_lipidsCH", ["Carbonyl_CO"], ["CH3as_lipids", "CH2as_lipids", "CH2s_lipids"]),
    ("log_AmideAOH_AmideI",   ["AmideA_OH"],   ["Amide_I"]),
]
CENTROIDS = [("centroid_carbohydrates", 995.0, 1185.0), ("centroid_AmideI", 1600.0, 1700.0),
             ("centroid_AmideA_OH", 3050.0, 3650.0)]
BLOCKS = ("core", "extra_bands", "subbands", "derivative", "extra_ratios", "centroids")

MEASURES = ["rel_area", "position", "fwhm", "sharpness", "asymmetry"]
FLAGS = ["no_peak", "peak_at_edge", "multiple_peaks", "fwhm_truncated", "narrow_window", "low_snr"]
FLAGS_INVALID = ["no_peak", "peak_at_edge", "fwhm_truncated"]
FLAGS_DOUBTFUL = ["multiple_peaks", "narrow_window", "low_snr"]
SNR_MIN = 10.0
SNR_INVALID = 3.0
EDGE_FRAC = 0.05
FWHM_MAX_FRAC = 0.6

PRIMARY = "Descriptor_GB"
REFERENCES = ["Full_spectrum", "PCA", "Second_derivative", "PLS-DA"]
COMPACT_REFERENCES = ["Binning", "Random_intervals", "VIP_topk", "SPA_LDA", "Literature_panel"]
CONTROLS = ["Neg_control_raw", "Neg_control_d2", "Diamond_control_d2"]
FOCAL = ["Descriptor_GB", "Descriptor_LR", "Descriptor_base_LR"]
METHOD_ORDER = ["Full_spectrum", "PCA", "Second_derivative", "PLS-DA",
                "Binning", "Random_intervals", "VIP_topk", "SPA_LDA", "Literature_panel",
                "Neg_control_raw", "Neg_control_d2", "Diamond_control_d2",
                "Descriptor_base_LR", "Descriptor_LR", "Descriptor_GB"]
EXPLORATORY_METHODS = ["Full_spectrum", "Second_derivative", "Binning", "Neg_control_raw", "Neg_control_d2",
                       "Diamond_control_d2", "Descriptor_base_LR", "Descriptor_LR", "Descriptor_GB"]

SPA_K_MAX, SPA_N_STARTS, SPA_STEP = 20, 20, 2
INTERVAL_WIDTH = 15
# Literature panels: (centre cm-1, half-width cm-1); band areas over a chord baseline on the spectrum
# normalized by its area in 900-1800 cm-1. T2D: bands reported higher in T2D saliva (Caixeta et al.).
LITERATURE_PANELS = {"T2D": [(2962.0, 15.0), (1641.0, 20.0), (1073.0, 20.0)]}

IMPORTANCE_METHODS = ["Second_derivative", "Full_spectrum", "Descriptor_LR", "Descriptor_GB"]
IMP_WIDTH = 20.0
IMP_REP = 5

NI_MARGIN = 0.03           # non-inferiority margin (balanced accuracy), fixed before the final run
N_OUTER, N_INNER, N_REPEATS = 5, 5, 3
N_PERM = 500
PERM_REFOLD = False
N_BOOT = 500
ALPHA = 0.05
SEED = 42
PERTURBATIONS = ["scale", "baseline", "noise"]
PERTURB_FACTOR = 2.0
PERTURB_DEFAULT = {"log_scale_sd": 0.15, "offset_sd": 0.05, "slope_sd": 0.10}

QUICK = os.environ.get("FTIR_QUICK", "0") == "1"
EXPLORATORY = os.environ.get("FTIR_EXPLORATORY", "0") == "1"
if QUICK:
    N_OUTER, N_INNER, N_REPEATS, N_PERM, N_BOOT = 3, 3, 1, 50, 100
if EXPLORATORY:
    N_OUTER, N_INNER, N_REPEATS, N_PERM, N_BOOT = 5, 3, 1, 0, 200
    PERTURBATIONS = []

GRID = np.arange(WN_MIN, WN_MAX + STEP / 2, STEP)
_min_band_pts = min(int(((GRID >= lo) & (GRID <= hi)).sum()) for _, lo, hi in BANDS + BANDS_EXTRA)
for _w, _name in [(DESC_SG_WINDOW, "DESC_SG_WINDOW"), (SG_WINDOW, "SG_WINDOW"), (D2F_WINDOW, "D2F_WINDOW")]:
    if _w % 2 == 0 or _w <= SG_POLY:
        raise ValueError(f"{_name} = {_w} must be odd and larger than SG_POLY = {SG_POLY}.")
if DESC_SG_WINDOW > _min_band_pts:
    raise ValueError(f"DESC_SG_WINDOW = {DESC_SG_WINDOW} exceeds the {_min_band_pts} points of the narrowest band.")


# =============================================================================
# DATA LOADING
# =============================================================================
def load_dataset(path, group_col=None):
    """CSV with one row per spectrum, the class column LABEL_COL, an optional subject column and numeric
    wavenumber headers (cm-1). Other non-numeric columns are ignored."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {Path(path).resolve()}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        raw = next(csv.reader(fh))
    raw_num = []
    for c in raw:
        try:
            raw_num.append(float(c))
        except ValueError:
            pass
    if len(set(raw_num)) != len(raw_num):
        raise ValueError(f"{path}: duplicated wavenumbers in the header.")
    df = pd.read_csv(path)
    meta = {LABEL_COL} | ({group_col} if group_col else set())
    spec_cols, wn = [], []
    for c in df.columns:
        if c in meta:
            continue
        try:
            wn.append(float(c))
            spec_cols.append(c)
        except ValueError:
            continue
    for col, what in [(LABEL_COL, "class"), (group_col, "subject")]:
        if col and col not in df.columns:
            raise ValueError(f"{path}: missing {what} column '{col}'.")
        if col and df[col].isna().any():
            raise ValueError(f"{path}: missing values in column '{col}'.")
    wn = np.array(wn)
    if len(wn) < 3:
        raise ValueError(f"{path}: fewer than 3 numeric spectral columns.")
    order = np.argsort(wn)
    X = df[spec_cols].to_numpy(float)[:, order]
    if not np.all(np.isfinite(X)):
        rows = np.where(~np.isfinite(X).all(axis=1))[0][:5]
        raise ValueError(f"{path}: NaN or infinite absorbances (first affected rows: {list(rows)}).")
    y = df[LABEL_COL].astype(str).to_numpy()
    g = df[group_col].astype(str).to_numpy() if group_col else None
    return wn[order], X, y, g


def resample(wn, X):
    if GRID[0] < wn[0] - 1e-9 or GRID[-1] > wn[-1] + 1e-9:
        raise ValueError(f"Spectral range {wn[0]:.0f}-{wn[-1]:.0f} does not cover {WN_MIN:.0f}-{WN_MAX:.0f} cm-1")
    return np.vstack([np.interp(GRID, wn, x) for x in X])


def binarize(y_raw, positive, name):
    classes = sorted(np.unique(y_raw))
    if len(classes) != 2:
        raise ValueError(f"{name}: exactly two classes are required, found {classes}")
    if positive not in classes:
        raise ValueError(f"{name}: positive class '{positive}' not in {classes}")
    neg = [c for c in classes if c != positive][0]
    return (y_raw == positive).astype(int), positive, neg


def check_sample_size(y, g, name, n_inner):
    unit = g if g is not None else np.arange(len(y)).astype(str)
    per_class = pd.DataFrame({"u": unit, "y": y}).drop_duplicates("u")["y"].value_counts()
    req = int(np.ceil(2 * n_inner * N_OUTER / (N_OUTER - 1)))
    if int(per_class.min()) < req:
        raise ValueError(f"{name}: {int(per_class.min())} subjects in the minority class; at least {req} are "
                         f"required for {N_OUTER}x{n_inner} nested validation.")
    if int(per_class.min()) < 30:
        print(f"  WARNING {name}: fewer than 30 subjects in the minority class; estimates will be imprecise.")


def check_groups(y, g, name):
    if g is None:
        return
    bad = pd.DataFrame({"y": y, "g": g}).groupby("g")["y"].nunique()
    if (bad > 1).any():
        raise ValueError(f"{name}: subjects with spectra of different classes: {list(bad[bad > 1].index)[:5]}")


# =============================================================================
# NOISE
# =============================================================================
def noise_abs(X):
    """Absolute noise per spectrum: high-frequency residual in band-free regions."""
    X = np.atleast_2d(np.asarray(X, float))
    sds = []
    for lo, hi in NOISE_WINDOWS:
        m = (GRID >= lo) & (GRID <= hi)
        sds.append((X[:, m] - savgol_filter(X[:, m], 21, 2, axis=1)).std(axis=1))
    return np.mean(sds, axis=0)


def noise_rel(X):
    X = np.atleast_2d(np.asarray(X, float))
    amp = X.max(axis=1) - X.min(axis=1)
    return np.divide(noise_abs(X), amp, out=np.full_like(amp, np.nan), where=amp > 0)


def harmonize_noise(name, X, y):
    """Adds white Gaussian noise to each spectrum up to a common level (NOISE_PERCENTILE of the measured
    absolute noise). Label-free; y is only used to report the check."""
    n0 = noise_abs(X)
    auc0 = roc_auc_score(y, n0)
    if HARMONIZE_NOISE == "auto" and abs(auc0 - 0.5) <= NOISE_AUC_TOL:
        print(f"  {name}: noise level does not separate the classes (AUC {auc0:.3f}); no harmonization applied.")
        pd.DataFrame({"noise_original": n0, "class": y}).to_csv(OUT_DIR / f"noise_harmonization_{name}.csv",
                                                                index=False)
        return X, {"dataset": name, "harmonized": False, "noise_target": np.nan,
                   "noise_auc_before": auc0, "noise_auc_after": auc0}
    target = float(np.percentile(n0, NOISE_PERCENTILE))
    rng = np.random.default_rng(SEED + 7)
    k = float(noise_abs(rng.normal(0.0, 1.0, (200, len(GRID)))).mean())   # fraction retained by noise_abs
    extra = np.sqrt(np.clip(target ** 2 - n0 ** 2, 0.0, None)) / k
    Xh = X + rng.normal(0.0, 1.0, X.shape) * extra[:, None]
    n1 = noise_abs(Xh)
    auc1 = roc_auc_score(y, n1)
    print(f"  {name}: noise harmonized to {target:.2e}. AUC of noise level for the class: "
          f"before {auc0:.3f}, after {auc1:.3f} (0.5 = no separation).")
    if abs(auc1 - 0.5) > 0.1:
        print(f"  WARNING {name}: after harmonization the noise level still separates the classes.")
    pd.DataFrame({"noise_original": n0, "noise_harmonized": n1, "class": y}).to_csv(
        OUT_DIR / f"noise_harmonization_{name}.csv", index=False)
    return Xh, {"dataset": name, "harmonized": True, "noise_target": target, "noise_auc_before": auc0,
                "noise_auc_after": auc1}


# =============================================================================
# DESCRIPTOR
# =============================================================================
def _crossing(x, y, j0, j1, level):
    if y[j1] == y[j0]:
        return x[j0]
    return x[j0] + (level - y[j0]) * (x[j1] - x[j0]) / (y[j1] - y[j0])


def _chord(x, y, edge=3):
    """Signal corrected with a local linear baseline (chord between the window ends)."""
    xa, ya = x[:edge].mean(), y[:edge].mean()
    xb, yb = x[-edge:].mean(), y[-edge:].mean()
    return y - (ya + (yb - ya) * (x - xa) / (xb - xa))


def _log_ratio(a, b):
    return np.log(a / b) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else np.nan


def band_geometry(x, y, d2, edge=3, noise=None):
    """Geometry of the dominant envelope of a region after a local linear (chord) baseline.
    Returns (area, position, FWHM, sharpness, asymmetry, flags)."""
    n = len(x)
    zone = max(edge, int(EDGE_FRAC * n))
    flags = dict.fromkeys(FLAGS, False)
    yc = _chord(x, y, edge)
    i = int(np.argmax(yc))
    h = yc[i]
    area = trapezoid(np.clip(yc, 0.0, None), x)
    if h <= 0 or (noise is not None and noise > 0 and h < SNR_INVALID * noise):
        flags["no_peak"] = True
        return area, np.nan, np.nan, np.nan, np.nan, flags
    if noise is not None and noise > 0 and h < SNR_MIN * noise:
        flags["low_snr"] = True
    if i < zone or i > n - 1 - zone:
        flags["peak_at_edge"] = True
        return area, np.nan, np.nan, np.nan, np.nan, flags
    peaks, _ = find_peaks(yc, height=0.5 * h, prominence=0.2 * h)
    if len(peaks) > 1:
        flags["multiple_peaks"] = True
    a, b, c = yc[i - 1], yc[i], yc[i + 1]
    den = a - 2 * b + c
    pos = x[i] + (0.5 * (a - c) / den if den != 0 else 0.0) * (x[1] - x[0])
    half = 0.5 * h
    j = i
    while j > 0 and yc[j] > half:
        j -= 1
    k = i
    while k < n - 1 and yc[k] > half:
        k += 1
    if yc[j] > half or yc[k] > half:
        flags["fwhm_truncated"] = True
        fwhm, asym = np.nan, np.nan
    else:
        xl = _crossing(x, yc, j, j + 1, half)
        xr = _crossing(x, yc, k - 1, k, half)
        fwhm = xr - xl
        if fwhm > FWHM_MAX_FRAC * (x[-1] - x[0]):
            flags["narrow_window"] = True
        left, right = pos - xl, xr - pos
        asym = np.log(right / left) if left > 0 and right > 0 else np.nan
    # The chord is linear, so its second derivative is zero and d2 equals the second derivative of the
    # baseline-corrected signal.
    sharp = -d2[i] / h
    return area, pos, fwhm, sharp, asym, flags


def band_state(flags):
    if any(flags[f] for f in FLAGS_INVALID):
        return "invalid"
    if any(flags[f] for f in FLAGS_DOUBTFUL):
        return "doubtful"
    return "valid"


ALL_BANDS = BANDS + BANDS_EXTRA


class BandDescriptor(BaseEstimator, TransformerMixin):
    """Stateless transformer: spectrum on GRID -> interpretable geometric variables.
    blocks: subset of BLOCKS; ("core",) gives the original 46-variable descriptor.
    exclude: bands excluded by quality control; d2_positions: (name, cm-1) pairs for the derivative block."""

    def __init__(self, sg_window=DESC_SG_WINDOW, sg_poly=SG_POLY, blocks=BLOCKS, exclude=(),
                 d2_positions=None, d2_tol=D2_TOL):
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.blocks = blocks
        self.exclude = exclude
        self.d2_positions = d2_positions
        self.d2_tol = d2_tol

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def _bands(self):
        cand = (BANDS if "core" in self.blocks else []) + (BANDS_EXTRA if "extra_bands" in self.blocks else [])
        return [b for b in cand if b[0] not in self.exclude]

    def _ratios(self):
        return [r for r in RATIOS if not any(nm in self.exclude for nm in r[1] + r[2])]

    def _positions(self):
        return list(self.d2_positions) if self.d2_positions is not None else list(D2_FIXED)

    def _band_ratios(self):
        avail = {b for b, _, _ in self._bands()}
        return [c for c in BAND_RATIOS if all(nm in avail for nm in c[1] + c[2])]

    def feature_names(self):
        bl, n = self.blocks, []
        if "core" in bl:
            n += [f"{b}__{m}" for b, _, _ in BANDS if b not in self.exclude for m in MEASURES]
            n += [r[0] for r in self._ratios()]
        if "extra_bands" in bl:
            n += [f"{b}__{m}" for b, _, _ in BANDS_EXTRA if b not in self.exclude for m in MEASURES]
        if "subbands" in bl:
            n += [f"{parent}__frac_{sb}" for parent, _, _, subs in SUBBANDS for sb, _, _ in subs]
        if "derivative" in bl:
            n += [f"{nm}__rel_AmideI" for nm, _ in self._positions()]
        if "extra_ratios" in bl:
            n += list(EXTRA_RATIOS) + [c[0] for c in self._band_ratios()]
        if "centroids" in bl:
            n += [c[0] for c in CENTROIDS]
        return n

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names(), dtype=object)

    def feature_spans(self):
        """Nominal spectral interval of each variable (None for ratios and centroids)."""
        bl, sp = self.blocks, []
        if "core" in bl:
            sp += [(lo, hi) for b, lo, hi in BANDS if b not in self.exclude for _ in MEASURES]
            sp += [None for _ in self._ratios()]
        if "extra_bands" in bl:
            sp += [(lo, hi) for b, lo, hi in BANDS_EXTRA if b not in self.exclude for _ in MEASURES]
        if "subbands" in bl:
            sp += [(slo, shi) for _, _, _, subs in SUBBANDS for _, slo, shi in subs]
        if "derivative" in bl:
            sp += [(c - self.d2_tol, c + self.d2_tol) for _, c in self._positions()]
        if "extra_ratios" in bl:
            sp += [None] * (len(EXTRA_RATIOS) + len(self._band_ratios()))
        if "centroids" in bl:
            sp += [None for _ in CENTROIDS]
        return sp

    @staticmethod
    def _d2_intensity(D2row, centre, tol):
        m = (GRID >= centre - tol) & (GRID <= centre + tol)
        return -float(D2row[m].min())

    def _compute(self, X):
        X = np.asarray(X, float)
        bl = self.blocks
        bands = self._bands()
        bnames = [b for b, _, _ in bands]
        core_names = {b for b, _, _ in BANDS}
        is_core = np.array([b in core_names for b in bnames], dtype=bool)
        ratios, positions = self._ratios(), self._positions()
        noise = noise_abs(X)
        S = savgol_filter(X, self.sg_window, self.sg_poly, axis=1)
        D2 = savgol_filter(X, self.sg_window, self.sg_poly, deriv=2, delta=STEP, axis=1)
        D2f = savgol_filter(X, D2F_WINDOW, self.sg_poly, deriv=2, delta=STEP, axis=1)
        masks = [(GRID >= lo) & (GRID <= hi) for _, lo, hi in bands]
        names = self.feature_names()
        out = np.full((X.shape[0], len(names)), np.nan)
        flags = np.zeros((X.shape[0], len(bands), len(FLAGS)), dtype=bool)
        states = np.empty((X.shape[0], len(bands)), dtype=object)
        for r in range(X.shape[0]):
            geo = [band_geometry(GRID[m], S[r, m], D2[r, m], noise=noise[r]) for m in masks]
            # Area policy: no_peak -> 0; other invalid states -> NaN (and relative areas undefined if the band
            # is a core band); valid/doubtful -> computed. The total uses only the core bands.
            areas = np.empty(len(bands))
            for bi, gg in enumerate(geo):
                states[r, bi] = band_state(gg[5])
                flags[r, bi] = [gg[5][f] for f in FLAGS]
                areas[bi] = 0.0 if gg[5]["no_peak"] else (np.nan if states[r, bi] == "invalid" else gg[0])
            tot = areas[is_core].sum()
            tot_ok = np.isfinite(tot) and tot > 0
            amap = dict(zip(bnames, areas))
            row = []
            for core_block in (True, False):
                if (core_block and "core" not in bl) or (not core_block and "extra_bands" not in bl):
                    continue
                for bi in np.where(is_core == core_block)[0]:
                    gg = geo[bi]
                    a = areas[bi] / tot if (tot_ok and np.isfinite(areas[bi])) else np.nan
                    row += [a, gg[1], gg[2], gg[3], gg[4]]
                if core_block:
                    for _, num, den in ratios:
                        row.append(_log_ratio(sum(amap[nm] for nm in num), sum(amap[nm] for nm in den)))
            if "subbands" in bl:
                for _, lo, hi, subs in SUBBANDS:
                    m = (GRID >= lo) & (GRID <= hi)
                    xs = GRID[m]
                    yc = np.clip(_chord(xs, S[r, m]), 0.0, None)
                    total = trapezoid(yc, xs)
                    for _, slo, shi in subs:
                        sm = (xs >= slo) & (xs <= shi)
                        row.append(trapezoid(yc[sm], xs[sm]) / total if total > 0 else np.nan)
            ref = self._d2_intensity(D2f[r], 0.5 * (D2_REF[0] + D2_REF[1]), 0.5 * (D2_REF[1] - D2_REF[0]))
            if "derivative" in bl:
                row += [self._d2_intensity(D2f[r], c, self.d2_tol) / ref if ref > 0 else np.nan
                        for _, c in positions]
            if "extra_ratios" in bl:
                lo, hi = SUBBANDS[0][1], SUBBANDS[0][2]
                m = (GRID >= lo) & (GRID <= hi)
                yc = _chord(GRID[m], S[r, m])

                def height(c):
                    return float(yc[np.abs(GRID[m] - c) <= 2.0].mean())

                row += [_log_ratio(height(1030.0), height(1080.0)),
                        _log_ratio(self._d2_intensity(D2f[r], 1630.0, D2_TOL), ref)]
                for _, num, den in self._band_ratios():
                    row.append(_log_ratio(sum(amap[nm] for nm in num), sum(amap[nm] for nm in den)))
            if "centroids" in bl:
                for _, lo, hi in CENTROIDS:
                    m = (GRID >= lo) & (GRID <= hi)
                    yc = np.clip(_chord(GRID[m], S[r, m]), 0.0, None)
                    sw = yc.sum()
                    row.append(float((GRID[m] * yc).sum() / sw) if sw > 0 else np.nan)
            out[r] = row
        return out, flags, states

    def transform(self, X):
        return self._compute(X)[0]

    def undefined_rel_area_fraction(self, X):
        _, flags, _ = self._compute(X)
        core_names = {b for b, _, _ in BANDS}
        ib = [i for i, (b, _, _) in enumerate(self._bands()) if b in core_names]
        idx = [FLAGS.index(f) for f in FLAGS_INVALID if f != "no_peak"]
        return float(flags[:, ib][:, :, idx].any(axis=(1, 2)).mean())

    def band_states(self, X):
        return pd.DataFrame(self._compute(X)[2], columns=[f"state__{b}" for b, _, _ in self._bands()])

    def quality(self, X):
        _, flags, _ = self._compute(X)
        inval = flags[:, :, [FLAGS.index(f) for f in FLAGS_INVALID]].any(axis=2)
        doubt = flags[:, :, [FLAGS.index(f) for f in FLAGS_DOUBTFUL]].any(axis=2) & ~inval
        bands = self._bands()
        q = pd.DataFrame(flags.mean(axis=0), index=[b for b, _, _ in bands], columns=FLAGS)
        q["valid"] = (~inval & ~doubt).mean(axis=0)
        q["doubtful"] = doubt.mean(axis=0)
        q["invalid"] = inval.mean(axis=0)
        return q


def assignment(pos):
    for lo, hi, lab in ASSIGNMENTS:
        if lo <= pos < hi:
            return lab
    return "unassigned"


def detect_d2_peaks(X_list):
    """Local minima of the second derivative of the pooled mean spectrum (mean of the dataset means;
    label-free). Excludes the amide I reference window and the edges of D2_WINDOWS."""
    mean_spec = np.mean([np.asarray(X, float).mean(axis=0) for X in X_list], axis=0)
    d2 = savgol_filter(mean_spec, D2F_WINDOW, SG_POLY, deriv=2, delta=STEP)
    rows = []
    for lo, hi in D2_WINDOWS:
        idx = np.where((GRID >= lo + D2_EDGE_MARGIN) & (GRID <= hi - D2_EDGE_MARGIN))[0]
        pk, props = find_peaks(-d2[idx], distance=max(1, int(round(D2_MIN_DIST / STEP))), prominence=0.0)
        rows += [(float(GRID[idx[k]]), float(p)) for k, p in zip(pk, props["prominences"])]
    df = pd.DataFrame(rows, columns=["position_cm1", "prominence"])
    df = df[~df.position_cm1.between(D2_REF[0], D2_REF[1])]
    df = df[df.prominence >= D2_PROM_REL * df.prominence.max()]
    df = df.nlargest(D2_MAX_PEAKS, "prominence").sort_values("position_cm1").reset_index(drop=True)
    df["name"] = [f"d2_{int(round(v))}" for v in df.position_cm1]
    df["tentative_assignment"] = [assignment(v) for v in df.position_cm1]
    df["relative_prominence"] = df.prominence / df.prominence.max()
    return df


def configure_descriptor(data):
    """Descriptor configuration shared by all datasets, decided without labels: bands excluded by quality
    control and second-derivative peak positions."""
    full = BandDescriptor()
    rows, exclude = [], set()
    for name, (X, *_rest) in data.items():
        for b, row in full.quality(X).iterrows():
            rows.append({"dataset": name, "band": b, "invalid_fraction": row["invalid"]})
            if row["invalid"] > QC_MAX_INVALID:
                exclude.add(b)
    excl = pd.DataFrame(rows)
    excl["excluded_from_descriptor"] = excl.band.isin(exclude)
    kw = {"exclude": tuple(sorted(exclude))}
    peaks = None
    if DETECT_PEAKS:
        peaks = detect_d2_peaks([d[0] for d in data.values()])
        kw["d2_positions"] = tuple(zip(peaks.name, peaks.position_cm1))
        kw["d2_tol"] = D2_TOL_DET
    return kw, excl, peaks


# =============================================================================
# REPRESENTATIONS AND CLASSIFIERS
# =============================================================================
def _window_mask(windows):
    mask = np.zeros_like(GRID, dtype=bool)
    for lo, hi in windows:
        mask |= (GRID >= lo) & (GRID <= hi)
    return mask


class Region(BaseEstimator, TransformerMixin):
    """Crops the spectrum to the given windows (default: BIO_WINDOWS)."""

    def __init__(self, windows=None):
        self.windows = windows

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, _window_mask(self.windows if self.windows is not None else BIO_WINDOWS)]


class SNV(BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        sd = X.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - X.mean(axis=1, keepdims=True)) / sd


class SGDeriv2(BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return savgol_filter(np.asarray(X, float), SG_WINDOW, SG_POLY, deriv=2, delta=STEP, axis=1)


class PLSDA(BaseEstimator, ClassifierMixin):
    """Binary PLS-DA; threshold at the midpoint of the training class means."""

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[1]).astype(float)
        nc = int(min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=nc, scale=False).fit(X, yb)
        s = self.pls_.predict(X).ravel()
        self.threshold_ = 0.5 * (s[yb == 1].mean() + s[yb == 0].mean())
        return self

    def decision_function(self, X):
        return self.pls_.predict(np.asarray(X, float)).ravel() - self.threshold_

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, self.classes_[1], self.classes_[0])


class Binning(BaseEstimator, TransformerMixin):
    """Mean of the (cropped) spectrum in n_bins contiguous intervals."""

    def __init__(self, n_bins=40):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        self.groups_ = np.array_split(np.arange(np.asarray(X).shape[1]), self.n_bins)
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.column_stack([X[:, gi].mean(axis=1) for gi in self.groups_])


class RandomIntervals(BaseEstimator, TransformerMixin):
    """Negative reference: n fixed-width intervals at random positions (fixed seed)."""

    def __init__(self, n=40, width=INTERVAL_WIDTH, random_state=SEED):
        self.n = n
        self.width = width
        self.random_state = random_state

    def fit(self, X, y=None):
        p = np.asarray(X).shape[1]
        rng = np.random.default_rng(self.random_state)
        self.starts_ = np.sort(rng.choice(p - self.width, size=self.n, replace=False))
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        return np.column_stack([X[:, s:s + self.width].mean(axis=1) for s in self.starts_])


class VIPSelect(BaseEstimator, TransformerMixin):
    """Selects the k wavenumbers with the highest PLS VIP score (fitted on training data)."""

    def __init__(self, k=40, n_components=5):
        self.k = k
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, float)
        nc = int(min(self.n_components, X.shape[1], X.shape[0] - 1))
        pls = PLSRegression(n_components=nc, scale=False).fit(X, np.asarray(y, float))
        t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
        ss = (t ** 2).sum(axis=0) * q.ravel() ** 2
        wn = w / np.linalg.norm(w, axis=0, keepdims=True)
        self.vip_ = np.sqrt(X.shape[1] * ((wn ** 2) @ ss) / ss.sum())
        self.support_ = np.sort(np.argsort(-self.vip_)[:min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.support_]


def _spa_chain(Xc, j0, K):
    P = Xc.copy()
    sel = [j0]
    for _ in range(K - 1):
        xs = P[:, sel[-1]].copy()
        nrm = xs @ xs
        if nrm <= 1e-12:
            break
        P = P - np.outer(xs, (xs @ P) / nrm)
        norms = (P ** 2).sum(axis=0)
        norms[sel] = -1.0
        sel.append(int(np.argmax(norms)))
    return sel


class SPASelect(BaseEstimator, TransformerMixin):
    """Successive projections algorithm for classification: chains from the n_starts variables with the
    highest Fisher ratio; start and number of variables chosen by LDA log-loss on a validation split of
    the training data."""

    def __init__(self, k_max=SPA_K_MAX, n_starts=SPA_N_STARTS, step=SPA_STEP, val_frac=0.3, random_state=SEED):
        self.k_max = k_max
        self.n_starts = n_starts
        self.step = step
        self.val_frac = val_frac
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        cols = np.arange(0, X.shape[1], self.step)
        Xd = X[:, cols]
        cal, val = next(StratifiedShuffleSplit(n_splits=1, test_size=self.val_frac,
                                               random_state=self.random_state).split(Xd, y))
        Xc = Xd[cal] - Xd[cal].mean(axis=0)
        y1, y0 = Xd[cal][y[cal] == 1], Xd[cal][y[cal] == 0]
        fisher = (y1.mean(0) - y0.mean(0)) ** 2 / (y1.var(0) + y0.var(0) + 1e-12)
        starts = np.argsort(-fisher)[:self.n_starts]
        K = int(max(1, min(self.k_max, len(cal) - 3, Xd.shape[1])))
        best = (np.inf, [int(starts[0])])
        for j0 in starts:
            chain = _spa_chain(Xc, int(j0), K)
            for k in range(1, len(chain) + 1):
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xd[cal][:, chain[:k]], y[cal])
                pr = np.clip(lda.predict_proba(Xd[val][:, chain[:k]]), 1e-6, 1 - 1e-6)
                cost = log_loss(y[val], pr, labels=lda.classes_)
                if cost < best[0]:
                    best = (cost, chain[:k])
        self.support_ = np.sort(cols[best[1]])
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.support_]


class LitPanel(BaseEstimator, TransformerMixin):
    """Literature band areas: spectrum normalized by its area in 900-1800 cm-1, chord baseline per window."""

    def __init__(self, bands=()):
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True

    def transform(self, X):
        X = np.asarray(X, float)
        mref = (GRID >= 900.0) & (GRID <= 1800.0)
        out = np.empty((X.shape[0], len(self.bands)))
        for r in range(X.shape[0]):
            x = X[r] - X[r, mref].min()
            norm = trapezoid(x[mref], GRID[mref])
            x = x / norm if norm > 0 else x
            for j, (c, hw) in enumerate(self.bands):
                m = (GRID >= c - hw) & (GRID <= c + hw)
                out[r, j] = trapezoid(np.clip(_chord(GRID[m], x[m]), 0.0, None), GRID[m])
        return out


def classifier_grids(extra):
    return [
        {**extra, "clf": [LogisticRegression(max_iter=5000, class_weight="balanced")],
         "clf__C": [0.01, 0.1, 1.0, 10.0]},
        {**extra, "clf": [SVC(kernel="rbf", class_weight="balanced")],
         "clf__C": [0.1, 1.0, 10.0, 100.0], "clf__gamma": ["scale", 0.01, 0.001]},
        {**extra, "clf": [RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=SEED, n_jobs=1)],
         "clf__max_depth": [None, 5], "clf__min_samples_leaf": [1, 3]},
    ]


def _enet():
    """Elastic-net logistic regression compatible with scikit-learn < 1.8 and >= 1.8."""
    major, minor = (int(v) for v in sklearn.__version__.split(".")[:2])
    kw = dict(solver="saga", l1_ratio=0.5, max_iter=5000, class_weight="balanced")
    if (major, minor) < (1, 8):
        kw["penalty"] = "elasticnet"
    return LogisticRegression(**kw)


def lr_grid(extra=None):
    """Logistic regression tuned in inner CV: L2 or elastic net, C in {0.01, 0.1, 1, 10}."""
    extra = extra or {}
    Cs = [0.01, 0.1, 1.0, 10.0]
    return [{**extra, "clf": [LogisticRegression(max_iter=5000, class_weight="balanced")], "clf__C": Cs},
            {**extra, "clf": [_enet()], "clf__C": Cs}]


def make_methods(n_samples, panel=None, desc_kw=None, n_budget=40):
    desc_kw = desc_kw or {}
    inner_train = 0.64 * n_samples
    vip_comps = [c for c in (3, 5, 10) if c <= 0.5 * inner_train] or [2]
    comps = [c for c in (5, 10, 20) if c <= 0.5 * inner_train] or [3]
    pls_comps = [c for c in (2, 5, 10, 15, 20) if c <= 0.5 * inner_train] or [2]
    desc_steps = lambda **kw: [("desc", BandDescriptor(**{**desc_kw, **kw})),
                               ("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    gb_grid = [{"clf": [HistGradientBoostingClassifier(max_iter=300, class_weight="balanced",
                                                       early_stopping=False, random_state=SEED)],
                "clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [7, 15],
                "clf__l2_regularization": [0.0, 1.0]}]
    m = {
        "Full_spectrum": ([("region", Region()), ("snv", SNV()), ("scaler", StandardScaler())],
                          classifier_grids({})),
        "PCA": ([("region", Region()), ("snv", SNV()), ("pca", PCA(random_state=SEED))],
                classifier_grids({"pca__n_components": comps})),
        "Second_derivative": ([("d2", SGDeriv2()), ("region", Region()), ("scaler", StandardScaler())],
                              classifier_grids({})),
        "PLS-DA": ([("region", Region()), ("snv", SNV())],
                   [{"clf": [PLSDA()], "clf__n_components": pls_comps}]),
        "Binning": ([("region", Region()), ("snv", SNV()), ("sel", Binning(n_bins=n_budget)),
                     ("scaler", StandardScaler())], lr_grid()),
        "Random_intervals": ([("region", Region()), ("snv", SNV()), ("sel", RandomIntervals(n=n_budget)),
                              ("scaler", StandardScaler())], lr_grid()),
        "VIP_topk": ([("region", Region()), ("snv", SNV()), ("sel", VIPSelect(k=n_budget)),
                      ("scaler", StandardScaler())], lr_grid({"sel__n_components": vip_comps})),
        "SPA_LDA": ([("region", Region()), ("snv", SNV()), ("sel", SPASelect())],
                    [{"clf": [LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")]}]),
        "Neg_control_raw": ([("region", Region(windows=NEG_CONTROL_WINDOWS)), ("snv", SNV()),
                             ("scaler", StandardScaler())], lr_grid()),
        "Neg_control_d2": ([("d2", SGDeriv2()), ("region", Region(windows=NEG_CONTROL_WINDOWS)),
                            ("scaler", StandardScaler())], classifier_grids({})),
        "Diamond_control_d2": ([("d2", SGDeriv2()), ("region", Region(windows=DIAMOND_WINDOWS)),
                                ("scaler", StandardScaler())], classifier_grids({})),
        "Descriptor_base_LR": (desc_steps(blocks=("core",)), lr_grid()),
        "Descriptor_LR": (desc_steps(), lr_grid()),
        "Descriptor_GB": ([("desc", BandDescriptor(**desc_kw))], gb_grid),   # native NaN handling
    }
    if panel in LITERATURE_PANELS:
        m["Literature_panel"] = ([("panel", LitPanel(bands=tuple(LITERATURE_PANELS[panel]))),
                                  ("scaler", StandardScaler())], lr_grid())
    if EXPLORATORY:
        m = {k: v for k, v in m.items() if k in EXPLORATORY_METHODS}
    return m


# =============================================================================
# UTILITIES
# =============================================================================
class SubjectStratifiedKFold:
    """K-fold over subjects, stratified by the subject's class: every fold receives a balanced share of the
    subjects of each class and all spectra of a subject stay together. Unlike StratifiedGroupKFold (which
    balances spectra, not subjects), it guarantees both classes in every fold whenever each class has at
    least n_splits subjects, and it does not depend on the scikit-learn version."""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y, groups):
        y, groups = np.asarray(y), np.asarray(groups)
        subj, first = np.unique(groups, return_index=True)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for s_tr, s_te in skf.split(subj, y[first]):
            te_mask = np.isin(groups, subj[s_te])
            yield np.where(~te_mask)[0], np.where(te_mask)[0]


def splitter(n_splits, seed, g=None):
    if g is not None:
        return SubjectStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def decision_scores(est, X):
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return est.predict_proba(X)[:, 1]


def estimate_perturbation(X, g):
    """Measured noise; scale and baseline variability estimated between replicates of the same subject."""
    rel = noise_rel(X)
    rel = rel[np.isfinite(rel)]
    noise = float(np.median(rel)) * PERTURB_FACTOR if len(rel) else 0.0
    sizes = pd.Series(g).value_counts() if g is not None else pd.Series(dtype=int)
    if g is None or (sizes > 1).sum() < 5:
        return {**PERTURB_DEFAULT, "noise_sd": noise}, "measured noise; scale and baseline = simulated scenario (no replicates)"
    t = (GRID - GRID.mean()) / (GRID.max() - GRID.min())
    A = np.vstack([np.ones_like(t), t]).T
    rows = []
    for s in sizes[sizes > 1].index:
        R = X[g == s]
        mu = R.mean(axis=0)
        amp = mu.max() - mu.min()
        if amp <= 0:
            continue
        for r in R:
            k = float((r @ mu) / (mu @ mu))
            if not np.isfinite(k) or k <= 0:
                continue
            coef, *_ = np.linalg.lstsq(A, r - k * mu, rcond=None)
            vals = (np.log(k), coef[0] / amp, coef[1] / amp)
            if np.all(np.isfinite(vals)):
                rows.append(vals)
    if len(rows) < 10:
        return {**PERTURB_DEFAULT, "noise_sd": noise}, "measured noise; scale and baseline = simulated scenario (too few replicates)"
    arr = np.array(rows) * PERTURB_FACTOR
    return ({"log_scale_sd": float(arr[:, 0].std(ddof=1)), "offset_sd": float(arr[:, 1].std(ddof=1)),
             "slope_sd": float(arr[:, 2].std(ddof=1)), "noise_sd": noise},
            f"all calibrated from replicate variability (x{PERTURB_FACTOR})")


def perturb(X, kind, rng, p):
    n = X.shape[0]
    amp = X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True)
    if kind == "scale":
        return X * np.exp(rng.normal(0.0, p["log_scale_sd"], (n, 1)))
    if kind == "baseline":
        t = (GRID - GRID.mean()) / (GRID.max() - GRID.min())
        return X + rng.normal(0.0, p["offset_sd"], (n, 1)) * amp + rng.normal(0.0, p["slope_sd"], (n, 1)) * amp * t
    if kind == "noise":
        return X + rng.normal(0.0, p["noise_sd"], X.shape) * amp
    raise ValueError(kind)


def feature_reliability(name, X, g, desc, n_rep=3):
    """Change of each variable when noise of the measured level is added, in units of between-subject SD."""
    names = desc.feature_names()
    F0 = desc.transform(X)
    sd = (pd.DataFrame(F0, columns=names).groupby(g).mean().std().to_numpy() if g is not None
          else np.nanstd(F0, axis=0))
    sd = np.where(sd > 0, sd, np.nan)
    lv = noise_rel(X)
    lv = float(np.median(lv[np.isfinite(lv)])) if np.isfinite(lv).any() else 0.0
    amp = (X.max(axis=1) - X.min(axis=1))[:, None]
    rng = np.random.default_rng(SEED)
    ch = np.mean([np.nanmean(np.abs(desc.transform(X + rng.normal(0, lv, X.shape) * amp) - F0), axis=0)
                  for _ in range(n_rep)], axis=0) / sd
    return pd.DataFrame({"dataset": name, "variable": names, "measured_relative_noise": lv,
                         "change_due_to_noise_in_SD": ch, "reliable": ch < RELIABILITY_MAX})


def bh_fdr(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return q
    order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.minimum(ranked, 1.0)
    q[ok] = tmp
    return q


# =============================================================================
# NESTED VALIDATION, ROBUSTNESS AND REGION IMPORTANCE
# =============================================================================
def check_splits(name, X, y, g, n_inner):
    """Checks every outer and inner split before training: both classes present and no subject on both sides."""
    def _check(yy, gg, tr, te, label):
        for part, idx in [("training", tr), ("test", te)]:
            if len(np.unique(yy[idx])) < 2:
                raise ValueError(f"{name}: {label}: the {part} set contains a single class.")
        if gg is not None and np.intersect1d(gg[tr], gg[te]).size:
            raise ValueError(f"{name}: {label}: subjects present in both training and test sets.")
    for r in range(N_REPEATS):
        for f, (tr, te) in enumerate(splitter(N_OUTER, SEED + r, g).split(X, y, g)):
            _check(y, g, tr, te, f"repeat {r + 1}, outer fold {f + 1}")
            gi = None if g is None else g[tr]
            for k, (itr, ite) in enumerate(splitter(n_inner, SEED + 100 + r, gi).split(X[tr], y[tr], gi)):
                _check(y[tr], gi, itr, ite, f"repeat {r + 1}, outer fold {f + 1}, inner fold {k + 1}")
    print(f"  {name}: outer and inner splits verified (both classes present, no shared subjects).")


def _importance_windows():
    return [(float(e0), float(min(e0 + IMP_WIDTH, hi))) for lo, hi in BIO_WINDOWS for e0 in np.arange(lo, hi, IMP_WIDTH)]


def importance_groups(est, windows, wn_region):
    """Column groups in the space seen by the classifier: (label, start, end, columns)."""
    groups = []
    desc = est.named_steps.get("desc")
    if desc is not None:
        centres = [None if sp is None else 0.5 * (sp[0] + sp[1]) for sp in desc.feature_spans()]
        for e0, e1 in windows:
            cols = [j for j, c in enumerate(centres) if c is not None and e0 <= c < e1]
            if cols:
                groups.append((f"{e0:.0f}-{e1:.0f}", e0, e1, np.array(cols)))
        no_pos = [j for j, c in enumerate(centres) if c is None]
        if no_pos:
            groups.append(("ratios_centroids", np.nan, np.nan, np.array(no_pos)))
    else:
        for e0, e1 in windows:
            cols = np.where((wn_region >= e0) & (wn_region < e1))[0]
            if len(cols):
                groups.append((f"{e0:.0f}-{e1:.0f}", e0, e1, cols))
    return groups


def grouped_importance(est, Xte, yte, rng, groups):
    """Drop in balanced accuracy when the columns of each group are jointly permuted across test subjects,
    in the input space of the classifier."""
    Z = np.asarray(est[:-1].transform(Xte), float)
    clf = est[-1]
    base = balanced_accuracy_score(yte, clf.predict(Z))
    rows = []
    for lab, e0, e1, cols in groups:
        drops = []
        for _ in range(IMP_REP):
            Zp = Z.copy()
            Zp[:, cols] = Z[rng.permutation(len(Z))][:, cols]
            drops.append(base - balanced_accuracy_score(yte, clf.predict(Zp)))
        rows.append((lab, e0, e1, len(cols), float(np.mean(drops))))
    return rows


def evaluate_dataset(name, cfg, X, y, g, pert, desc_kw, n_budget):
    n_inner = min(cfg.get("n_inner", N_INNER), N_INNER)
    check_splits(name, X, y, g, n_inner)
    methods = make_methods(len(np.unique(g)) if g is not None else len(y), cfg.get("literature_panel"),
                           desc_kw, n_budget)
    wn_region = GRID[_window_mask(BIO_WINDOWS)]
    windows = _importance_windows()
    rows, imp_rows = [], []
    for r in range(N_REPEATS):
        for f, (tr, te) in enumerate(splitter(N_OUTER, SEED + r, g).split(X, y, g)):
            rng = np.random.default_rng(SEED + 1000 * r + f)
            test_sets = [("clean", X[te])] + [(k, perturb(X[te], k, rng, pert)) for k in PERTURBATIONS]
            n_tr = len(np.unique(g[tr])) if g is not None else len(tr)
            n_te = len(np.unique(g[te])) if g is not None else len(te)
            for mname, (steps, grid) in methods.items():
                pipe = Pipeline(steps + [("clf", LogisticRegression())])
                gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy",
                                  cv=splitter(n_inner, SEED + 100 + r, g), n_jobs=-1, refit=True)
                gs.fit(X[tr], y[tr], groups=None if g is None else g[tr])
                est = gs.best_estimator_
                n_feat = est[:-1].transform(X[tr][:1]).shape[1]
                params = {k: v for k, v in gs.best_params_.items() if k != "clf"}
                if r == 0 and mname in IMPORTANCE_METHODS:
                    for lab, e0, e1, nc, d in grouped_importance(est, X[te], y[te], np.random.default_rng(SEED + f),
                                                                 importance_groups(est, windows, wn_region)):
                        imp_rows.append({"dataset": name, "method": mname, "fold": f, "group": lab,
                                         "start_cm1": e0, "end_cm1": e1, "centre_cm1": 0.5 * (e0 + e1),
                                         "n_variables": nc, "bal_acc_drop": d})
                sel = est.named_steps.get("sel")
                selection = (json.dumps([float(v) for v in wn_region[sel.support_]])
                             if sel is not None and hasattr(sel, "support_") else "")
                for cond, Xc in test_sets:
                    rows.append({"dataset": name, "method": mname, "selected_classifier": type(est[-1]).__name__,
                                 "repeat": r, "fold": f, "condition": cond,
                                 "bal_acc": balanced_accuracy_score(y[te], est.predict(Xc)),
                                 "auc": roc_auc_score(y[te], decision_scores(est, Xc)),
                                 "n_variables": n_feat, "n_train": n_tr, "n_test": n_te,
                                 "best_params": json.dumps(params, default=str), "selected_cm1": selection})
            print(f"  [{name}] repeat {r + 1}/{N_REPEATS}, fold {f + 1}/{N_OUTER} done", flush=True)
    return rows, imp_rows


def paired_comparisons(cv):
    """Paired fold-wise differences with the Nadeau-Bengio correction; non-inferiority and superiority."""
    rows = []
    for (ds, cond), df in cv.groupby(["dataset", "condition"]):
        ratio = (df["n_test"] / df["n_train"]).mean()
        for met in ["bal_acc", "auc"]:
            piv = df.pivot_table(index=["repeat", "fold"], columns="method", values=met)
            for focal in FOCAL:
                for ref in REFERENCES + COMPACT_REFERENCES + CONTROLS:
                    if focal not in piv or ref not in piv:
                        continue
                    d = (piv[focal] - piv[ref]).dropna().to_numpy()
                    n = len(d)
                    mean = d.mean() if n else np.nan
                    var = d.var(ddof=1) if n > 1 else np.nan
                    if n >= 2 and np.isfinite(var) and var > 0:
                        se = np.sqrt((1.0 / n + ratio) * var)
                        tcrit = student_t.ppf(0.975, n - 1)
                        lo, hi = mean - tcrit * se, mean + tcrit * se
                        t_stat = mean / se
                        p_diff = 2 * student_t.sf(abs(t_stat), n - 1)
                        p_ni = student_t.sf((mean + NI_MARGIN) / se, n - 1)
                        p_sup = student_t.sf(t_stat, n - 1)
                        ni, sup = bool(lo > -NI_MARGIN), bool(lo > 0)
                    else:
                        se = lo = hi = p_diff = p_ni = p_sup = np.nan
                        ni = sup = np.nan
                    is_ba = met == "bal_acc"
                    rows.append({"dataset": ds, "condition": cond, "metric": met, "method": focal, "reference": ref,
                                 "mean_difference": mean, "corrected_se": se, "ci95_low": lo, "ci95_high": hi,
                                 "p_difference": p_diff, "ni_margin": NI_MARGIN if is_ba else np.nan,
                                 "p_noninferiority": p_ni if is_ba else np.nan,
                                 "noninferior": ni if is_ba else np.nan,
                                 "p_superiority": p_sup, "superior": sup, "n_folds": n})
    return pd.DataFrame(rows)


def selection_stability(cv, n_budget, tol=SPA_STEP):
    """Mean pairwise Jaccard index between folds of the selected wavenumbers (match within tol cm-1)."""
    rows = []
    sub = cv[(cv.condition == "clean") & (cv.selected_cm1 != "")]
    for (ds, met), df in sub.groupby(["dataset", "method"]):
        sets = [np.array(json.loads(v)) for v in df.selected_cm1]
        jac = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                inter = sum(np.any(np.abs(sets[j] - v) < tol) for v in sets[i])
                union = len(sets[i]) + len(sets[j]) - inter
                jac.append(inter / union if union else np.nan)
        freq = pd.Series(np.round(np.concatenate(sets))).value_counts()
        rows.append({"dataset": ds, "method": met, "n_folds": len(sets),
                     "median_variables_per_fold": float(np.median([len(x) for x in sets])),
                     "mean_jaccard": float(np.nanmean(jac)) if jac else np.nan,
                     "min_jaccard": float(np.nanmin(jac)) if jac else np.nan,
                     "distinct_wavenumbers": int(len(freq)),
                     "most_frequent_cm1": json.dumps([float(v) for v in freq.index[:15]])})
    for ds in cv.dataset.unique():
        rows.append({"dataset": ds, "method": PRIMARY, "n_folds": np.nan, "median_variables_per_fold": float(n_budget),
                     "mean_jaccard": 1.0, "min_jaccard": 1.0, "distinct_wavenumbers": np.nan,
                     "most_frequent_cm1": "fixed by construction"})
    return pd.DataFrame(rows)


# =============================================================================
# PERMUTATION TEST (association between the descriptor and the class)
# =============================================================================
def permutation_test(name, F, y, g):
    base = Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))])
    fixed = list(splitter(N_OUTER, SEED, g).split(F, y, g))

    def cv_score(yy):
        splits = list(splitter(N_OUTER, SEED, g).split(F, yy, g)) if PERM_REFOLD else fixed
        return float(np.mean([balanced_accuracy_score(yy[te], clone(base).fit(F[tr], yy[tr]).predict(F[te]))
                              for tr, te in splits]))

    obs = cv_score(y)
    rng = np.random.default_rng(SEED)
    if g is not None:
        subj = np.unique(g)
        lab = np.array([y[g == s][0] for s in subj])
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        if g is None:
            yp = rng.permutation(y)
        else:
            mp = dict(zip(subj, rng.permutation(lab)))
            yp = np.array([mp[s] for s in g])
        null[i] = cv_score(yp)
    return {"dataset": name, "model": "Descriptor + LR (C=1), association test",
            "permuted_unit": "subject" if g is not None else "spectrum",
            "observed_bal_acc": obs, "null_mean": null.mean(), "null_sd": null.std(),
            "p_permutation": (1 + np.sum(null >= obs)) / (N_PERM + 1), "n_permutations": N_PERM}


# =============================================================================
# INTERPRETATION
# =============================================================================
def _cliff(a, b):
    return 2.0 * mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) - 1.0


def interpret(name, F, y, g, names):
    """Cliff's delta per variable (subject level when there are replicates), Mann-Whitney p, BH q and a
    stratified subject-level bootstrap standard error."""
    if g is not None:
        df = pd.DataFrame(F, columns=names)
        df["_g"], df["_y"] = g, y
        agg = df.groupby("_g").mean()
        F, y = agg[names].to_numpy(), agg["_y"].to_numpy().astype(int)
    rng = np.random.default_rng(SEED)
    rows = []
    for j, fn in enumerate(names):
        a, b = F[y == 1, j], F[y == 0, j]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 3 or len(b) < 3:
            d = p = se = np.nan
        else:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            d = _cliff(a, b)
            se = float(np.std([_cliff(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(N_BOOT)], ddof=1))
        rows.append({"dataset": name, "variable": fn, "cliff_delta": d, "bootstrap_se": se, "p": p,
                     "median_positive": np.median(a) if len(a) else np.nan,
                     "median_negative": np.median(b) if len(b) else np.nan,
                     "n_positive": len(a), "n_negative": len(b)})
    for r, qq in zip(rows, bh_fdr([r["p"] for r in rows])):
        r["q_BH"] = qq
    return rows


def compare_datasets(long_df, names, datasets):
    """Inverse-variance weighted heterogeneity Q of Cliff's deltas across the disease datasets. A variable is
    called specific only if heterogeneity is significant (BH q < ALPHA)."""
    D = long_df.pivot(index="variable", columns="dataset", values="cliff_delta").reindex(index=names, columns=datasets)
    Q = long_df.pivot(index="variable", columns="dataset", values="q_BH").reindex(index=names, columns=datasets)
    E = long_df.pivot(index="variable", columns="dataset", values="bootstrap_se").reindex(index=names, columns=datasets)
    out = []
    for v in names:
        row = {"variable": v}
        d, e = D.loc[v].to_numpy(float), E.loc[v].to_numpy(float)
        ok = np.isfinite(d) & np.isfinite(e) & (e > 0)
        if len(datasets) > 1 and ok.sum() >= 2:
            w = 1.0 / e[ok] ** 2
            dbar = np.sum(w * d[ok]) / np.sum(w)
            qstat = float(np.sum(w * (d[ok] - dbar) ** 2))
            row.update({"Q_heterogeneity": qstat, "df": int(ok.sum() - 1),
                        "p_heterogeneity": float(chi2.sf(qstat, ok.sum() - 1)), "pooled_delta": dbar})
        else:
            row.update({"Q_heterogeneity": np.nan, "df": np.nan, "p_heterogeneity": np.nan, "pooled_delta": np.nan})
        for p in datasets:
            row[f"delta_{p}"], row[f"se_{p}"], row[f"q_{p}"] = D.loc[v, p], E.loc[v, p], Q.loc[v, p]
        out.append(row)
    res = pd.DataFrame(out)
    res["q_heterogeneity"] = bh_fdr(res["p_heterogeneity"].to_numpy())
    cats = []
    for _, r in res.iterrows():
        sig = pd.Series({p: (pd.notna(r[f"q_{p}"]) and r[f"q_{p}"] < ALPHA) for p in datasets})
        k = int(sig.sum())
        if len(datasets) == 1:
            cats.append("no difference" if k == 0 else ("significant (+)" if r[f"delta_{datasets[0]}"] > 0
                                                        else "significant (-)"))
            continue
        het = pd.notna(r["q_heterogeneity"]) and r["q_heterogeneity"] < ALPHA
        signs = np.sign([r[f"delta_{p}"] for p in datasets])
        if k == 0:
            cats.append("no difference")
        elif not het:
            cats.append("shared (no heterogeneity)" if k == len(datasets) and len(set(signs)) == 1
                        else "no evidence of heterogeneity (specificity not supported)")
        elif k == 1:
            cats.append(f"specific: {sig[sig].index[0]} (significant heterogeneity)")
        else:
            cats.append("heterogeneous (magnitude or direction differs)")
    res.insert(1, "category", cats)
    return res, D, Q


# =============================================================================
# FIGURES
# =============================================================================
def _pretty(s):
    return s.replace("_", " ")


def fig_spectra(data, desc):
    names = list(data.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 3.0 * len(names)), sharex=True, squeeze=False)
    snv = SNV()
    for ax, ds in zip(axes[:, 0], names):
        X, y, _, pos, neg = data[ds]
        Xs = snv.transform(X)
        for i, (_, lo, hi) in enumerate(desc._bands()):
            ax.axvspan(lo, hi, color="0.88" if i % 2 == 0 else "0.80", lw=0)
        if DIAMOND_ATR:
            ax.axvspan(*DIAMOND_REGION, color="#f3d9d9", lw=0, label="diamond region (excluded)")
        ax.plot(GRID, Xs[y == 0].mean(0), lw=1.1, label=neg)
        ax.plot(GRID, Xs[y == 1].mean(0), lw=1.1, label=pos)
        ax.set_title(_pretty(ds), loc="left", fontsize=10)
        ax.set_ylabel("Absorbance (SNV)")
        ax.legend(frameon=False, fontsize=8, ncol=3)
    axes[-1, 0].set_xlim(WN_MAX, WN_MIN)
    axes[-1, 0].set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_spectra_bands.png", dpi=300)
    plt.close(fig)


def fig_performance(cv):
    names = [n for n in DATASETS if n in set(cv.dataset)]
    mets = [m for m in METHOD_ORDER if m in cv.method.unique()]
    clean = cv[cv.condition == "clean"]
    has_pert = len(PERTURBATIONS) > 0
    fig, axes = plt.subplots(1, 2 if has_pert else 1, figsize=(15 if has_pert else 11, 5), squeeze=False)
    a1 = axes[0, 0]
    w = 0.85 / len(mets)
    x = np.arange(len(names))
    cmap = plt.get_cmap("tab20")
    for i, mt in enumerate(mets):
        s = clean[clean.method == mt].groupby("dataset")["bal_acc"]
        a1.bar(x + i * w - 0.425 + w / 2, s.mean().reindex(names), w, yerr=s.std().reindex(names), capsize=1.5,
               color=cmap(i % 20), label=_pretty(mt), hatch="//" if mt in CONTROLS else None)
    a1.set_xticks(x, [_pretty(n) for n in names], fontsize=8)
    a1.set_ylabel("Balanced accuracy")
    a1.set_ylim(0.3, 1.0)
    a1.axhline(0.5, color="k", lw=0.6, ls=":")
    a1.legend(frameon=False, fontsize=6.5, ncol=2)
    a1.set_title("A. Nested cross-validation (hatched: negative controls)", loc="left", fontsize=10)
    if has_pert:
        a2 = axes[0, 1]
        piv = cv.pivot_table(index=["dataset", "method", "repeat", "fold"], columns="condition",
                             values="bal_acc").reset_index()
        xp = np.arange(len(PERTURBATIONS))
        for i, mt in enumerate(mets):
            s = piv[piv.method == mt]
            a2.bar(xp + i * w - 0.425 + w / 2, [(s["clean"] - s[k]).mean() for k in PERTURBATIONS], w,
                   color=cmap(i % 20), label=_pretty(mt))
        a2.set_xticks(xp, PERTURBATIONS)
        a2.set_ylabel("Drop in balanced accuracy")
        a2.axhline(0, color="k", lw=0.6)
        a2.set_title("B. Robustness to perturbations (mean over datasets)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_performance_robustness.png", dpi=300)
    plt.close(fig)


def fig_effects(D, Q):
    fig, ax = plt.subplots(figsize=(1.7 * D.shape[1] + 3.5, 0.2 * D.shape[0] + 1.6))
    im = ax.imshow(D.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(D.shape[1]), [_pretty(c) for c in D.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(D.shape[0]), D.index, fontsize=6)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            q = Q.iloc[i, j]
            if pd.notna(q) and q < ALPHA:
                ax.text(j, i, "*", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Cliff's delta (positive > negative class)")
    ax.set_title(f"* q < {ALPHA} (Benjamini-Hochberg)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_effect_map.png", dpi=300)
    plt.close(fig)


def fig_noninferiority(comp):
    s = comp[(comp.method == PRIMARY) & (comp.metric == "bal_acc") & (comp.condition == "clean")
             & comp.reference.isin(REFERENCES) & np.isfinite(comp.ci95_low)].reset_index(drop=True)
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 2.0))
    yy = np.arange(len(s))[::-1]
    ax.errorbar(s.mean_difference, yy, xerr=[s.mean_difference - s.ci95_low, s.ci95_high - s.mean_difference],
                fmt="o", color="k", capsize=3)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvline(-NI_MARGIN, color="r", lw=0.8, ls="--", label=f"non-inferiority margin (-{NI_MARGIN})")
    ax.set_yticks(yy, [f"{_pretty(p)} | vs {_pretty(r)}" for p, r in zip(s.dataset, s.reference)], fontsize=8)
    ax.set_xlabel(f"Difference in balanced accuracy ({_pretty(PRIMARY)} - reference)\n"
                  "95% CI, Nadeau-Bengio corrected")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_noninferiority.png", dpi=300)
    plt.close(fig)


def fig_importance(imp_s, desc, peaks):
    names = [n for n in DATASETS if n in set(imp_s.dataset)]
    fig, axes = plt.subplots(len(names), len(BIO_WINDOWS), figsize=(14, 3.4 * len(names)), squeeze=False,
                             gridspec_kw={"width_ratios": [hi - lo for lo, hi in BIO_WINDOWS]})
    for row, ds in zip(axes, names):
        d = imp_s[imp_s.dataset == ds]
        for ax, (lo, hi) in zip(row, BIO_WINDOWS):
            for i, (_, blo, bhi) in enumerate(desc._bands()):
                if bhi >= lo and blo <= hi:
                    ax.axvspan(blo, bhi, color="0.88" if i % 2 == 0 else "0.80", lw=0)
            if peaks is not None:
                for v in peaks.position_cm1[(peaks.position_cm1 >= lo) & (peaks.position_cm1 <= hi)]:
                    ax.axvline(v, color="0.55", lw=0.5, ls=":")
            for met in [m for m in IMPORTANCE_METHODS if m in d.method.unique()]:
                dm = d[(d.method == met) & (d.centre_cm1 >= lo) & (d.centre_cm1 <= hi)].sort_values("centre_cm1")
                is_desc = met.startswith("Descriptor")
                ax.plot(dm.centre_cm1, dm.mean_drop, marker="o", ms=4 if is_desc else 3, lw=1.1,
                        ls="none" if is_desc else "-", label=_pretty(met))
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xlim(hi, lo)
        row[0].set_ylabel("Drop in balanced accuracy")
        row[0].set_title(f"{_pretty(ds)}: grouped permutation importance, {IMP_WIDTH:.0f} cm$^{{-1}}$ blocks "
                         "(shaded: descriptor bands; dotted: detected peaks)", loc="left", fontsize=9)
        row[0].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_region_importance.png", dpi=300)
    plt.close(fig)


def fig_audit(summ, noise_rows):
    """Acquisition audit: negative controls vs. the primary model and the full spectrum, per dataset."""
    cl = summ[summ.condition == "clean"]
    names = [n for n in DATASETS if n in set(cl.dataset)]
    show = [m for m in ["Full_spectrum", PRIMARY, "Neg_control_raw", "Neg_control_d2", "Diamond_control_d2"]
            if m in cl.method.unique()]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(show)
    x = np.arange(len(names))
    colors = {"Full_spectrum": "#4c72b0", PRIMARY: "#55a868", "Neg_control_raw": "#c44e52",
              "Neg_control_d2": "#dd8452", "Diamond_control_d2": "#8172b3"}
    for i, mt in enumerate(show):
        s = cl[cl.method == mt].set_index("dataset").reindex(names)
        ax.bar(x + i * w - 0.4 + w / 2, s.bal_acc_mean, w, yerr=s.bal_acc_sd, capsize=2,
               color=colors.get(mt), hatch="//" if mt in CONTROLS else None, label=_pretty(mt))
    nz = {r["dataset"]: r for r in noise_rows}
    ax.set_xticks(x, [f"{_pretty(n)}\nnoise AUC {nz[n]['noise_auc_before']:.2f} -> {nz[n]['noise_auc_after']:.2f}"
                      if n in nz else _pretty(n) for n in names], fontsize=8)
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Acquisition audit: models restricted to regions without sample bands should stay at chance",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_acquisition_audit.png", dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    if N_OUTER < 2 or N_INNER < 2 or N_REPEATS < 1 or (N_PERM < 1 and not EXPLORATORY):
        raise ValueError("N_OUTER and N_INNER must be >= 2; N_REPEATS and N_PERM >= 1.")
    mode = "EXPLORATORY" if EXPLORATORY else ("QUICK TEST" if QUICK else "FULL")
    print(f"Mode: {mode} | nested CV {N_OUTER}x{N_INNER}x{N_REPEATS} | permutations {N_PERM} | "
          f"nested validation for: {', '.join(RUN_CV)}")
    print(f"Non-inferiority margin = {NI_MARGIN} (balanced accuracy), two-sided 95% CI.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    data, pert_rows, perts, noise_rows = {}, [], {}, []
    for name, cfg in DATASETS.items():
        gcol = cfg.get("group")
        wn, X, y_raw, g = load_dataset(cfg["path"], gcol)
        X = resample(wn, X)
        y, pos, neg = binarize(y_raw, cfg["positive"], name)
        if g is not None and pd.Series(g).value_counts().max() == 1:
            g = None                                   # one spectrum per subject: plain stratified folds
        check_groups(y, g, name)
        check_sample_size(y, g, name, min(cfg.get("n_inner", N_INNER), N_INNER))
        nsubj = len(np.unique(g)) if g is not None else len(y)
        print(f"{name}: {len(y)} spectra, {nsubj} subjects | {pos}={int(y.sum())}, {neg}={int((1 - y).sum())} spectra")
        if HARMONIZE_NOISE:
            X, nrow = harmonize_noise(name, X, y)
            noise_rows.append(nrow)
        data[name] = (X, y, g, pos, neg)
        perts[name], origin = estimate_perturbation(X, g)
        pert_rows.append({"dataset": name, "origin": origin, **perts[name]})
    pd.DataFrame(pert_rows).to_csv(OUT_DIR / "perturbation_magnitudes.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(OUT_DIR / "noise_harmonization_summary.csv", index=False)

    desc_kw, excl, peaks = configure_descriptor(data)
    excl.to_csv(OUT_DIR / "excluded_bands.csv", index=False)
    desc = BandDescriptor(**desc_kw)
    names = desc.feature_names()
    n_budget = len(names)
    with open(OUT_DIR / "descriptor_configuration.json", "w") as fh:
        json.dump({"version": DESCRIPTOR_VERSION, "n_variables": n_budget, "excluded_bands": list(desc_kw["exclude"]),
                   "bio_windows": BIO_WINDOWS, "variables": names}, fh, indent=1)
    print(f"Descriptor v{DESCRIPTOR_VERSION}: {n_budget} variables "
          f"({len(BandDescriptor(blocks=('core',), exclude=desc_kw['exclude']).feature_names())} in the core block).")
    print(f"  Bands excluded by quality control (invalid in >{QC_MAX_INVALID:.0%} of the spectra of any dataset): "
          f"{', '.join(desc_kw['exclude']) or 'none'}")
    if peaks is not None:
        peaks.to_csv(OUT_DIR / "detected_peaks.csv", index=False)
        print(f"  Second-derivative peaks detected in the pooled mean spectrum: {len(peaks)}")

    interp_rows, perm_rows, qc_rows, rel_rows, glob_rows = [], [], [], [], []
    for name, (X, y, g, pos, neg) in data.items():
        F = desc.transform(X)
        out = pd.DataFrame(F, columns=names)
        out.insert(0, "class", np.where(y == 1, pos, neg))
        if g is not None:
            out.insert(1, "subject", g)
        pd.concat([out, desc.band_states(X)], axis=1).to_csv(OUT_DIR / f"descriptors_{name}.csv", index=False)
        qc = desc.quality(X)
        qc.insert(0, "band", qc.index)
        qc.insert(0, "dataset", name)
        qc_rows.append(qc)
        glob_rows.append({"dataset": name, "spectra": len(y),
                          "fraction_undefined_rel_area": desc.undefined_rel_area_fraction(X)})
        rel = feature_reliability(name, X, g, desc)
        rel_rows.append(rel)
        nf = rel[~rel.reliable]
        print(f"  {name}: variables sensitive to the measured noise (> {RELIABILITY_MAX} SD): "
              f"{', '.join(nf.variable) if len(nf) else 'none'}")
        interp_rows += interpret(name, F, y, g, names)
        if N_PERM > 0:
            print(f"Permutation test: {name}", flush=True)
            perm_rows.append(permutation_test(name, F, y, g))

    pd.concat(qc_rows).to_csv(OUT_DIR / "band_quality.csv", index=False)
    pd.DataFrame(glob_rows).to_csv(OUT_DIR / "global_quality.csv", index=False)
    pd.concat(rel_rows).to_csv(OUT_DIR / "feature_reliability.csv", index=False)
    long_df = pd.DataFrame(interp_rows)
    long_df.to_csv(OUT_DIR / "feature_effects.csv", index=False)
    disease = [n for n in DATASETS if DATASETS[n].get("role") == "disease"]
    if disease:
        cross, _, _ = compare_datasets(long_df, names, disease)
        cross.to_csv(OUT_DIR / "cross_dataset_effects.csv", index=False)
    _, D, Q = compare_datasets(long_df, names, list(DATASETS))
    pd.DataFrame(perm_rows).to_csv(OUT_DIR / "permutation_test.csv", index=False)
    fig_spectra(data, desc)
    fig_effects(D, Q)

    cv_rows, imp_rows = [], []
    parts = OUT_DIR / "cv_parts"
    parts.mkdir(exist_ok=True)
    for name in RUN_CV:
        X, y, g, _, _ = data[name]
        f_cv, f_imp = parts / f"{name}_cv.csv", parts / f"{name}_importance.csv"
        if f_cv.exists() and os.environ.get("FTIR_RESUME", "1") == "1":
            print(f"Nested validation: {name} already finished; reusing {f_cv.name} (FTIR_RESUME=0 to recompute)")
            cv_rows += pd.read_csv(f_cv, keep_default_na=False, na_values=[""]).to_dict("records")
            if f_imp.exists() and f_imp.stat().st_size > 1:
                imp_rows += pd.read_csv(f_imp).to_dict("records")
            continue
        print(f"Nested validation: {name}", flush=True)
        r_, i_ = evaluate_dataset(name, DATASETS[name], X, y, g, perts[name], desc_kw, n_budget)
        pd.DataFrame(r_).to_csv(f_cv, index=False)
        pd.DataFrame(i_).to_csv(f_imp, index=False)
        cv_rows += r_
        imp_rows += i_
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT_DIR / "cv_details.csv", index=False)
    imp = pd.DataFrame(imp_rows)
    if len(imp):
        imp_s = (imp.groupby(["dataset", "method", "group"], dropna=False)
                    .agg(start_cm1=("start_cm1", "first"), end_cm1=("end_cm1", "first"),
                         centre_cm1=("centre_cm1", "first"), n_variables=("n_variables", "first"),
                         mean_drop=("bal_acc_drop", "mean"), sd_drop=("bal_acc_drop", "std")).reset_index())
        imp_s.to_csv(OUT_DIR / "region_importance.csv", index=False)
        fig_importance(imp_s, desc, peaks)
    summ = (cv.groupby(["dataset", "method", "condition"])
              .agg(bal_acc_mean=("bal_acc", "mean"), bal_acc_sd=("bal_acc", "std"),
                   auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                   median_n_variables=("n_variables", "median"),
                   most_selected_classifier=("selected_classifier", lambda s: s.value_counts().index[0]))
              .reset_index())
    summ.to_csv(OUT_DIR / "cv_summary.csv", index=False)
    comp = paired_comparisons(cv)
    comp.to_csv(OUT_DIR / "paired_comparisons.csv", index=False)
    stab = selection_stability(cv, n_budget)
    stab.to_csv(OUT_DIR / "selection_stability.csv", index=False)
    gl = []
    for (ds, cond), df_ in comp[(comp.method == PRIMARY) & (comp.metric == "bal_acc")].groupby(["dataset", "condition"]):
        v = df_.set_index("reference").reindex(REFERENCES)
        gl.append({"dataset": ds, "condition": cond, "expected_references": len(REFERENCES),
                   "references_with_uncertainty": int(v["noninferior"].notna().sum()),
                   "noninferior_to_all": bool((v["noninferior"] == True).all()),
                   "max_p_noninferiority": v["p_noninferiority"].max(),
                   "superior_to_all": bool((v["superior"] == True).all())})
    glob = pd.DataFrame(gl)
    glob.to_csv(OUT_DIR / "noninferiority_global.csv", index=False)
    fig_performance(cv)
    fig_noninferiority(comp)
    fig_audit(summ, noise_rows)

    print("\nPerformance (clean condition, nested cross-validation):")
    cl = summ[summ.condition == "clean"].copy()
    cl["method"] = pd.Categorical(cl["method"], METHOD_ORDER, ordered=True)
    print(cl.sort_values(["dataset", "method"])[["dataset", "method", "bal_acc_mean", "bal_acc_sd", "auc_mean",
          "median_n_variables", "most_selected_classifier"]].round(3).to_string(index=False))
    sel = comp[(comp.method == PRIMARY) & (comp.metric == "bal_acc") & (comp.condition == "clean")]
    cols = ["dataset", "reference", "mean_difference", "ci95_low", "ci95_high"]
    print(f"\nNon-inferiority of the primary model ({PRIMARY}) vs. full-spectrum references, margin {NI_MARGIN}:")
    print(sel[sel.reference.isin(REFERENCES)][cols + ["p_noninferiority", "noninferior"]].round(4).to_string(index=False))
    print(f"\n{PRIMARY} vs. equal-budget compact comparators:")
    print(sel[sel.reference.isin(COMPACT_REFERENCES)][cols + ["p_superiority", "superior"]].round(4).to_string(index=False))
    print("\nAcquisition audit (balanced accuracy of models restricted to regions without sample bands):")
    for name in RUN_CV:
        vals = {m: cl[(cl.dataset == name) & (cl.method == m)].bal_acc_mean for m in CONTROLS}
        txt = ", ".join(f"{_pretty(m)} = {float(v.iloc[0]):.3f}" for m, v in vals.items() if len(v))
        worst = max([float(v.iloc[0]) for v in vals.values() if len(v)] or [np.nan])
        verdict = ("ACQUISITION DIFFERENCES BETWEEN CLASSES: interpret with caution" if worst >= 0.65
                   else "close to chance: no evidence of acquisition differences")
        print(f"  {name}: {txt} -> {verdict}")
    print("\nSelection stability across folds (Jaccard):")
    print(stab[["dataset", "method", "median_variables_per_fold", "mean_jaccard", "distinct_wavenumbers"]]
          .round(3).to_string(index=False))
    if perm_rows:
        print("\nPermutation test (descriptor + LR association with the class):")
        print(pd.DataFrame(perm_rows)[["dataset", "observed_bal_acc", "null_mean", "p_permutation"]]
              .round(4).to_string(index=False))
    if len(imp):
        print(f"\nMost important regions (mean drop in balanced accuracy, {IMP_WIDTH:.0f} cm-1 groups):")
        for (ds, met), d in imp_s.groupby(["dataset", "method"]):
            top = d.nlargest(6, "mean_drop")
            print(f"  {ds} | {met}: " + ", ".join(f"{a} ({c:.3f})" for a, c in zip(top.group, top.mean_drop)))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
