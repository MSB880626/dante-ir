#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Writes a natural-language report (plain text, ready to paste into Word) from the outputs of
ftir_descriptor.py, plus a glossary of all descriptor variables. Nothing is recomputed.

Usage:
  python make_report.py                          (reads results_covid and results_t2d if they exist,
                                                  otherwise results_ftir_descriptor)
  python make_report.py results_covid results_t2d
Outputs (in the current folder):
  results_report.txt     narrative report of all results
  variable_glossary.csv  meaning of each of the descriptor variables (supplementary table)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY, SECONDARY, BASE = "Descriptor_GB", "Descriptor_LR", "Descriptor_base_LR"
REFERENCES = ["Full_spectrum", "PCA", "Second_derivative", "PLS-DA"]
COMPACT = ["Binning", "Random_intervals", "VIP_topk", "SPA_LDA", "Literature_panel"]
CONTROLS = ["Neg_control_raw", "Neg_control_d2", "Diamond_control_d2"]
NI_MARGIN = 0.03
CHANCE_LIMIT = 0.65
DATASET_INFO = {
    "T2D_saliva": ("type 2 diabetes (T2D) vs. control saliva", "T2D", "controls"),
    "COVID_original": ("COVID-19 PCR-positive vs. negative saliva as in the original study, where the enrolment "
                       "group is almost completely confounded with the collection site", "COVID-positive",
                       "COVID-negative"),
    "COVID_NSW_PCR": ("follow-up PCR positive vs. negative on the day of collection, within a single site (NSW); "
                      "the only COVID-19 comparison free of site confounding", "PCR-positive", "PCR-negative"),
    "COVID_site_NSW_vs_TPCH": ("COVID-positive subjects from two collection sites (TPCH vs. NSW); an audit "
                               "comparison whose label is the site, not the disease", "TPCH", "NSW"),
}
METHOD_NAMES = {
    "Full_spectrum": "full spectrum", "PCA": "PCA", "Second_derivative": "second-derivative spectrum",
    "PLS-DA": "PLS-DA", "Binning": "equal-width binning", "Random_intervals": "random intervals",
    "VIP_topk": "VIP-selected wavenumbers", "SPA_LDA": "SPA-LDA", "Literature_panel": "literature band panel",
    "Neg_control_raw": "raw negative-control region", "Neg_control_d2": "second-derivative negative-control region",
    "Diamond_control_d2": "diamond-region control", PRIMARY: "descriptor with gradient boosting (primary)",
    SECONDARY: "descriptor with logistic regression", BASE: "core 46-variable descriptor (ablation)",
}

BAND_MEANING = {
    "CH3as_lipids": "asymmetric CH3 stretching of lipids (~2960 cm-1)",
    "CH2as_lipids": "asymmetric CH2 stretching of lipid chains (~2920 cm-1)",
    "CH2s_lipids": "symmetric CH2 stretching of lipid chains (~2850 cm-1)",
    "Amide_I": "amide I band of proteins (C=O of the peptide bond, ~1650 cm-1)",
    "Amide_II": "amide II band of proteins (N-H bending / C-N stretching, ~1545 cm-1)",
    "COO_sym": "symmetric stretching of carboxylate groups (fatty acids, amino acids; ~1400 cm-1)",
    "AmideIII_PO2as": "amide III of proteins and asymmetric phosphate stretching (~1240 cm-1)",
    "PO2s_CO_carbohydrates": "symmetric phosphate stretching and C-O of carbohydrates (~1080 cm-1)",
    "Carbonyl_CO": "C=O of carboxylic acids and esters (~1715-1740 cm-1)",
    "CH2CH3_bending": "CH2/CH3 bending of lipids and protein side chains (~1455 cm-1)",
    "AmideA_OH": "amide A of proteins (N-H) and O-H of water (~3290-3400 cm-1)",
}
MEASURE_MEANING = {
    "rel_area": "relative area (amount relative to the total of the core bands)",
    "position": "peak position (shifts indicate a change in the chemical environment)",
    "fwhm": "full width at half maximum (heterogeneity of the band components)",
    "sharpness": "sharpness (curvature at the maximum relative to height; changes when overlapping components change)",
    "asymmetry": "asymmetry (log ratio of right to left half-widths; reveals hidden shoulder components)",
}
RATIO_MEANING = {
    "log_AmideI_AmideII": "log ratio amide I / amide II (protein secondary structure and composition)",
    "log_Lipids_AmideI": "log ratio lipids / proteins",
    "log_CH3_CH2": "log ratio CH3 / CH2 (lipid chain length and branching)",
    "log_COO_AmideI": "log ratio carboxylates / proteins",
    "log_PO2as_AmideI": "log ratio phosphates (amide III region) / proteins",
    "log_PO2sCOcarb_AmideI": "log ratio phosphates and carbohydrates / proteins",
    "log_I1030_I1080": "log ratio of absorbance at 1030 / 1080 cm-1 (glycogen-glucose relative to phosphate)",
    "log_beta_alpha_d2": "log ratio of beta-sheet (1630 cm-1) to the amide I maximum, second derivative",
    "log_Carbonyl_lipidsCH": "log ratio carbonyl / lipid CH stretching",
    "log_AmideAOH_AmideI": "log ratio amide A-OH / amide I (hydration relative to protein)",
}
SUBBAND_MEANING = {
    "CO_glycogen_1030": "fraction of the carbohydrate region at 995-1055 cm-1 (C-O of glucose/glycogen)",
    "PO2s_CO_1080": "fraction of the carbohydrate region at 1055-1095 cm-1 (symmetric phosphate, C-O)",
    "CO_COH_1110": "fraction of the carbohydrate region at 1095-1130 cm-1 (C-O / C-O-H of carbohydrates)",
    "COC_1150": "fraction of the carbohydrate region at 1130-1185 cm-1 (C-O-C of glycogen and glycoproteins)",
}
CENTROID_MEANING = {
    "centroid_carbohydrates": "centre of mass of the carbohydrate region (995-1185 cm-1)",
    "centroid_AmideI": "centre of mass of the amide I band (1600-1700 cm-1)",
    "centroid_AmideA_OH": "centre of mass of the amide A / O-H band (3050-3650 cm-1)",
}
ASSIGNMENTS = [
    (900, 990, "PO4 / ribose, C-C"), (990, 1060, "C-O of carbohydrates (glucose/glycogen)"),
    (1060, 1095, "symmetric PO2- / C-O"), (1095, 1130, "C-O / C-O-H of carbohydrates"),
    (1130, 1185, "C-O-C of glycogen / glycoproteins"), (1185, 1290, "asymmetric PO2- / amide III"),
    (1290, 1360, "amide III / CH2 wagging"), (1360, 1430, "symmetric COO- / CH3"),
    (1430, 1480, "CH2/CH3 bending"), (1480, 1525, "tyrosine / aromatic ring"), (1525, 1600, "amide II"),
    (1600, 1640, "amide I: beta sheet"), (1640, 1665, "amide I: alpha helix / random coil"),
    (1665, 1700, "amide I: turns / antiparallel beta"), (1700, 1780, "C=O of esters / acids"),
    (2800, 2870, "symmetric CH2 (lipids)"), (2870, 2900, "symmetric CH3"), (2900, 2940, "asymmetric CH2 (lipids)"),
    (2940, 3000, "asymmetric CH3"), (3000, 3050, "olefinic =C-H"),
]


def meaning(var):
    """Plain-English meaning of a descriptor variable, derived from its name."""
    if "__" in var:
        head, tail = var.split("__", 1)
        if head in BAND_MEANING and tail in MEASURE_MEANING:
            return f"{BAND_MEANING[head]}: {MEASURE_MEANING[tail]}"
        if head == "Carbohydrates" and tail.startswith("frac_"):
            return SUBBAND_MEANING.get(tail[5:], tail)
        if head.startswith("d2_") and tail == "rel_AmideI":
            try:
                pos = float(head.split("_")[-1])
                lab = next((t for lo, hi, t in ASSIGNMENTS if lo <= pos < hi), "unassigned")
                return (f"second-derivative intensity of the component at {pos:.0f} cm-1 ({lab}), "
                        "relative to the amide I maximum")
            except ValueError:
                return f"second-derivative intensity ({head}), relative to the amide I maximum"
    return RATIO_MEANING.get(var) or CENTROID_MEANING.get(var) or var


def family(var):
    if var.startswith("d2_"):
        return "second-derivative component"
    if var.startswith("log_"):
        return "log ratio"
    if var.startswith("centroid_"):
        return "centroid"
    if var.startswith("Carbohydrates__"):
        return "carbohydrate sub-band fraction"
    return "band geometry"


def load(folders, name, **kw):
    parts = [pd.read_csv(f / name, **kw) for f in folders if (f / name).exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fmt(m, s=None):
    return f"{m:.3f}" if s is None or not np.isfinite(s) else f"{m:.3f} ± {s:.3f}"


def main():
    args = [Path(a) for a in sys.argv[1:]]
    folders = args or [p for p in [Path("results_covid"), Path("results_t2d")] if p.exists()] or \
        [Path("results_ftir_descriptor")]
    folders = [f for f in folders if (f / "cv_summary.csv").exists()]
    if not folders:
        raise FileNotFoundError("No results folder with cv_summary.csv was found.")

    summ = load(folders, "cv_summary.csv").drop_duplicates(["dataset", "method", "condition"], keep="last")
    comp = load(folders, "paired_comparisons.csv").drop_duplicates(
        ["dataset", "condition", "metric", "method", "reference"], keep="last")
    noise = load(folders, "noise_harmonization_summary.csv").drop_duplicates("dataset", keep="last")
    perm = load(folders, "permutation_test.csv").drop_duplicates("dataset", keep="last")
    effects = load(folders, "feature_effects.csv").drop_duplicates(["dataset", "variable"], keep="last")
    imp = load(folders, "region_importance.csv").drop_duplicates(["dataset", "method", "group"], keep="last")
    stab = load(folders, "selection_stability.csv").drop_duplicates(["dataset", "method"], keep="last")
    rel = load(folders, "feature_reliability.csv")
    excl = load(folders, "excluded_bands.csv")
    peaks = load(folders, "detected_peaks.csv").drop_duplicates("name")
    cfg = json.load(open(folders[0] / "descriptor_configuration.json"))
    variables = cfg["variables"]
    clean = summ[summ.condition == "clean"]
    ba = comp[(comp.metric == "bal_acc") & (comp.condition == "clean")]
    datasets = [d for d in DATASET_INFO if d in set(clean.dataset)]

    def acc(ds, m):
        r = clean[(clean.dataset == ds) & (clean.method == m)]
        return (float(r.bal_acc_mean.iloc[0]), float(r.bal_acc_sd.iloc[0]), float(r.auc_mean.iloc[0])) if len(r) else None

    L = []
    w = L.append
    w("RESULTS REPORT - interpretable band descriptor for salivary ATR-FTIR spectra")
    w(f"Generated from: {', '.join(str(f) for f in folders)}")
    w("Values are balanced accuracy (mean ± SD over the outer folds of repeated nested cross-validation) unless "
      "stated otherwise. Differences are paired fold-wise differences with Nadeau-Bengio corrected 95% CIs.")
    w("")

    # ------------------------------------------------------------------ descriptor
    w("1. DESCRIPTOR CONFIGURATION")
    fams = pd.Series([family(v) for v in variables]).value_counts()
    w(f"The descriptor contains {len(variables)} variables: " +
      ", ".join(f"{n} {f} variables" for f, n in fams.items()) + ". "
      "The configuration was decided without labels and is shared by all datasets.")
    if cfg.get("excluded_bands"):
        bands = ", ".join(cfg["excluded_bands"])
        frac = excl[excl.band.isin(cfg["excluded_bands"])].groupby("band").invalid_fraction.max()
        w(f"Quality control excluded the band(s) {bands} (invalid in up to "
          f"{', '.join(f'{v:.0%}' for v in frac.values)} of the spectra of at least one dataset).")
    if len(peaks):
        w(f"{len(peaks)} second-derivative components were detected in the pooled mean spectrum, spanning "
          f"{peaks.position_cm1.min():.0f}-{peaks.position_cm1.max():.0f} cm-1.")
    if len(rel):
        nf = rel[~rel.reliable.astype(bool)].variable.unique()
        w("Sensitivity to the measured noise: " + (f"only {', '.join(nf)} changed by more than 0.5 between-subject "
          "SD when noise of the measured level was added." if len(nf) else "no variable changed by more than "
          "0.5 between-subject SD when noise of the measured level was added."))
    w("")

    # ------------------------------------------------------------------ per dataset
    for k, ds in enumerate(datasets, start=2):
        desc, pos, neg = DATASET_INFO[ds]
        w(f"{k}. {ds}: {desc}")
        e = effects[effects.dataset == ds]
        if len(e):
            w(f"Subjects: {int(e.n_positive.max())} {pos} and {int(e.n_negative.max())} {neg}.")
        nz = noise[noise.dataset == ds]
        if len(nz):
            r = nz.iloc[0]
            if str(r.get("harmonized", True)) in ("True", "true", "1", "1.0"):
                w(f"The high-frequency noise level separated the classes (AUC {r.noise_auc_before:.2f}); after "
                  f"label-free noise harmonization it no longer did (AUC {r.noise_auc_after:.2f}).")
            else:
                w(f"The noise level did not separate the classes (AUC {r.noise_auc_before:.2f}); no harmonization "
                  "was applied.")

        p, s2, b = acc(ds, PRIMARY), acc(ds, SECONDARY), acc(ds, BASE)
        refs = {m: acc(ds, m) for m in REFERENCES if acc(ds, m)}
        best_ref = max(refs, key=lambda m: refs[m][0]) if refs else None
        txt = []
        if p:
            txt.append(f"The {METHOD_NAMES[PRIMARY]} reached {fmt(p[0], p[1])} (AUC {p[2]:.3f})")
        if s2:
            txt.append(f"the {METHOD_NAMES[SECONDARY]} {fmt(s2[0], s2[1])} (AUC {s2[2]:.3f})")
        if b:
            txt.append(f"and the {METHOD_NAMES[BASE]} {fmt(b[0], b[1])}")
        if txt:
            w(", ".join(txt) + ".")
        if refs:
            w("Full-spectrum references: " + "; ".join(f"{METHOD_NAMES[m]} {fmt(v[0], v[1])}" for m, v in refs.items())
              + f". The best reference was the {METHOD_NAMES[best_ref]}.")
        sub = ba[(ba.dataset == ds) & (ba.method == PRIMARY) & ba.reference.isin(REFERENCES)]
        for _, r in sub.iterrows():
            if not np.isfinite(r.ci95_low):
                continue
            if r.ci95_low > 0:
                verdict = "superior"
            elif r.ci95_low > -NI_MARGIN:
                verdict = f"non-inferior (margin {NI_MARGIN})"
            elif r.ci95_high < -NI_MARGIN:
                verdict = "inferior"
            elif r.ci95_high < 0:
                verdict = "worse, and non-inferiority could not be shown"
            else:
                verdict = "inconclusive (the CI includes both no difference and the margin)"
            w(f"  vs. {METHOD_NAMES[r.reference]}: difference {r.mean_difference:+.3f} "
              f"(95% CI {r.ci95_low:+.3f} to {r.ci95_high:+.3f}) -> {verdict}.")
        comp_vals = {m: acc(ds, m) for m in COMPACT if acc(ds, m)}
        if comp_vals and p:
            better = [m for m, v in comp_vals.items() if v[0] > p[0]]
            w("Equal-budget compact representations: " + "; ".join(f"{METHOD_NAMES[m]} {v[0]:.3f}"
              for m, v in comp_vals.items()) + ". " + (
                f"{len(better)} of them scored higher than the primary model ("
                + ", ".join(METHOD_NAMES[m] for m in better) + ")." if better
                else "None scored higher than the primary model."))
            sup = ba[(ba.dataset == ds) & (ba.method == PRIMARY) & ba.reference.isin(COMPACT) & (ba.superior == True)]
            w("The primary model was significantly superior to: " +
              (", ".join(METHOD_NAMES[m] for m in sup.reference) if len(sup) else "none of them") + ".")
        ctrl = {m: acc(ds, m) for m in CONTROLS if acc(ds, m)}
        if ctrl:
            worst = max(v[0] for v in ctrl.values())
            w("Acquisition audit: " + "; ".join(f"{METHOD_NAMES[m]} {v[0]:.3f}" for m, v in ctrl.items()) + ". " + (
                f"Models restricted to regions without sample bands classified well above chance (up to {worst:.3f}), "
                "so the classes differ in acquisition conditions and the accuracies of all methods in this dataset "
                "cannot be attributed to sample biochemistry alone." if worst >= CHANCE_LIMIT else
                "All controls stayed close to chance, with no evidence of acquisition differences between classes."))
        pr = perm[perm.dataset == ds]
        if len(pr):
            r = pr.iloc[0]
            w(f"Permutation test (descriptor + logistic regression): observed {r.observed_bal_acc:.3f} vs. null mean "
              f"{r.null_mean:.3f}, p = {r.p_permutation:.3f} ({int(r.n_permutations)} label permutations at the "
              f"{r.permuted_unit} level).")
        top = e[e.q_BH < 0.05].assign(a=lambda d: d.cliff_delta.abs()).nlargest(5, "a")
        if len(top):
            w(f"Variables with the largest significant differences (Cliff's delta, BH q < 0.05):")
            for _, r in top.iterrows():
                direction = f"higher in {pos}" if r.cliff_delta > 0 else f"higher in {neg}"
                w(f"  - {r.variable} (delta {r.cliff_delta:+.2f}, q = {r.q_BH:.1e}; {direction}): {meaning(r.variable)}.")
        else:
            w("No descriptor variable differed significantly between classes after BH correction.")
        for m in [PRIMARY, SECONDARY]:
            ti = imp[(imp.dataset == ds) & (imp.method == m)].nlargest(3, "mean_drop")
            if len(ti):
                w(f"Most important regions for the {METHOD_NAMES[m]} (grouped permutation importance): " +
                  ", ".join(f"{g} cm-1 ({d:.3f})" if g != "ratios_centroids" else f"ratios and centroids ({d:.3f})"
                            for g, d in zip(ti.group, ti.mean_drop)) + ".")
        st = stab[(stab.dataset == ds) & stab.method.isin(["VIP_topk", "SPA_LDA"])]
        if len(st):
            w("Selection stability across folds (mean Jaccard): " + "; ".join(
                f"{METHOD_NAMES[m]} {j:.2f}" for m, j in zip(st.method, st.mean_jaccard)) +
              "; the descriptor is fixed by construction (1.00).")
        w("")

    # ------------------------------------------------------------------ robustness
    det = load(folders, "cv_details.csv")
    if len(det) and det.condition.nunique() > 1:
        w(f"{len(datasets) + 2}. ROBUSTNESS TO PERTURBATIONS")
        piv = det.pivot_table(index=["dataset", "method", "repeat", "fold"], columns="condition", values="bal_acc")
        for m in [PRIMARY, SECONDARY, "Full_spectrum", "Second_derivative"]:
            s = piv.xs(m, level="method") if m in piv.index.get_level_values("method") else None
            if s is None:
                continue
            drops = {c: float((s["clean"] - s[c]).mean()) for c in s.columns if c != "clean"}
            w(f"{METHOD_NAMES[m]}: mean drop in balanced accuracy " +
              ", ".join(f"{c} {v:+.3f}" for c, v in drops.items()) + ".")
        w("Perturbation magnitudes were calibrated from replicate variability when replicates were available "
          "(see perturbation_magnitudes.csv).")
        w("")

    # ------------------------------------------------------------------ glossary
    gl = pd.DataFrame({"variable": variables, "family": [family(v) for v in variables],
                       "meaning": [meaning(v) for v in variables]})
    gl.to_csv("variable_glossary.csv", index=False)
    w("APPENDIX. GLOSSARY OF DESCRIPTOR VARIABLES")
    for _, r in gl.iterrows():
        w(f"{r.variable}: {r.meaning}.")
    Path("results_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("\nSaved: results_report.txt and variable_glossary.csv")


if __name__ == "__main__":
    main()
