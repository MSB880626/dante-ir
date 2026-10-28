# -*- coding: utf-8 -*-
"""
Natural-language report (plain text, ready to paste into a word processor) and variable glossary, written
from the outputs of a DANTE-IR run. Nothing is recomputed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY, SECONDARY, BASE = "Descriptor_GB", "Descriptor_LR", "Descriptor_base_LR"
REFERENCES = ["Full_spectrum", "PCA", "Second_derivative", "PLS-DA"]
COMPACT = ["Binning", "Random_intervals", "VIP_topk", "SPA_LDA", "Literature_panel"]
CONTROLS = ["Neg_control_raw", "Neg_control_d2", "Diamond_control_d2"]
NI_MARGIN = 0.03
CHANCE_LIMIT = 0.65
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
    "Thiocyanate_SCN": "thiocyanate (SCN-, ~2055 cm-1; salivary module, not validated)",
    "Urea_CN_1460": "urea C-N stretching (~1460 cm-1; urine module, not validated)",
    "Urea_NH2_1160": "urea NH2 rocking (~1160 cm-1; urine module, not validated)",
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
    "log_SCN_AmideI": "log ratio thiocyanate / proteins (salivary module, not validated)",
    "log_Urea_AmideII": "log ratio urea / amide II (urine module, not validated)",
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


_COLUMNS = {
    "permutation_test.csv": ["dataset", "observed_bal_acc", "null_mean", "p_permutation", "n_permutations",
                             "permuted_unit"],
    "region_importance.csv": ["dataset", "method", "group", "mean_drop"],
    "selection_stability.csv": ["dataset", "method", "mean_jaccard"],
    "feature_reliability.csv": ["dataset", "variable", "reliable"],
    "excluded_bands.csv": ["dataset", "band", "invalid_fraction"],
    "detected_peaks.csv": ["name", "position_cm1"],
    "noise_harmonization_summary.csv": ["dataset", "harmonized", "noise_auc_before", "noise_auc_after"],
    "feature_effects.csv": ["dataset", "variable", "cliff_delta", "q_BH", "n_positive", "n_negative"],
    "batch_audit.csv": ["dataset"],
    "cv_details.csv": ["dataset", "method", "repeat", "fold", "condition", "bal_acc"],
}


def _top(df, n, col):
    return df if df.empty else df.assign(**{col: pd.to_numeric(df[col], errors="coerce")}).nlargest(n, col)


def load(folders, name, dedup=None):
    parts = [pd.read_csv(f / name) for f in folders if (f / name).exists() and (f / name).stat().st_size > 5]
    if not parts:
        return pd.DataFrame(columns=_COLUMNS.get(name, ["dataset"]))
    df = pd.concat(parts, ignore_index=True)
    if dedup is not None:
        df = df.drop_duplicates(dedup, keep="last")
    return df


def fmt(m, s=None):
    return f"{m:.3f}" if s is None or not np.isfinite(s) else f"{m:.3f} ± {s:.3f}"


def write_report(folders, out_path=None, glossary_path=None):
    """folders: one or more result folders of the same study. Returns the report text."""
    folders = [Path(f) for f in ([folders] if isinstance(folders, (str, Path)) else folders)]
    folders = [f for f in folders if (f / "cv_summary.csv").exists()]
    if not folders:
        raise FileNotFoundError("No results folder with cv_summary.csv was found.")
    out_path = Path(out_path) if out_path else folders[0] / "report.txt"
    glossary_path = Path(glossary_path) if glossary_path else folders[0] / "variable_glossary.csv"
    summary = {}
    for f in folders:
        if (f / "run_summary.json").exists():
            s = json.load(open(f / "run_summary.json"))
            summary.setdefault("datasets", {}).update(s.get("datasets", {}))
            for k in ("sample_type", "crystal", "mode", "unvalidated_bands", "wn_range", "version"):
                summary.setdefault(k, s.get(k))
    summ = load(folders, "cv_summary.csv", dedup=["dataset", "method", "condition"])
    comp = load(folders, "paired_comparisons.csv", dedup=["dataset", "condition", "metric", "method", "reference"])
    noise = load(folders, "noise_harmonization_summary.csv", dedup="dataset")
    perm = load(folders, "permutation_test.csv", dedup="dataset")
    effects = load(folders, "feature_effects.csv", dedup=["dataset", "variable"])
    imp = load(folders, "region_importance.csv", dedup=["dataset", "method", "group"])
    stab = load(folders, "selection_stability.csv", dedup=["dataset", "method"])
    rel = load(folders, "feature_reliability.csv")
    excl = load(folders, "excluded_bands.csv")
    peaks = load(folders, "detected_peaks.csv", dedup="name")
    cfg = json.load(open(folders[0] / "descriptor_configuration.json"))
    variables = cfg["variables"]
    clean = summ[summ.condition == "clean"]
    ba = comp[(comp.metric == "bal_acc") & (comp.condition == "clean")]
    dinfo = summary.get("datasets", {})
    datasets = [d for d in dinfo if d in set(clean.dataset)] + sorted(set(clean.dataset) - set(dinfo))

    def acc(ds, m):
        r = clean[(clean.dataset == ds) & (clean.method == m)]
        return (float(r.bal_acc_mean.iloc[0]), float(r.bal_acc_sd.iloc[0]), float(r.auc_mean.iloc[0])) if len(r) else None

    L = []
    w = L.append
    st, cr = summary.get("sample_type", "unknown"), summary.get("crystal", "unknown")
    w(f"DANTE-IR RESULTS REPORT (version {summary.get('version', '?')}, mode: {summary.get('mode', '?')})")
    w(f"Sample type: {st}. ATR crystal: {cr}." + (" The diamond multiphonon region (1800-2700 cm-1) was excluded."
                                                   if cr == "diamond" else ""))
    w(f"Generated from: {', '.join(str(f) for f in folders)}")
    w("Values are balanced accuracy (mean ± SD over the outer folds of repeated nested cross-validation) unless "
      "stated otherwise. Differences are paired fold-wise differences with Nadeau-Bengio corrected 95% CIs.")
    w("")

    # ------------------------------------------------------------------ summary verdict
    w("SUMMARY")
    ba0 = load(folders, "batch_audit.csv")
    for ds in datasets:
        msgs = []
        ctrl = clean[(clean.dataset == ds) & clean.method.isin(CONTROLS)]
        if len(ctrl):
            worst = float(ctrl.bal_acc_mean.max())
            msgs.append(f"models restricted to regions without sample bands reached up to {worst:.2f} balanced accuracy "
                        + ("-> the classes differ in acquisition conditions (instrument, environment or session); "
                           "classification results are NOT interpretable as sample biochemistry"
                           if worst >= CHANCE_LIMIT else "-> no evidence of acquisition differences between classes"))
        nz = noise[noise.dataset == ds]
        if len(nz) and abs(float(nz.iloc[0].noise_auc_before) - 0.5) > 0.1:
            msgs.append(f"the noise level alone separated the classes (AUC {float(nz.iloc[0].noise_auc_before):.2f}), "
                        "which indicates different acquisition or processing; it was harmonized before modelling")
        bb = ba0[ba0.dataset == ds] if len(ba0) else ba0
        if len(bb) and float(bb.iloc[0].cramers_v_class_batch) >= 0.5:
            msgs.append(f"class and batch/site are strongly associated (Cramer's V {float(bb.iloc[0].cramers_v_class_batch):.2f}); "
                        "accuracy may reflect the batch rather than the condition")
        best = clean[(clean.dataset == ds) & ~clean.method.isin(CONTROLS)].sort_values("bal_acc_mean").tail(1)
        if len(best):
            msgs.append(f"best model: {METHOD_NAMES.get(best.method.iloc[0], best.method.iloc[0])} "
                        f"({float(best.bal_acc_mean.iloc[0]):.2f} balanced accuracy)")
        w(f"- {ds}: " + "; ".join(msgs) + ".")
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
    if summary.get("unvalidated_bands"):
        w("Sample-type module bands included (literature-based, NOT validated on data; interpret with caution): "
          + ", ".join(summary["unvalidated_bands"]) + ".")
    if len(rel):
        nf = rel[~rel.reliable.astype(bool)].variable.unique()
        w("Sensitivity to the measured noise: " + (f"only {', '.join(nf)} changed by more than 0.5 between-subject "
          "SD when noise of the measured level was added." if len(nf) else "no variable changed by more than "
          "0.5 between-subject SD when noise of the measured level was added."))
    w("")

    # ------------------------------------------------------------------ per dataset
    for k, ds in enumerate(datasets, start=2):
        di = dinfo.get(ds, {})
        pos = di.get("positive_label") or di.get("positive") or "positive class"
        neg = di.get("negative_label") or "negative class"
        w(f"{k}. {ds}" + (f": {di['description']}" if di.get("description") else ""))
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
            txt.append(f"the {METHOD_NAMES[SECONDARY]} {'reached ' if not p else ''}{fmt(s2[0], s2[1])} (AUC {s2[2]:.3f})")
        if b:
            txt.append(f"and the {METHOD_NAMES[BASE]} {fmt(b[0], b[1])}")
        if txt:
            line = ", ".join(txt) + "."
            w(line[0].upper() + line[1:])
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
        top = _top(e[pd.to_numeric(e.q_BH, errors="coerce") < 0.05].assign(
            a=lambda d: pd.to_numeric(d.cliff_delta, errors="coerce").abs()), 5, "a")
        if len(top):
            w(f"Variables with the largest significant differences (Cliff's delta, BH q < 0.05):")
            for _, r in top.iterrows():
                direction = f"higher in {pos}" if r.cliff_delta > 0 else f"higher in {neg}"
                w(f"  - {r.variable} (delta {r.cliff_delta:+.2f}, q = {r.q_BH:.1e}; {direction}): {meaning(r.variable)}.")
        else:
            w("No descriptor variable differed significantly between classes after BH correction.")
        for m in [PRIMARY, SECONDARY]:
            ti = _top(imp[(imp.dataset == ds) & (imp.method == m)], 3, "mean_drop")
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

    # ------------------------------------------------------------------ batch audit
    ba_ = load(folders, "batch_audit.csv")
    if len(ba_):
        w(f"{len(datasets) + 2}. BATCH / SITE AUDIT")
        for _, r in ba_.iterrows():
            txt = (f"{r.dataset}: {int(r.n_batches)} batches; association between class and batch, Cramer's V = "
                   f"{r.cramers_v_class_batch:.2f}")
            if isinstance(r.get("batches_with_a_single_class"), str) and r.batches_with_a_single_class:
                txt += f"; batches containing a single class: {r.batches_with_a_single_class}"
            w(txt + ".")
            if "batch_from_full_spectrum" in r and pd.notna(r.get("batch_from_full_spectrum")):
                w(f"  The batch could be predicted from the full spectrum with balanced accuracy "
                  f"{r.batch_from_full_spectrum:.3f}" + (
                      f" and from the negative-control region with {r.batch_from_negative_control_d2:.3f}"
                      if pd.notna(r.get("batch_from_negative_control_d2", np.nan)) else "")
                  + f" (chance {r.chance_level:.2f}).")
            if r.cramers_v_class_batch >= 0.5:
                w("  WARNING: class and batch are strongly associated; classification accuracy may reflect the batch. "
                  "Compare classes within batches whenever possible.")
        w("")

    # ------------------------------------------------------------------ robustness
    det = load(folders, "cv_details.csv")
    if len(det) and det.condition.nunique() > 1:
        w(f"{len(datasets) + 3}. ROBUSTNESS TO PERTURBATIONS")
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
    gl.to_csv(glossary_path, index=False)
    w("APPENDIX. GLOSSARY OF DESCRIPTOR VARIABLES")
    for _, r in gl.iterrows():
        w(f"{r.variable}: {r.meaning}.")
    text = "\n".join(L)
    out_path.write_text(text, encoding="utf-8")
    return text
