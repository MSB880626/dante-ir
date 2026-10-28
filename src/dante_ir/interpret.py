# -*- coding: utf-8 -*-
"""
Region-level interpretation of classifiers of biofluid IR spectra.

For two interpretable linear models, the full spectrum (Full_spectrum_LR) and the band descriptor
(Descriptor_LR), the analysis estimates in repeated grouped cross-validation:

  1. Region importance: drop in test AUC when the variables of a spectral region (20 cm-1 blocks; for the
     descriptor, the variables whose band, sub-band or peak is centred in the block) are jointly permuted
     across test subjects.
  2. Null-calibrated threshold: the same analysis repeated after permuting the class labels between subjects;
     the 95th percentile of the maximum region importance over all regions gives a family-wise threshold.
     A region is called informative only if its importance exceeds that threshold.
  3. Null reference regions: blocks of a spectral region without sample bands (3750+ cm-1) are included as
     extra inputs ("null probes"). In data without acquisition differences they should stay below the
     threshold; if they exceed it, the region attributions of that dataset cannot be read as chemistry.
  4. Linear SHAP: for a linear model on standardized inputs, the exact SHAP value of variable j for a
     spectrum is coef_j * (x_j - mean_j); region SHAP importance is the mean absolute value summed over the
     variables of the region. Its agreement with the permutation importance is reported.
  5. Stability: Spearman correlation of the region-importance profiles between folds and frequency with which
     each region appears among the five most important.
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _windows(core, include_null=True):
    w = []
    for lo, hi in core.BIO_WINDOWS:
        for e0 in np.arange(lo, hi, core.IMP_WIDTH):
            w.append((float(e0), float(min(e0 + core.IMP_WIDTH, hi)), "sample"))
    if include_null:
        for lo, hi in core.NEG_CONTROL_WINDOWS:
            for e0 in np.arange(lo, hi, core.IMP_WIDTH):
                w.append((float(e0), float(min(e0 + core.IMP_WIDTH, hi)), "null"))
    return w


def _build_inputs(core, X, desc):
    """Full-spectrum matrix (SNV over sample + null regions), descriptor matrix with null probes, and the
    column groups of each."""
    windows = _windows(core, include_null=bool(core.NEG_CONTROL_WINDOWS))
    mask = core._window_mask(list(core.BIO_WINDOWS) + list(core.NEG_CONTROL_WINDOWS))
    wn = core.GRID[mask]
    Z = core.SNV().transform(np.asarray(X)[:, mask])
    groups_full = []
    probes, probe_groups = [], []
    for e0, e1, kind in windows:
        cols = np.where((wn >= e0) & (wn < e1))[0]
        if len(cols):
            groups_full.append((f"{e0:.0f}-{e1:.0f}", e0, e1, kind, cols))
            if kind == "null":
                probes.append(Z[:, cols].mean(axis=1))
                probe_groups.append((f"{e0:.0f}-{e1:.0f}", e0, e1))
    F = desc.transform(X)
    centres = [None if sp is None else 0.5 * (sp[0] + sp[1]) for sp in desc.feature_spans()]
    groups_desc = []
    for e0, e1, kind in windows:
        if kind != "sample":
            continue
        cols = [j for j, c in enumerate(centres) if c is not None and e0 <= c < e1]
        if cols:
            groups_desc.append((f"{e0:.0f}-{e1:.0f}", e0, e1, "sample", np.array(cols)))
    nopos = [j for j, c in enumerate(centres) if c is None]
    if nopos:
        groups_desc.append(("ratios_centroids", np.nan, np.nan, "sample", np.array(nopos)))
    if probes:
        P = np.column_stack(probes)
        start = F.shape[1]
        for k, (lab, e0, e1) in enumerate(probe_groups):
            groups_desc.append((lab, e0, e1, "null", np.array([start + k])))
        F = np.hstack([F, P])
    return {"Full_spectrum_LR": (Z, groups_full), "Descriptor_LR": (F, groups_desc)}


def _pipe(C):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))])


def _fold_importance(est, Xte, yte, groups, rng, n_rep):
    s = est.decision_function(Xte)
    base = roc_auc_score(yte, s)
    imp = np.empty(len(groups))
    for k, (_, _, _, _, cols) in enumerate(groups):
        d = []
        for _ in range(n_rep):
            Xp = Xte.copy()
            Xp[:, cols] = Xte[rng.permutation(len(Xte))][:, cols]
            d.append(base - roc_auc_score(yte, est.decision_function(Xp)))
        imp[k] = np.mean(d)
    return imp, base


def _linear_shap(est, Xte, groups):
    Zt = est[:-1].transform(Xte)                     # imputed + standardized test inputs (train statistics)
    phi = np.abs(Zt * est[-1].coef_.ravel()).mean(axis=0)
    return np.array([phi[cols].sum() for *_, cols in groups])


def interpret_regions_core(core, X, y, g, desc, n_repeats=2, n_splits=5, n_perm=100, n_rep=5, seed=42):
    inputs = _build_inputs(core, X, desc)
    rows, summary = [], {}
    for model, (M, groups) in inputs.items():
        rng = np.random.default_rng(seed)
        imps, shaps, aucs, Cs = [], [], [], []
        for r in range(n_repeats):
            for tr, te in core.splitter(n_splits, seed + r, g).split(M, y, g):
                gi = None if g is None else g[tr]
                gs = GridSearchCV(_pipe(1.0), {"clf__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc",
                                  cv=core.splitter(3, seed + 100 + r, gi), n_jobs=-1)
                gs.fit(M[tr], y[tr], groups=gi)
                est = gs.best_estimator_
                Cs.append(gs.best_params_["clf__C"])
                imp, auc = _fold_importance(est, M[te], y[te], groups, rng, n_rep)
                imps.append(imp)
                aucs.append(auc)
                shaps.append(_linear_shap(est, M[te], groups))
        imps, shaps = np.array(imps), np.array(shaps)
        C_fixed = float(np.median(Cs))
        # Null distribution of the maximum region importance (labels permuted between subjects)
        unit = g if g is not None else np.arange(len(y)).astype(str)
        subj = pd.Series(y, index=unit).groupby(level=0).first()
        null_max = np.empty(n_perm)
        for p in range(n_perm):
            mp = dict(zip(subj.index, rng.permutation(subj.values)))
            yp = np.array([mp[u] for u in unit])
            fold_imp = []
            for tr, te in core.splitter(n_splits, seed + 500 + p, g).split(M, yp, g):
                if len(np.unique(yp[te])) < 2:
                    continue
                est = _pipe(C_fixed).fit(M[tr], yp[tr])
                fold_imp.append(_fold_importance(est, M[te], yp[te], groups, rng, 1)[0])
            null_max[p] = np.max(np.mean(fold_imp, axis=0))
        thr = float(np.percentile(null_max, 95))
        mean_imp = imps.mean(axis=0)
        top5 = np.zeros(len(groups))
        for v in imps:
            top5[np.argsort(-v)[:5]] += 1
        pair = [spearmanr(imps[i], imps[j])[0] for i in range(len(imps)) for j in range(i + 1, len(imps))]
        sample_idx = [k for k, gr in enumerate(groups) if gr[3] == "sample"]
        null_idx = [k for k, gr in enumerate(groups) if gr[3] == "null"]
        agree = spearmanr(mean_imp[sample_idx], shaps.mean(axis=0)[sample_idx])[0]
        for k, (lab, e0, e1, kind, cols) in enumerate(groups):
            rows.append({"model": model, "region": lab, "start_cm1": e0, "end_cm1": e1, "region_type": kind,
                         "n_variables": len(cols), "importance_auc_drop": mean_imp[k], "importance_sd": imps[:, k].std(),
                         "null_threshold_95": thr, "informative": bool(mean_imp[k] > thr),
                         "p_fwer": float((1 + np.sum(null_max >= mean_imp[k])) / (n_perm + 1)),
                         "linear_shap": shaps.mean(axis=0)[k], "top5_frequency": top5[k] / len(imps)})
        summary[model] = {
            "cv_auc_mean": float(np.mean(aucs)), "cv_auc_sd": float(np.std(aucs)), "folds": len(imps),
            "null_threshold_95": thr, "n_label_permutations": n_perm,
            "informative_sample_regions": [groups[k][0] for k in sample_idx if mean_imp[k] > thr],
            "null_probe_regions_above_threshold": [groups[k][0] for k in null_idx if mean_imp[k] > thr],
            "max_null_probe_importance": float(mean_imp[null_idx].max()) if null_idx else None,
            "stability_spearman_between_folds": float(np.nanmean(pair)),
            "agreement_permutation_vs_linear_shap_spearman": float(agree)}
        print(f"  {model}: CV AUC {np.mean(aucs):.3f}; null threshold {thr:.4f}; informative regions: "
              f"{', '.join(summary[model]['informative_sample_regions']) or 'none'}; null probes above threshold: "
              f"{', '.join(summary[model]['null_probe_regions_above_threshold']) or 'none'}", flush=True)
    res = pd.DataFrame(rows)
    # Agreement between the descriptor and the full spectrum on common sample windows
    a = res[(res.model == "Full_spectrum_LR") & (res.region_type == "sample")].set_index("region").importance_auc_drop
    b = res[(res.model == "Descriptor_LR") & (res.region_type == "sample")].set_index("region").importance_auc_drop
    common = a.index.intersection(b.index)
    summary["agreement_descriptor_vs_full_spectrum_spearman"] = (float(spearmanr(a[common], b[common])[0])
                                                                 if len(common) > 3 else None)
    return res, summary


def plot_regions(res, core, out_path, title=""):
    models = list(res.model.unique())
    panels = [(lo, hi, "sample") for lo, hi in core.BIO_WINDOWS] + [(lo, hi, "null") for lo, hi in core.NEG_CONTROL_WINDOWS]
    fig, axes = plt.subplots(len(models), len(panels), figsize=(15, 3.3 * len(models)), squeeze=False,
                             gridspec_kw={"width_ratios": [hi - lo for lo, hi, _ in panels]})
    for row, model in zip(axes, models):
        d = res[(res.model == model) & res.start_cm1.notna()]
        thr = float(d.null_threshold_95.iloc[0])
        smax = d.linear_shap.max() or 1.0
        for ax, (lo, hi, kind) in zip(row, panels):
            s = d[(d.start_cm1 >= lo) & (d.end_cm1 <= hi + 1e-6)]
            cen = 0.5 * (s.start_cm1 + s.end_cm1)
            col = np.where(s.informative, "#c44e52" if kind == "null" else "#2a9d8f", "0.7")
            ax.bar(cen, s.importance_auc_drop, width=core.IMP_WIDTH * 0.85, color=col)
            ax.plot(cen, s.linear_shap / smax * max(d.importance_auc_drop.max(), thr), "k.", ms=3,
                    label="linear SHAP (scaled)")
            ax.axhline(thr, color="k", ls="--", lw=0.8, label="null threshold (95%)")
            ax.set_xlim(hi, lo)
            if kind == "null":
                ax.set_title("null reference region", fontsize=8)
        row[0].set_ylabel(f"{model.replace('_', ' ')}\nAUC drop")
        row[0].legend(frameon=False, fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    if title:
        fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summary_text(summary, dataset):
    L = [f"REGION INTERPRETATION - {dataset}"]
    for model in ("Full_spectrum_LR", "Descriptor_LR"):
        s = summary.get(model)
        if not s:
            continue
        L.append(f"{model}: cross-validated AUC {s['cv_auc_mean']:.3f} ± {s['cv_auc_sd']:.3f} ({s['folds']} folds). "
                 f"Null-calibrated threshold (95th percentile of the maximum region importance over "
                 f"{s['n_label_permutations']} label permutations): {s['null_threshold_95']:.4f} AUC drop.")
        L.append(f"  Informative sample regions: {', '.join(s['informative_sample_regions']) or 'none'}.")
        if s["max_null_probe_importance"] is not None:
            L.append(f"  Null reference region: maximum importance {s['max_null_probe_importance']:.4f}; blocks above "
                     f"threshold: {', '.join(s['null_probe_regions_above_threshold']) or 'none'}"
                     + (" -> region attributions of this dataset should not be read as sample chemistry."
                        if s["null_probe_regions_above_threshold"] else " -> attributions are not driven by the null region."))
        L.append(f"  Stability between folds (mean Spearman of region profiles): "
                 f"{s['stability_spearman_between_folds']:.2f}; agreement with linear SHAP (Spearman): "
                 f"{s['agreement_permutation_vs_linear_shap_spearman']:.2f}.")
    if summary.get("agreement_descriptor_vs_full_spectrum_spearman") is not None:
        L.append(f"Agreement of region profiles, descriptor vs. full spectrum (Spearman): "
                 f"{summary['agreement_descriptor_vs_full_spectrum_spearman']:.2f}.")
    return "\n".join(L)
