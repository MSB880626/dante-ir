# -*- coding: utf-8 -*-
"""
High-level entry points: run() for one dataset (used by the Colab notebook) and run_config() for several
datasets described in a JSON file (used to reproduce a study).
"""
import importlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from .io import load_spectra

MODES = ("audit", "exploratory", "full", "quick")


def _prepare_core(wn_max, crystal, sample_type):
    """Sets the run-wide settings shared with the worker processes and (re)loads the core module."""
    os.environ["DANTE_IR_WNMAX"] = f"{wn_max:.1f}"
    os.environ["DANTE_IR_CRYSTAL"] = crystal
    os.environ["DANTE_IR_SAMPLE"] = sample_type
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)       # workers must import the module with the new settings
    except Exception:
        pass
    from . import core
    return importlib.reload(core)


def _auto_inner(core, y, g, n_inner):
    unit = g if g is not None else np.arange(len(y)).astype(str)
    import pandas as pd
    n_min = int(pd.DataFrame({"u": unit, "y": y}).drop_duplicates("u")["y"].value_counts().min())
    for k in [n_inner, 4, 3, 2]:
        if k <= n_inner and math.ceil(2 * k * core.N_OUTER / (core.N_OUTER - 1)) <= n_min:
            return k
    raise ValueError(f"Only {n_min} subjects in the minority class: at least "
                     f"{math.ceil(4 * core.N_OUTER / (core.N_OUTER - 1))} are needed for nested validation.")


def _common_range(datasets):
    lo = max(float(np.min(d["wn"])) for d in datasets.values())
    hi = min(float(np.max(d["wn"])) for d in datasets.values())
    if lo > 900.0:
        raise ValueError(f"The spectra must start at or below 900 cm-1 (they start at {lo:.0f}).")
    if hi < 3100.0:
        raise ValueError(f"The spectra must reach at least 3100 cm-1 (they end at {hi:.0f}).")
    return min(3995.0, math.floor(hi) - 1.0)


def _execute(datasets, sample_type, crystal, mode, output_dir, run_cv=None, zip_results=True, resume=True):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    core = _prepare_core(_common_range(datasets), crystal, sample_type)
    core.set_mode(mode)
    core.set_output(output_dir)
    for name, d in datasets.items():
        y = (np.asarray(d["y_raw"]).astype(str) == str(d["positive"])).astype(int)
        g = d.get("g")
        if g is not None and len(set(g)) == len(g):
            g = None
        d["n_inner"] = _auto_inner(core, y, g, min(int(d.get("n_inner", core.N_INNER)), core.N_INNER))
    summary = core.run_pipeline(datasets, run_cv=run_cv, resume=resume)
    from .report import write_report
    write_report(output_dir)
    if zip_results:
        archive = shutil.make_archive(str(Path(output_dir)), "zip", root_dir=output_dir)
        summary["zip"] = archive
    summary["output_dir"] = str(Path(output_dir).resolve())
    return summary


def run(spectra, label_column, positive_class, metadata=None, id_column=None, subject_column=None,
        batch_column=None, sample_type="saliva", crystal="diamond", mode="audit",
        output_dir="dante_ir_results", dataset_name="dataset", description="", units="auto", zip_results=True):
    """Analyses one dataset.

    spectra         spectral table (csv, tsv, txt, xlsx), wide (one row per spectrum) or transposed
                    (one row per wavenumber; then `metadata` is required)
    label_column    column with the class of each spectrum (two classes)
    positive_class  value of label_column for the disease / positive group
    subject_column  column identifying the person when there are replicate spectra (recommended)
    batch_column    optional column with the acquisition batch, session, site or day (enables the batch audit)
    sample_type     saliva, serum, plasma, blood, urine or other
    crystal         diamond, ZnSe, Ge, Si or transmission
    mode            audit (minutes), exploratory (about an hour), full (several hours) or quick (smoke test)
    """
    d = load_spectra(spectra, label_column, metadata_path=metadata, id_column=id_column,
                     subject_column=subject_column, batch_column=batch_column, units=units)
    print(f"Read: {d['read_info']}")
    classes = sorted(set(d["y_raw"]))
    if len(classes) != 2:
        raise ValueError(f"Exactly two classes are required in '{label_column}', found: {classes}")
    if str(positive_class) not in classes:
        raise ValueError(f"Positive class '{positive_class}' not found; classes are {classes}")
    negative = [c for c in classes if c != str(positive_class)][0]
    d.update({"positive": str(positive_class), "role": "disease", "description": description,
              "positive_label": str(positive_class), "negative_label": negative})
    return _execute({dataset_name: d}, sample_type, crystal, mode, output_dir, zip_results=zip_results)


def run_config(config_path, run_cv=None, resume=True):
    """Analyses several datasets that share sample type and crystal, described in a JSON file:
    {"sample_type": "saliva", "crystal": "diamond", "mode": "full", "output_dir": "results",
     "datasets": {"name": {"spectra": "...", "label_column": "label", "positive_class": "...",
                           "subject_column": null, "batch_column": null, "metadata": null,
                           "role": "disease", "description": "...", "literature_panel": null}}}"""
    cfg = json.load(open(config_path))
    base = Path(config_path).parent
    datasets = {}
    for name, c in cfg["datasets"].items():
        meta = c.get("metadata")
        d = load_spectra(base / c["spectra"], c.get("label_column", "label"),
                         metadata_path=(base / meta) if meta else None, id_column=c.get("id_column"),
                         subject_column=c.get("subject_column"), batch_column=c.get("batch_column"),
                         units=c.get("units", "auto"))
        print(f"{name}: {d['read_info']}")
        classes = sorted(set(d["y_raw"]))
        neg = [x for x in classes if x != str(c["positive_class"])]
        d.update({"positive": str(c["positive_class"]), "role": c.get("role", "disease"),
                  "description": c.get("description", ""), "positive_label": c.get("positive_label", c["positive_class"]),
                  "negative_label": c.get("negative_label", neg[0] if neg else ""),
                  "literature_panel": c.get("literature_panel")})
        if c.get("n_inner"):
            d["n_inner"] = int(c["n_inner"])
        datasets[name] = d
    out = base / cfg.get("output_dir", "dante_ir_results")
    return _execute(datasets, cfg.get("sample_type", "other"), cfg.get("crystal", "diamond"),
                    cfg.get("mode", "full"), out, run_cv=run_cv, zip_results=cfg.get("zip", True), resume=resume)


def external_validation(train_spectra, test_spectra, label_column, positive_class, subject_column=None,
                        covariates=None, sample_type="plasma", crystal="diamond", output_dir="dante_ir_external",
                        methods=None, zip_results=True):
    """Trains every method on the training table (hyperparameters tuned by grouped inner CV on the training
    data only) and evaluates it once on the independent test table, per subject, with bootstrap CIs.
    covariates: optional list of numeric clinical columns (e.g. ["age", "diabetes"]) present in both tables;
    adds a clinical-only model and a descriptor + clinical model."""
    import pandas as pd
    tr = load_spectra(train_spectra, label_column, subject_column=subject_column)
    te = load_spectra(test_spectra, label_column, subject_column=subject_column)
    print(f"Training: {tr['read_info']}\nTest: {te['read_info']}")
    core = _prepare_core(_common_range({"tr": tr, "te": te}), crystal, sample_type)
    core.set_mode("full")
    core.set_output(output_dir)
    out = {}
    for nm, d in (("train", tr), ("test", te)):
        y = (d["y_raw"] == str(positive_class)).astype(int)
        g = d["g"] if d["g"] is not None else np.array([f"{nm}_{i}" for i in range(len(y))])
        cov = None
        if covariates:
            cov = d["info"][covariates].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        out[nm] = {"X": core.resample(d["wn"], d["X"]), "y": y, "g": np.asarray(g).astype(str), "cov": cov}
    shared = set(out["train"]["g"]) & set(out["test"]["g"])
    if shared:
        raise ValueError(f"Subjects present in both training and test sets: {sorted(shared)[:10]}")
    n_inner = _auto_inner(core, out["train"]["y"], out["train"]["g"], 5)
    res = core.run_external(out["train"], out["test"], covariates=covariates, methods=methods, n_inner=n_inner)
    lines = ["DANTE-IR EXTERNAL VALIDATION",
             f"Training: {train_spectra} ({len(set(out['train']['g']))} subjects); test: {test_spectra} "
             f"({len(set(out['test']['g']))} subjects). Positive class: {positive_class}.",
             "Each method was tuned by grouped inner cross-validation on the training set only and applied once to "
             "the test set; metrics are per test subject (mean score of its spectra), with 95% bootstrap CIs.", ""]
    for _, r in res.sort_values("auc", ascending=False).iterrows():
        lines.append(f"{r.method}: AUC {r.auc:.3f} ({r.auc_ci_low:.3f}-{r.auc_ci_high:.3f}); balanced accuracy "
                     f"{r.bal_acc:.3f} ({r.bal_acc_ci_low:.3f}-{r.bal_acc_ci_high:.3f}); sensitivity "
                     f"{r.sensitivity:.3f}; specificity {r.specificity:.3f}")
    Path(output_dir, "external_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    summary = {"output_dir": str(Path(output_dir).resolve())}
    if zip_results:
        summary["zip"] = shutil.make_archive(str(Path(output_dir)), "zip", root_dir=output_dir)
    return summary


def interpret_regions(spectra, label_column, positive_class, metadata=None, id_column=None, subject_column=None,
                      sample_type="saliva", crystal="diamond", output_dir="dante_ir_interpretation",
                      dataset_name="dataset", n_perm=100, n_repeats=2, zip_results=True):
    """Region-level interpretation with null-calibrated thresholds, null reference regions, linear SHAP and
    stability across folds, for the full spectrum and the descriptor (see dante_ir.interpret)."""
    from . import interpret as itp
    d = load_spectra(spectra, label_column, metadata_path=metadata, id_column=id_column,
                     subject_column=subject_column)
    print(f"Read: {d['read_info']}")
    core = _prepare_core(_common_range({"d": d}), crystal, sample_type)
    core.set_mode("full")
    core.set_output(output_dir)
    y = (d["y_raw"] == str(positive_class)).astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(f"Positive class '{positive_class}' not found or it is the only class.")
    g = d["g"] if d["g"] is not None and len(set(d["g"])) < len(d["g"]) else None
    X = core.resample(d["wn"], d["X"])
    noise_auc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, core.noise_abs(X))
    if abs(noise_auc - 0.5) > core.NOISE_AUC_TOL:
        X, _ = core.harmonize_noise(dataset_name, X, y)
    desc_kw, _, _ = core.configure_descriptor({dataset_name: (X, y, g, "", "")})
    desc = core.BandDescriptor(**desc_kw)
    res, summary = itp.interpret_regions_core(core, X, y, g, desc, n_repeats=n_repeats, n_perm=n_perm)
    out = Path(output_dir)
    res.to_csv(out / "region_interpretation.csv", index=False)
    with open(out / "region_interpretation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    itp.plot_regions(res, core, out / "fig_region_interpretation.png", title=dataset_name)
    text = itp.summary_text(summary, dataset_name)
    (out / "region_interpretation_report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    result = {"output_dir": str(out.resolve()), "summary": summary}
    if zip_results:
        result["zip"] = shutil.make_archive(str(out), "zip", root_dir=output_dir)
    return result
