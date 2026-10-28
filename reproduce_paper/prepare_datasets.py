#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converts the two public salivary ATR-FTIR datasets to a common format for ftir_descriptor.py:
one row per spectrum, columns 'label', 'subject', 'site', followed by numeric wavenumber headers (cm-1).

Inputs (working folder):
  diabetes.csv                         T2D dataset: one row per spectrum, column 'label' (DM2 / control),
                                       wavenumber headers.
  Australia_Human_SpectralData.csv     COVID-19 dataset (Zenodo 5703689): rows = wavenumbers,
  (or Human_SpectralData.csv)          columns = spectra named "<subject> rep<k>".
  Human_clinical_and_PCR_data.xlsx     PCR results and collection site of each subject (same record).

Outputs:
  t2d_saliva.csv               T2D vs Control.
  covid_original.csv           COVID_POS vs COVID_NEG, all sites (as in the original study).
  covid_nsw_pcr.csv            NSW only: follow-up PCR positive (PCR_POS) vs negative (PCR_NEG).
  covid_site_nsw_vs_tpch.csv   COVID-positive subjects only: NSW vs TPCH (label = site).
A dataset whose input files are missing is skipped.
"""
from pathlib import Path
import pandas as pd


def save(df, labels, path, description):
    out = df.copy()
    out.insert(0, "label", labels)
    out.to_csv(path, index=False)
    subj = out.drop_duplicates("subject")
    print(f"{path}: {description}\n  {len(out)} spectra, subjects per class: {subj.label.value_counts().to_dict()}")


# ----------------------------------------------------------------------------- T2D
if Path("diabetes.csv").exists():
    d = pd.read_csv("diabetes.csv")
    lab = d.pop("label").astype(str).str.strip()
    if set(lab.unique()) != {"DM2", "control"} and set(lab.str.lower().unique()) != {"dm2", "control"}:
        raise ValueError(f"diabetes.csv: unexpected labels {sorted(lab.unique())}")
    def _is_num(c):
        try:
            float(c)
            return True
        except ValueError:
            return False
    spec = d[[c for c in d.columns if _is_num(c)]]
    meta = pd.DataFrame({"subject": [f"T2D_{i:04d}" for i in range(len(d))], "site": "single_site"})
    save(pd.concat([meta, spec.reset_index(drop=True)], axis=1),
         lab.str.lower().map({"dm2": "T2D", "control": "Control"}).to_numpy(),
         "t2d_saliva.csv", "type 2 diabetes vs control (one spectrum per subject)")
else:
    print("diabetes.csv not found: T2D dataset skipped.")

# ----------------------------------------------------------------------------- COVID-19 (Australia)
spectra = next((p for p in ["Australia_Human_SpectralData.csv", "Human_SpectralData.csv"] if Path(p).exists()), None)
if spectra and Path("Human_clinical_and_PCR_data.xlsx").exists():
    raw = pd.read_csv(spectra)
    wn = raw.iloc[:, 0].astype(float).to_numpy()
    spec = raw.iloc[:, 1:].T
    spec.columns = [f"{v:.4f}" for v in wn]
    spec.index = [str(c).strip() for c in spec.index]
    subject = [c.split(" rep")[0].strip() for c in spec.index]
    cl = pd.read_excel("Human_clinical_and_PCR_data.xlsx", sheet_name="Clinical data")
    cl.columns = [str(c).strip() for c in cl.columns]
    cl["Sample name"] = cl["Sample name"].astype(str).str.strip()
    cl = cl.set_index("Sample name")
    missing = sorted(set(subject) - set(cl.index))
    if missing:
        raise ValueError(f"Subjects without clinical data: {missing[:10]}")
    meta = pd.DataFrame({"subject": subject, "site": cl.loc[subject, "Source"].astype(str).str.strip().to_numpy()},
                        index=spec.index)
    group = cl.loc[subject, "Enrolment group"].astype(str).str.strip().str.replace(".", "_", regex=False).to_numpy()
    fu = cl.loc[subject, "Follow-up PCR test"].astype(str).str.strip().to_numpy()
    base = pd.concat([meta, spec], axis=1).reset_index(drop=True)
    print("\nCOVID-19 subjects per site and enrolment group:")
    sub = pd.DataFrame({"subject": base.subject, "site": base.site, "group": group}).drop_duplicates("subject")
    print(pd.crosstab(sub.site, sub.group))

    save(base, group, "covid_original.csv", "COVID_POS vs COVID_NEG, all sites (confounded with site)")
    m = (base.site == "NSW").to_numpy() & pd.Series(fu).isin(["POS", "NEG"]).to_numpy()
    save(base[m], ("PCR_" + pd.Series(fu[m])).to_numpy(), "covid_nsw_pcr.csv",
         "NSW only, follow-up PCR positive vs negative (no site confounding)")
    m = (group == "COVID_POS") & base.site.isin(["NSW", "TPCH"]).to_numpy()
    save(base[m], base.site[m].to_numpy(), "covid_site_nsw_vs_tpch.csv",
         "COVID-positive subjects only, NSW vs TPCH (label = site; acquisition audit)")
else:
    print("COVID-19 files not found: COVID datasets skipped.")
