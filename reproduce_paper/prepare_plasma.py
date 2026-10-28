#!/usr/bin/env python3
"""Converts the plasma COVID-19 severity dataset (Zenodo 4805258) to the DANTE-IR wide format.

Needs, in this folder:
  - the extracted folders "Data for predictive model" and "Data for testing the model"
    (in Colab: !unrar x -o+ "Data for predictive model.rar"   and   !unrar x -o+ "Data for testing the model.rar")
  - plasma_clinical.csv: clinical tables S1-S4 of the supporting information of the dataset (plasma id,
    hospital registration number, age, sex, diabetes, hypertension, severity, training/test set).

Outputs (one row per spectrum; columns label, subject, sample, day, replicate, age, sex_male, diabetes,
hypertension, then the wavenumbers):
  plasma_train.csv             training set; subject = hospital registration number, so that repeated plasma
                               samples of the same patient are never split between folds. Patients listed in
                               both severity groups are excluded.
  plasma_test_independent.csv  blind test set without the patients that also appear in the training set.
  plasma_test_all.csv          the complete blind test set, as used in the original study."""
import glob
import os
import re

import numpy as np
import pandas as pd

clin = pd.read_csv("plasma_clinical.csv", dtype={"reg": str})


def read_folder(pattern):
    rows, spectra, wn = [], [], None
    for f in sorted(glob.glob(pattern, recursive=True)):
        m = re.match(r"(P\d+)_\d{8}_R(\d+)_(\d{4}-\d{2}-\d{2})T", os.path.basename(f))
        x = pd.read_csv(f)
        if wn is None:
            wn = x["Wavenumber"].to_numpy()
        spectra.append(np.interp(wn, x["Wavenumber"].to_numpy(), x["Intensity"].to_numpy()))
        rows.append({"sample": m.group(1), "day": m.group(3), "replicate": int(m.group(2))})
    if not rows:
        raise SystemExit(f"No spectra found for {pattern}: extract the .rar files in this folder first.")
    meta = pd.DataFrame(rows).merge(clin, left_on="sample", right_on="plasma", how="left")
    if meta.reg.isna().any():
        raise ValueError(f"Samples without clinical data: {meta[meta.reg.isna()]['sample'].unique()[:5]}")
    meta["label"] = meta.severity
    meta["subject"] = "R" + meta.reg
    meta["sex_male"] = (meta.sex == "M").astype(int)
    meta = meta[["label", "subject", "sample", "day", "replicate", "age", "sex_male", "diabetes", "hypertension"]]
    return pd.concat([meta, pd.DataFrame(np.vstack(spectra), columns=[f"{v:.4f}" for v in wn])], axis=1)


train = read_folder("**/Data for predictive model/**/CSV Format/*.csv")
test = read_folder("**/Data for testing the model/**/CSV Format/*.csv")

both = train.groupby("subject").label.nunique()
both = both[both > 1].index.tolist()
train = train[~train.subject.isin(both)]
overlap = sorted(set(test.subject) & set(train.subject))
train.to_csv("plasma_train.csv", index=False)
test.to_csv("plasma_test_all.csv", index=False)
test[~test.subject.isin(overlap)].to_csv("plasma_test_independent.csv", index=False)


def describe(df, name):
    s = df.drop_duplicates("subject")
    print(f"{name}: {len(df)} spectra, {df['sample'].nunique()} plasma samples, {len(s)} patients "
          f"{s.label.value_counts().to_dict()}")


describe(train, "plasma_train.csv")
describe(test, "plasma_test_all.csv")
describe(test[~test.subject.isin(overlap)], "plasma_test_independent.csv")
print(f"Patients excluded from training because they appear in both severity groups: {both}")
print(f"Test patients that also appear in the training set (excluded from the independent test): {overlap}")
