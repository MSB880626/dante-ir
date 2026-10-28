# -*- coding: utf-8 -*-
"""
Reading of spectral tables in the two common layouts:

  wide        one row per spectrum, numeric wavenumber column headers, plus metadata columns
              (class label, optional subject and batch/site columns).
  transposed  one row per wavenumber (first column) and one column per spectrum; the metadata (class,
              subject, batch) must then be given in a separate table with one row per spectrum.

Supported files: .csv, .tsv, .txt (comma, tab or semicolon separated) and .xlsx / .xls.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def _is_number(x):
    try:
        float(str(x).strip())
        return True
    except ValueError:
        return False


def _layout(df):
    numeric_headers = sum(_is_number(c) for c in df.columns)
    if numeric_headers >= 50:
        return "wide"
    first = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    if first.notna().mean() > 0.95 and len(df) >= 50:
        return "transposed"
    raise ValueError("Could not recognize the layout: use one row per spectrum with numeric wavenumber headers, "
                     "or one row per wavenumber with the wavenumbers in the first column.")


def _to_absorbance(X, units):
    if units == "absorbance":
        return X, "absorbance"
    if units == "transmittance_percent" or (units == "auto" and np.nanmedian(X) > 1.5 and np.nanmax(X) <= 105):
        T = np.clip(X / 100.0, 1e-6, None)
        return -np.log10(T), "converted from transmittance (%) to absorbance"
    return X, "absorbance (assumed)"


def _match_ids(spec_ids, meta_ids):
    """Exact match; otherwise, try removing a trailing replicate tag such as ' rep2', '_r3' or '-2'."""
    meta = {str(m).strip(): str(m).strip() for m in meta_ids}
    out = []
    for s in spec_ids:
        s = str(s).strip()
        if s in meta:
            out.append(meta[s])
            continue
        base = re.sub(r"[\s_\-]*(rep|r|replicate)?[\s_\-]*\d+$", "", s, flags=re.I).strip()
        out.append(meta.get(base))
    return out


def load_spectra(spectra_path, label_column, metadata_path=None, id_column=None, subject_column=None,
                 batch_column=None, units="auto"):
    """Returns a dict with wn, X (spectra x wavenumbers, absorbance), y_raw, g (subjects or None),
    batch (or None), ids and a short description of what was read."""
    df = read_table(spectra_path)
    layout = _layout(df)
    meta = read_table(metadata_path) if metadata_path else None

    if layout == "wide":
        spec_cols = [c for c in df.columns if _is_number(c)]
        wn = np.array([float(c) for c in spec_cols])
        X = df[spec_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        info = df.drop(columns=spec_cols)
        ids = (info[id_column].astype(str).to_numpy() if id_column and id_column in info
               else np.array([f"spectrum_{i + 1}" for i in range(len(df))]))
        if meta is not None:
            key = id_column or meta.columns[0]
            if key not in info.columns:
                raise ValueError(f"To join the metadata file, the spectra file needs the id column '{key}'.")
            m = meta.set_index(meta[key].astype(str).str.strip())
            matched = _match_ids(info[key].astype(str), m.index)
            info = pd.concat([info.reset_index(drop=True),
                              m.reindex(matched).reset_index(drop=True).drop(columns=[key], errors="ignore")], axis=1)
    else:
        wn = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy(float)
        ok = np.isfinite(wn)
        spectra = df.loc[ok, df.columns[1:]].apply(pd.to_numeric, errors="coerce")
        wn = wn[ok]
        ids = np.array([str(c).strip() for c in spectra.columns])
        X = spectra.to_numpy(float).T
        if meta is None:
            raise ValueError("The spectra file has one row per wavenumber; a metadata file with one row per "
                             "spectrum (id, class and, optionally, subject and batch) is required.")
        key = id_column or meta.columns[0]
        m = meta.set_index(meta[key].astype(str).str.strip())
        matched = _match_ids(ids, m.index)
        if any(v is None for v in matched):
            missing = [i for i, v in zip(ids, matched) if v is None][:5]
            raise ValueError(f"Spectra without a matching row in the metadata file (e.g. {missing}).")
        info = m.reindex(matched).reset_index(drop=True)

    if label_column not in info.columns:
        raise ValueError(f"Class column '{label_column}' not found. Available columns: {list(info.columns)[:20]}")
    for col in [subject_column, batch_column]:
        if col and col not in info.columns:
            raise ValueError(f"Column '{col}' not found. Available columns: {list(info.columns)[:20]}")
    keep = info[label_column].notna().to_numpy() & np.isfinite(X).all(axis=1)
    dropped = int((~keep).sum())
    order = np.argsort(wn)
    X, unit_note = _to_absorbance(X[keep][:, order], units)
    out = {"wn": wn[order], "X": X, "y_raw": info.loc[keep, label_column].astype(str).str.strip().to_numpy(),
           "g": info.loc[keep, subject_column].astype(str).to_numpy() if subject_column else None,
           "batch": info.loc[keep, batch_column].astype(str).to_numpy() if batch_column else None,
           "ids": ids[keep], "info": info.loc[keep].reset_index(drop=True)}
    out["read_info"] = (f"{layout} layout, {keep.sum()} spectra, {len(wn)} wavenumbers "
                        f"({wn.min():.0f}-{wn.max():.0f} cm-1), {unit_note}"
                        + (f"; {dropped} spectra dropped (missing class or non-numeric values)" if dropped else ""))
    return out
