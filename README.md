# DANTE-IR

**Diagnostic Audit aNd inTerpretable band dEscriptors for InfraRed spectra of biofluids**

DANTE-IR answers two questions about an infrared (ATR-FTIR) dataset of two groups, for example patients and
controls:

1. **Do the groups differ because of the samples or because of how they were measured?** Models restricted to
   spectral regions without sample bands (negative controls), the noise level, and the association between the
   class and the acquisition batch or collection site reveal differences in instrument, environment, session or
   handling that make classification results look better than they are.
2. **What changes between the groups, in chemical terms?** An interpretable descriptor (band areas, positions,
   widths, shapes, ratios, second-derivative components) of proteins, lipids, phosphates, carbohydrates and
   hydration, with a plain-language report.

## Use it without programming (Google Colab)

Open `notebooks/DANTE_IR_Colab.ipynb` in Google Colab, run the four cells in order and fill in the form:
spectra file, class column, positive class, person column (if there are replicate spectra), batch/site column
(optional), sample type and ATR crystal. A ZIP with the report, figures and tables is downloaded at the end.

## Use it from Python or the command line

```bash
pip install git+https://github.com/YOUR_USER/dante-ir
dante-ir --spectra examples/example_spectra.csv --label group --positive disease --subject subject --batch batch --mode audit
```

```python
import dante_ir
dante_ir.run("examples/example_spectra.csv", "group", "disease", subject_column="subject",
             batch_column="batch", sample_type="saliva", crystal="diamond", mode="audit")
```

## External validation and region-level interpretation

```python
import dante_ir
# Train on one table, evaluate once on an independent one (patient-level metrics with confidence intervals)
dante_ir.external_validation("train.csv", "test.csv", "label", "Severe", subject_column="subject",
                             covariates=["age", "diabetes"], sample_type="plasma")
# Null-calibrated region attribution, null reference regions, linear SHAP and stability
dante_ir.interpret_regions("train.csv", "label", "Severe", subject_column="subject", sample_type="plasma")
```

## Input data

- **One row per spectrum** (recommended): numeric wavenumber column headers (cm⁻¹), a class column, and
  optionally person and batch/site columns. CSV, TSV, TXT or Excel.
- **One row per wavenumber**: first column = wavenumbers, one column per spectrum, plus a metadata table with
  one row per spectrum (first column = spectrum name). Replicate tags such as `P01 rep2` are matched to `P01`.
- Spectral range: at least 900–3100 cm⁻¹; up to ~4000 cm⁻¹ enables the negative-control region.
  Absorbance, or transmittance in % (converted automatically).

## Modes

| Mode | Time (2 CPU cores) | Content |
|---|---|---|
| `audit` | minutes | negative controls, noise check, batch/site audit, full spectrum vs. descriptor |
| `exploratory` | ~1 h | adds second derivative, binning and descriptor variants, region importance |
| `full` | several hours | all reference and equal-budget methods, non-inferiority tests, robustness, permutation tests |
| `quick` | minutes | smoke test of the full pipeline |

## Main outputs

`report.txt` (summary verdict, results and glossary), `fig6_acquisition_audit.png`, `fig1_spectra_bands.png`,
`fig2_performance_robustness.png`, `fig3_effect_map.png`, `cv_summary.csv`, `feature_effects.csv`,
`batch_audit.csv`, `descriptors_<dataset>.csv`, `variable_glossary.csv`.

## Method notes and limitations

- Diamond ATR: the multiphonon region (1800–2700 cm⁻¹) is excluded; other crystals keep it.
- Nested cross-validation grouped by person; hyperparameters tuned in the inner loop.
- Noise harmonization is applied only when the noise level separates the classes.
- Sample-type modules (urine urea bands, salivary thiocyanate with non-diamond crystals) are literature-based
  and **have not been validated on data**.
- A clean audit does not prove the absence of confounding (for example, differences in sample handling that
  affect the biological bands); always compare groups within batches/sites when the design allows it.

## Reproducing the paper

See `reproduce_paper/README.md`.

## License and citation

MIT license. Please cite the software (see `CITATION.cff`) and the associated article.
