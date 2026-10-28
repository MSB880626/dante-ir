# Reproducing the paper

## Salivary cohorts (type 2 diabetes and COVID-19)
1. Download into this folder `diabetes.csv` (type 2 diabetes saliva dataset, Comput. Struct. Biotechnol. J. 2022),
   and `Australia_Human_SpectralData.csv` and `Human_clinical_and_PCR_data.xlsx` (Zenodo record 5703689).
2. `python prepare_datasets.py` converts both datasets to the common format.
3. `python run_paper.py` runs nested cross-validation for the four salivary comparisons
   (`python run_paper.py T2D_saliva` runs one dataset; finished datasets are reused).
4. `python run_interpretation_saliva.py` runs the null-calibrated region interpretation.

## Plasma cohort (COVID-19 severity)
1. Download `Data for predictive model.rar` and `Data for testing the model.rar` from Zenodo record 4805258 and
   extract them here. `plasma_clinical.csv` contains the clinical tables S1–S4 of the dataset's supporting information.
2. `python prepare_plasma.py` builds the training set grouped by patient and the independent test set.
3. `python run_plasma.py` runs nested cross-validation, external validation and region interpretation.

## Legacy scripts
`legacy/` contains the stand-alone scripts used to produce the salivary cross-validation results
(`ftir_descriptor.py`, `make_report.py`, `prepare_datasets.py`). The package implements the same procedure.

Approximate run times on Google Colab (2 CPU cores): type 2 diabetes about 6 h; the three COVID-19 salivary
comparisons about 4 h; plasma about 3 h; region interpretation from minutes (plasma) to about 1 h (type 2 diabetes).
