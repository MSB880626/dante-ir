"""Reproduces the analyses of the paper.
1) Put diabetes.csv, Australia_Human_SpectralData.csv and Human_clinical_and_PCR_data.xlsx in this folder.
2) python prepare_datasets.py
3) python run_paper.py                      (full run: about 10 hours on 2 CPU cores)
   python run_paper.py T2D_saliva           (nested validation only for the listed datasets; finished
                                            datasets are reused, so the run can be split or resumed)"""
import sys
from dante_ir import run_config

run_config("study_config.json", run_cv=sys.argv[1:] or None)
