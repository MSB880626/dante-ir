"""Reproduces the plasma analyses of the paper (COVID-19 severity, Zenodo 4805258).
1) Extract "Data for predictive model.rar" and "Data for testing the model.rar" in this folder.
2) python prepare_plasma.py
3) python run_plasma.py              (nested cross-validation, external validation and region interpretation)"""
import dante_ir

dante_ir.run("plasma_train.csv", "label", "Severe", subject_column="subject", batch_column="day",
             sample_type="plasma", crystal="diamond", mode="full", output_dir="results_plasma",
             dataset_name="Plasma_COVID_severity",
             description="COVID-19 severe vs non-severe, ATR-FTIR of plasma, training set grouped by patient")
dante_ir.external_validation("plasma_train.csv", "plasma_test_independent.csv", "label", "Severe",
                             subject_column="subject", covariates=["age", "sex_male", "diabetes", "hypertension"],
                             sample_type="plasma", crystal="diamond", output_dir="external_validation_plasma")
dante_ir.interpret_regions("plasma_train.csv", "label", "Severe", subject_column="subject",
                           sample_type="plasma", crystal="diamond", output_dir="interpretation_plasma",
                           dataset_name="Plasma_COVID_severity")
