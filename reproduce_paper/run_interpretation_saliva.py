"""Region-level interpretation of the salivary cohorts (run after prepare_datasets.py)."""
import dante_ir

dante_ir.interpret_regions("covid_nsw_pcr.csv", "label", "PCR_POS", subject_column="subject",
                           sample_type="saliva", crystal="diamond", output_dir="interpretation_covid_nsw",
                           dataset_name="COVID_NSW_PCR")
dante_ir.interpret_regions("t2d_saliva.csv", "label", "T2D", subject_column="subject",
                           sample_type="saliva", crystal="diamond", output_dir="interpretation_t2d",
                           dataset_name="T2D_saliva")
