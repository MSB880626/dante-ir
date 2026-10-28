# -*- coding: utf-8 -*-
"""Command line: dante-ir --spectra data.csv --label class --positive T2D [options]
                 dante-ir --config study.json"""
import argparse

from .run import run, run_config, MODES


def main():
    p = argparse.ArgumentParser(prog="dante-ir", description="Acquisition audit and interpretable band descriptor "
                                "for infrared spectra of biofluids.")
    p.add_argument("--config", help="JSON file describing several datasets (overrides the other options)")
    p.add_argument("--spectra", help="spectral table (csv, tsv, txt, xlsx)")
    p.add_argument("--metadata", help="metadata table, required if the spectra file has one row per wavenumber")
    p.add_argument("--id", dest="id_column", help="spectrum id column used to join the metadata")
    p.add_argument("--label", help="class column")
    p.add_argument("--positive", help="positive (disease) class")
    p.add_argument("--subject", help="subject column (replicate spectra of the same person)")
    p.add_argument("--batch", help="batch / session / site column (enables the batch audit)")
    p.add_argument("--sample", default="saliva", choices=["saliva", "serum", "plasma", "blood", "urine", "other"])
    p.add_argument("--crystal", default="diamond", choices=["diamond", "ZnSe", "Ge", "Si", "transmission"])
    p.add_argument("--mode", default="audit", choices=MODES)
    p.add_argument("--out", default="dante_ir_results")
    p.add_argument("--name", default="dataset")
    a = p.parse_args()
    if a.config:
        run_config(a.config)
        return
    if not (a.spectra and a.label and a.positive):
        p.error("--spectra, --label and --positive are required (or use --config)")
    run(a.spectra, a.label, a.positive, metadata=a.metadata, id_column=a.id_column, subject_column=a.subject,
        batch_column=a.batch, sample_type=a.sample, crystal=a.crystal, mode=a.mode, output_dir=a.out,
        dataset_name=a.name)


if __name__ == "__main__":
    main()
