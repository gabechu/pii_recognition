"""CLI support for running PII validation pipeline."""
import argparse

from pii_recognition.pipelines.pii_validation_pipeline import exec_pipeline

parser = argparse.ArgumentParser(prog="pii_validation_pipeline")
parser.add_argument("--config_yaml", help="Path of config yaml file")
args = parser.parse_args()

exec_pipeline(args.config_yaml)
