"""CLI support for running Pakkr evaluation pipeline."""
import argparse

from pii_recognition.evaluation.pakkr_pipeline import execute_evaluation_pipeline

parser = argparse.ArgumentParser(prog="pakkr_evaluation")
parser.add_argument("--config_yaml", help="Path of config yaml file")
args = parser.parse_args()

execute_evaluation_pipeline(args.config_yaml)
