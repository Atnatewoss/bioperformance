"""
BioPerformance CLI — orchestrates train / shap / explain steps.

Usage:
    python src/main.py train
        Train and evaluate XGBoost models (group + per-athlete).
        Reads:  data/processed/features.csv
        Writes: models/readiness_group_model.json

    python src/main.py shap
        Compute per-item SHAP values for the test set.
        Reads:  data/processed/features.csv
        Writes: data/processed/shap_per_item.json
                data/processed/shap_beeswarm_<item>.png

    python src/main.py explain [--sample N] [--pln-cache DIR]
        Run PLN explanations on the SHAP output.
        Reads:  data/processed/shap_per_item.json
        Writes: data/processed/pln_explanations.json

        --sample N       number of example rows to print per item (default 3)
        --pln-cache DIR  path to a local trueagi-io/PLN clone
                         (default: ~/.cache/bioperformance/pln,
                          auto-populated on first run)

Full pipeline:
    python src/main.py train
    python src/main.py shap
    python src/main.py explain
"""

import argparse
import os

from readiness_model import main as train_models
from shap_analysis import run_shap_analysis


def main():
    parser = argparse.ArgumentParser(
        description='BioPerformance pipeline CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='step', required=True)

    sub.add_parser('train', help='train and evaluate XGBoost models')
    sub.add_parser('shap',  help='compute per-item SHAP values')

    explain_parser = sub.add_parser('explain', help='run PLN explanations')
    explain_parser.add_argument(
        '--sample', type=int, default=3,
        help='number of example athlete-days to print per item (default 3)',
    )
    explain_parser.add_argument(
        '--pln-cache', type=str, default=None,
        metavar='DIR',
        help=(
            'path to a local trueagi-io/PLN clone '
            '(default: ~/.cache/bioperformance/pln, auto-populated on first run)'
        ),
    )

    args = parser.parse_args()

    if args.step == 'train':
        train_models()

    elif args.step == 'shap':
        run_shap_analysis()

    elif args.step == 'explain':
        from pln.pln_explain import run_explain, PLN_CACHE_DIR
        pln_cache = args.pln_cache or PLN_CACHE_DIR
        run_explain(sample=args.sample, pln_cache_dir=pln_cache)


if __name__ == '__main__':
    main()
