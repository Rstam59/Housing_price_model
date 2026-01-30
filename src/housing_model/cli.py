import argparse
import logging
import platform
import sklearn
import yaml 
from pathlib import Path


from .logging_setup import setup_logging
from .config import load_config
from .data import load_housing, stratified_split
from .train import fit
from .io import write_json
from .versioning import sha256_file, sha256_bytes, sha256_json, short_hash

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = "configs/train.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

        # Hashes for reproducibility
    config_text = Path(args.config).read_text(encoding="utf-8")
    config_hash = sha256_bytes(config_text.encode("utf-8"))
    data_hash = sha256_file(cfg.data.csv_path)

    df = load_housing(cfg.data.csv_path)

    X_train, X_test, y_train, y_test = stratified_split(
        df = df,
        target= cfg.data.target,
        stratify_col= cfg.data.stratify_col,
        bins = cfg.data.income_cat_bins,
        labels= cfg.data.income_cat_labels,
        test_size= cfg.data.test_size,
        random_state = cfg.data.random_state
    )

    result = fit(cfg, X_train, y_train, X_test, y_test)

    manifest = {
        'python': platform.python_version(),
        'sklearn': sklearn.__version__,
        'config_path': args.config,
        "config_hash": config_hash,
        'data_path': cfg.data.csv_path,
        "data_hash": data_hash,
        'model_path': result['model_path'],
        "training_profile_path": result["training_profile_path"],
        'metrics': result['metrics'],
        'meta': result['meta'],
    }

    manifest["run_id"] = f"{short_hash(config_hash)}_{short_hash(data_hash)}"
    write_json(cfg.output.manifest_path, manifest)

    logger.info(
        "Done. run_id=%s Test RMSE=%.4f MAE=%.4f R2=%.4f",
        manifest["run_id"],
        result["metrics"]["rmse"],
        result["metrics"]["mae"],
        result["metrics"]["r2"],
    )
    
if __name__ == '__main__':
    main()