import argparse
import logging
import platform
import sklearn


from .logging_setup import setup_logging
from .config import load_config
from .data import load_housing, stratified_split
from .train import fit
from .io import write_json

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default = "configs/train.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

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
        'data_path': cfg.data.csv_path,
        'model_path': result['model_path'],
        'metrics': result['metrics'],
        'meta': result['meta'],
    }

    write_json(cfg.output.manifest_path, manifest)

    logger.info('Done. Test RMSE=%.4f MAE=%.4f R2=%.4f',
                result['metrics']['rmse'], result['metrics']['mae'], 
                result['metrics']['r2'])
    
if __name__ == '__main__':
    main()