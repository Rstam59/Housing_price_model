import logging
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)

def load_housing(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f'Dataset is empty: {csv_path}')
    return df 

def add_income_cat(df: pd.DataFrame, 
                   income_col: str, 
                   bins: list[float], 
                   labels: list[int]) -> pd.DataFrame:
    if income_col not in df.columns:
        raise KeyError(f'Missing stratify column: {income_col}')
    out = df.copy()
    out['income_cat'] = pd.cut(out[income_col], bins = bins, labels = labels)
    
    if out['income_cat'].isna().any():
        n = int(out['income_cat'].isna().sum())
        raise ValueError(f'income_cat has {n} NANs (bins may not cover all values)')
    return out


def stratified_split(
        df: pd.DataFrame,
        target: str,
        stratify_col: str,
        bins: list[float],
        labels: list[int],
        test_size: float,
        random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target not in df.columns:
        raise KeyError(f'Missing target column: {target}')
    
    df2 = add_income_cat(df, stratify_col, bins, labels)

    train_df, test_df = train_test_split(
        df2, 
        test_size= test_size,
        stratify= df2['income_cat'],
        random_state= random_state
    )

    for s in (train_df, test_df):
        s.drop(columns = ['income_cat'], inplace = True)

    X_train = train_df.drop(columns = [target])
    y_train = train_df[target].copy()

    X_test = test_df.drop(columns = [target])
    y_test = test_df[target].copy()

    logger.info("Split sizes: train=%d test=%d", len(train_df), len(test_df))

    return X_train, X_test, y_train, y_test