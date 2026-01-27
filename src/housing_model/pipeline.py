import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from .features import ratio_transformer, log_transformer

logger = logging.getLogger(__name__)

def make_preprocessing() -> ColumnTransformer:
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        ratio_transformer,
        StandardScaler(),
    )

    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        log_transformer,
        StandardScaler(),
    )

    default_num = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    cat_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=True),
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("bedrooms", ratio_pipeline, ["total_bedrooms", "total_rooms"]),
            ("rooms_per_house", ratio_pipeline, ["total_rooms", "households"]),
            ("people_per_house", ratio_pipeline, ["population", "households"]),
            ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]),
            ("cat", cat_pipeline, ["ocean_proximity"]),
        ],
        remainder=default_num,
        verbose_feature_names_out=False,
    )

    return preprocessing

def make_model(random_state: int, n_jobs: int):
    return RandomForestRegressor(
        random_state=random_state,
        n_jobs=n_jobs,
    )

def build_pipeline(random_state: int, n_jobs: int):
    preprocessing = make_preprocessing()
    model = make_model(random_state=random_state, n_jobs=n_jobs)
    return make_pipeline(preprocessing, model)
