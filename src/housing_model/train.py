import logging
import joblib
from sklearn.model_selection import GridSearchCV, cross_val_score

from housing_model.profiling import build_training_profile

from .config import AppConfig
from .pipeline import build_pipeline
from .evaluate import regression_metrics
from .io import write_json, ensure_parent

logger = logging.getLogger(__name__)

def train_and_select(cfg: AppConfig, X_train, y_train):
    pipe = build_pipeline(
        random_state=cfg.model.random_state,
        n_jobs=cfg.model.n_jobs,
    )

    # quick sanity CV (optional)
    cv_scores = cross_val_score(
        pipe, X_train, y_train,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=cfg.model.n_jobs,
    )
    logger.info("Sanity CV RMSE: mean=%.4f std=%.4f",
                (-cv_scores).mean(), (-cv_scores).std())

    if not cfg.grid.enabled:
        pipe.fit(X_train, y_train)
        meta = {"cv_rmse_mean": float((-cv_scores).mean()), "cv_rmse_std": float((-cv_scores).std())}
        return pipe, meta

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=cfg.grid.param_grid,
        cv=cfg.grid.cv,
        scoring=cfg.grid.scoring,
        n_jobs=cfg.model.n_jobs,
        refit=True,
    )
    gs.fit(X_train, y_train)

    logger.info("Best params: %s", gs.best_params_)
    best = gs.best_estimator_

    meta = {
        "best_params": gs.best_params_,
        "best_cv_score": float(gs.best_score_),
        "sanity_cv_rmse_mean": float((-cv_scores).mean()),
        "sanity_cv_rmse_std": float((-cv_scores).std()),
    }
    return best, meta

def save_model(model, path: str) -> None:
    ensure_parent(path)
    joblib.dump(model, path)

def fit(cfg: AppConfig, X_train, y_train, X_test, y_test) -> dict:
    model, meta = train_and_select(cfg, X_train, y_train)

    # test evaluation
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    profile = build_training_profile(X_train)

    # persist
    save_model(model, cfg.output.model_path)
    write_json(cfg.output.metrics_path, {**metrics, **meta})
    write_json("artifacts/reports/training_profile.json", profile)

    return {"model_path": cfg.output.model_path,
            "metrics": metrics, 
            "meta": meta,
            "training_profile_path": "artifacts/reports/training_profile.json"}
