import logging
from pathlib import Path

from sklearn.model_selection import GridSearchCV, cross_val_score

from .config import AppConfig
from .pipeline import build_pipeline
from .evaluate import regression_metrics
from .io import write_json
from .profiling import build_training_profile
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


def train_and_select(cfg: AppConfig, X_train, y_train):
    pipe = build_pipeline(random_state=cfg.model.random_state, n_jobs=cfg.model.n_jobs)

    cv_scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=cfg.model.n_jobs,
    )
    logger.info("Sanity CV RMSE: mean=%.4f std=%.4f", (-cv_scores).mean(), (-cv_scores).std())

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


def fit(cfg: AppConfig, run_id: str, X_train, y_train, X_test, y_test) -> dict:
    model, meta = train_and_select(cfg, X_train, y_train)

    # Evaluate on test
    y_pred = model.predict(X_test)
    metrics = regression_metrics(y_test, y_pred)

    # Build & persist training profile (for drift checks in service)
    profile = build_training_profile(X_train)
    profile_path = "artifacts/reports/training_profile.json"
    write_json(profile_path, profile)

    # Save model into registry + activate
    registry = ModelRegistry(Path("artifacts/models/registry"))
    model_path = str(registry.save(model, run_id=run_id))
    active_path = str(registry.set_active(run_id))

    # Persist metrics for debugging/ops
    metrics_path = "artifacts/reports/metrics.json"
    write_json(metrics_path, {**metrics, **meta, "run_id": run_id, "model_path": model_path})

    logger.info(
        "train_done run_id=%s rmse=%.4f mae=%.4f r2=%.4f model=%s active=%s",
        run_id, metrics["rmse"], metrics["mae"], metrics["r2"], model_path, active_path
    )

    return {
        "run_id": run_id,
        "registry_model_path": model_path,
        "active_model_path": active_path,
        "training_profile_path": profile_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "meta": meta,
    }
