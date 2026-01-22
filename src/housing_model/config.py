from dataclasses import dataclass
from pathlib import Path 
import yaml 

@dataclass(frozen=True)
class TrainConfig:
    csv_path: str
    target: str 
    stratify_col: str
    test_size: float 
    random_state: int 
    income_cat_bins: list[float]
    income_cat_labels: list[int]


@dataclass(frozen=True)
class ModelConfig:
    random_state: int 
    n_jobs: int 



@dataclass(frozen=True)
class GridConfig:
    enabled: bool 
    cv: int 
    scoring: str
    param_grid: dict 


@dataclass(frozen=True)
class OutputConfig:
    artifacts_dir: str 
    model_path: str 
    metrics_path: str 
    manifest_path: str 


@dataclass
class AppConfig:
    data: TrainConfig
    model: ModelConfig
    grid: GridConfig
    output: OutputConfig


def load_config(path: str) -> AppConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8'))

    data = payload['data']
    model = payload['model']
    grid = payload['grid']
    output = payload['output']

    return AppConfig(
        data=TrainConfig(
            csv_path=data["csv_path"],
            target=data["target"],
            stratify_col=data["stratify_col"],
            test_size=float(data["test_size"]),
            random_state=int(data["random_state"]),
            income_cat_bins=[float(x) if x != ".inf" else float("inf") for x in data["income_cat_bins"]],
            income_cat_labels=[int(x) for x in data["income_cat_labels"]],
        ),
        model=ModelConfig(
            random_state=int(model["random_state"]),
            n_jobs=int(model["n_jobs"]),
        ),
        grid=GridConfig(
            enabled=bool(grid["enabled"]),
            cv=int(grid["cv"]),
            scoring=str(grid["scoring"]),
            param_grid=grid["param_grid"] or {},
        ),
        output=OutputConfig(
            artifacts_dir=output["artifacts_dir"],
            model_path=output["model_path"],
            metrics_path=output["metrics_path"],
            manifest_path=output["manifest_path"],
        ),
    )