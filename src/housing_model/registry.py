from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib


@dataclass(frozen=True)
class ModelRegistry:
    registry_dir: Path  # artifacts/models/registry

    def ensure(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, run_id: str) -> Path:
        # keep file extension stable for tools
        return self.registry_dir / f"{run_id}.joblib"

    def active_symlink(self) -> Path:
        return self.registry_dir / "active"

    def save(self, model, run_id: str) -> Path:
        self.ensure()
        path = self.model_path(run_id)
        joblib.dump(model, path)
        return path

    def set_active(self, run_id: str) -> Path:
        self.ensure()
        target = self.model_path(run_id)
        if not target.exists():
            raise FileNotFoundError(f"Model not found in registry: {target}")

        link = self.active_symlink()

        # On Windows symlinks can be painful; fallback to copy.
        if link.exists() or link.is_symlink():
            link.unlink()

        try:
            os.symlink(target.name, link)  # relative symlink inside registry dir
        except OSError:
            shutil.copy2(target, link)

        return link

    def resolve_active(self) -> Path:
        link = self.active_symlink()
        if not link.exists():
            raise FileNotFoundError("No active model set in registry.")
        # If it's a symlink, resolve; if it's a copied file fallback, use it directly
        return link.resolve() if link.is_symlink() else link
