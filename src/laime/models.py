import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from laime.backends import backends
from laime.backends.abc import Backend
from laime.settings import get_settings


@lru_cache
def get_models():
	models: dict[str, tuple[type[Backend[Any]], BaseModel]] = {}  # pyright: ignore [reportExplicitAny]

	for path in Path(get_settings().models_dir).glob("**/*.toml"):
		with open(path, "rb") as modelf:
			modeld = tomllib.load(modelf)
			backend = backends[modeld["meta"]["backend"]]
			models[modeld["meta"]["name"]] = (
				backend,
				backend.backend_config_model.model_validate(modeld["config"]),
			)

	return models


model_backends: dict[str, Backend[Any]] = {}  # pyright: ignore [reportExplicitAny]
