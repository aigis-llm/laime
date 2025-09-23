import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from fastapi import HTTPException
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

T = TypeVar("T")


def get_backend(model_to_get: str, model_class: str, model_type: type[T]) -> T:
	models = get_models()
	backend = None

	if not model_backends.get(model_to_get):
		model = models.get(model_to_get)
		if model is None:
			raise HTTPException(
				status_code=422, detail={"message": f"No model {model_to_get}."}
			)
		if not issubclass(model[0], model_type):
			raise HTTPException(
				status_code=422,
				detail={"message": f"{model_to_get} is not an {model_class} model."},
			)
		backend = model[0](model[1])
		model_backends[model_to_get] = backend
	else:
		backend = model_backends[model_to_get]
		if not isinstance(backend, model_type):
			raise HTTPException(
				status_code=422,
				detail={"message": f"{model_to_get} is not an {model_class} model."},
			)

	return backend
